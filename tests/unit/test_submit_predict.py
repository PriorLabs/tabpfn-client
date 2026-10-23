#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""`submit_predict()` / `wait_for_predict()`: a prediction started without
waiting for it, the pending prediction the model holds until it is waited for,
and what `predict()` gives the client to route each prediction by.

Hermetic by default: `init` and the `InferenceClient` boundary are patched. The
tests that take `mock_server` run the real HTTP layer: for poll answers that
only the transport tells apart, and, at the end, for a model fitted in one run
that a later run predicts with as a background job.
"""

from __future__ import annotations

import copy
import json
import logging
import pickle
import shutil
import sys
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import UUID

import httpx
import numpy as np
import pandas as pd
import pytest
import respx
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from tabpfn_client import PredictFailedError, RefitRequiredError, init, reset
from tabpfn_client.api_models import FitResponse, FitStatus, GetSettingsResponse
from tabpfn_client.client import ServiceClient
from tabpfn_client.config import Config
from tabpfn_client.constants import CACHE_DIR
from tabpfn_client.errors import PredictionGoneError
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.models import ApiMode, PredictionResult
from tabpfn_client.service_wrapper import InferenceClient, UserAuthenticationClient
from tests.mock_tabpfn_server import MockTabPFNServer
from tests.unit.test_client import _api_settings_payload, _FakeTime

FIT_ID = UUID("00000000-0000-0000-0000-000000000abc")
PREDICTION_ID = UUID("00000000-0000-0000-0000-0000000000aa")
NEW_PREDICTION_ID = UUID("00000000-0000-0000-0000-0000000000bb")
METADATA = {"test_set_num_rows": 3}
TIMINGS = {
    "test_set_transform_queue_wait_s": 0.5,
    "test_set_transform_s": 1.0,
    "predict_queue_wait_s": 0.0,
    "predict_s": 2.0,
}
# The server's own words.
NEEDS_REFITTING = (
    "This fitted model was created by an earlier version of the service and can "
    "no longer predict. Fit it again, then predict with the new "
    "fitted_train_set_id."
)
# What a new process runs to get a prediction the model no longer holds.
RESUME_HINT = (
    "its result stays on the server for about a day: call tabpfn_client.init(), "
    "then tabpfn_client.service_wrapper.InferenceClient.wait_for_predict("
    f'"{PREDICTION_ID}")'
)

X_TRAIN = np.arange(18, dtype=float).reshape(9, 2)
Y_TRAIN = np.array([0, 1, 2] * 3)
Y_TRAIN_REGRESSION = np.arange(9.0)
# Rows whose first feature tells them apart: 0, 2 and 4.
X_TEST = np.arange(6, dtype=float).reshape(3, 2)
BORDERS = np.linspace(0.0, 1.0, 4)

ESTIMATORS = pytest.mark.parametrize(
    ("estimator_cls", "y"),
    [(TabPFNClassifier, Y_TRAIN), (TabPFNRegressor, Y_TRAIN_REGRESSION)],
    ids=["classifier", "regressor"],
)

OUTPUTS = pytest.mark.parametrize(
    ("estimator_cls", "y", "output_type", "predict_kwargs"),
    [
        (TabPFNClassifier, Y_TRAIN, "preds", {}),
        (TabPFNClassifier, Y_TRAIN, "probas", {}),
        (TabPFNRegressor, Y_TRAIN_REGRESSION, "mean", {}),
        (TabPFNRegressor, Y_TRAIN_REGRESSION, "quantiles", {"quantiles": [0.1, 0.9]}),
        (TabPFNRegressor, Y_TRAIN_REGRESSION, "main", {}),
        (TabPFNRegressor, Y_TRAIN_REGRESSION, "full", {}),
    ],
    ids=["preds", "probas", "mean", "quantiles", "main", "full"],
)

# Without the optional dependencies, a full output warns that it has no criterion.
NO_CRITERION = pytest.mark.filterwarnings("ignore:Optional dependencies")


def _y_pred(X: Any, output_type: str) -> Any:
    """What the server predicts for the rows of `X`, which its values tell apart."""
    rows = np.asarray(X, dtype=float)[:, 0]
    quantiles = rows + np.array([[0.1], [0.9]])
    match output_type:
        case "preds":
            return rows.astype(int) % 3
        case "probas":
            return np.full((len(rows), 3), 1 / 3)
        case "quantiles":
            return quantiles
        case "main":
            return {"mean": rows, "median": rows, "mode": rows, "quantiles": quantiles}
        case "full":
            logits = rows[:, None] + np.arange(3.0)
            # A bar outside each row's support.
            logits[:, -1] = np.nan
            return {
                "mean": rows,
                "median": rows,
                "mode": rows,
                "quantiles": quantiles,
                "logits": logits,
                "borders": BORDERS,
            }
        case _:
            return rows


def _settings(**limits: int) -> GetSettingsResponse:
    """The settings of a server whose model limits are `limits` and the defaults."""
    payload = _api_settings_payload()
    for limit in [payload["max_model_limit"], *payload["model_limits"].values()]:
        limit.update(limits)
    return GetSettingsResponse.model_validate(payload)


@contextmanager
def _offline_server(
    y_pred: Callable[[], Any] = lambda: np.zeros(3),
    settings: GetSettingsResponse | None = None,
) -> Generator[SimpleNamespace]:
    """Run the estimators with no network: no auth, mocked inference.

    The server fits every model as `FIT_ID`, completed unless it was submitted,
    and accepts every submitted prediction as `PREDICTION_ID`. A prediction,
    whether predicted with one request or waited for, is `y_pred()` with
    `METADATA` and `TIMINGS`. `settings` are what `get_settings` returns; None
    leaves the model limits unchecked.
    """
    use_server_before = Config.use_server
    Config.use_server = True

    def fit_with_result(*args: Any, wait: bool, **kwargs: Any) -> FitResponse:
        status = FitStatus.COMPLETED if wait else FitStatus.PENDING
        return FitResponse(fitted_train_set_id=FIT_ID, status=status)

    def prediction(*args: Any, **kwargs: Any) -> PredictionResult:
        return PredictionResult(y_pred=y_pred(), metadata=METADATA, timings=TIMINGS)

    try:
        with (
            patch("tabpfn_client.estimator.init") as init_,
            patch.object(ServiceClient, "get_settings", return_value=settings),
            patch.object(
                InferenceClient, "fit_with_result", side_effect=fit_with_result
            ) as fit,
            patch.object(InferenceClient, "predict", side_effect=prediction) as predict,
            patch.object(
                InferenceClient, "submit_predict", return_value=PREDICTION_ID
            ) as submit,
            patch.object(
                InferenceClient, "wait_for_predict", side_effect=prediction
            ) as wait,
        ):
            yield SimpleNamespace(
                init=init_, fit=fit, predict=predict, submit=submit, wait=wait
            )
    finally:
        Config.use_server = use_server_before


@contextmanager
def _without_criterion_dependencies() -> Generator[None]:
    """Make the optional imports behind a full output's criterion fail."""
    with patch.dict(
        sys.modules, dict.fromkeys(["tabpfn", "tabpfn.regressor", "torch"])
    ):
        yield


def _fitted(
    estimator_cls: type[TabPFNClassifier | TabPFNRegressor] = TabPFNClassifier,
    y: np.ndarray = Y_TRAIN,
    **params: Any,
) -> Any:
    """Return an estimator whose fit has completed."""
    with _offline_server():
        return estimator_cls(**params).fit(X_TRAIN, y)


def _predict(model: Any, output_type: str, **kwargs: Any) -> Any:
    """What `predict()`, or `predict_proba()` for probabilities, returns."""
    if output_type == "preds":
        return model.predict(X_TEST)
    if output_type == "probas":
        return model.predict_proba(X_TEST)
    return model.predict(X_TEST, output_type=output_type, **kwargs)


def _pending_id(model: Any) -> UUID | None:
    """The id of the prediction the model holds, or None if it holds none."""
    pending = model._pending_predict
    return None if pending is None else pending.fitted_test_set_id


def _dropped(caplog: pytest.LogCaptureFixture) -> list[str]:
    """The warnings about pending predictions the model let go of."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "no longer pending" in r.getMessage()
    ]


def _assert_same(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            np.testing.assert_array_equal(actual[key], value, err_msg=key)
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for actual_part, expected_part in zip(actual, expected):
            np.testing.assert_array_equal(actual_part, expected_part)
    else:
        assert isinstance(actual, np.ndarray)
        np.testing.assert_array_equal(actual, expected)


class TestSubmitPredict:
    @ESTIMATORS
    def test_returns_once_the_server_accepted_the_prediction(self, estimator_cls, y):
        model = _fitted(estimator_cls, y)

        with _offline_server() as server:
            assert model.submit_predict(X_TEST) is model

        server.submit.assert_called_once()
        np.testing.assert_array_equal(server.submit.call_args.args[0], X_TEST)
        assert server.submit.call_args.kwargs["fitted_train_set_id"] == FIT_ID
        assert server.submit.call_args.kwargs["client_options"] is model.client_options
        server.predict.assert_not_called()
        server.wait.assert_not_called()
        assert _pending_id(model) == PREDICTION_ID
        assert model.get_timings()["predict"] is None

    @ESTIMATORS
    @pytest.mark.parametrize("api_mode", list(ApiMode))
    def test_runs_the_prediction_as_a_background_job_whatever_the_api_mode(
        self, estimator_cls, y, api_mode
    ):
        model = _fitted(estimator_cls, y, api_mode=api_mode)

        with _offline_server() as server:
            model.submit_predict(X_TEST)

        server.submit.assert_called_once()
        server.predict.assert_not_called()

    @OUTPUTS
    @NO_CRITERION
    def test_sends_the_request_predict_sends(
        self, estimator_cls, y, output_type, predict_kwargs
    ):
        model = _fitted(estimator_cls, y, n_estimators=4)

        with _offline_server(lambda: _y_pred(X_TEST, output_type)) as server:
            _predict(model, output_type, **predict_kwargs)
            model.submit_predict(X_TEST, output_type=output_type, **predict_kwargs)

        sent = server.submit.call_args.kwargs["task_config"]
        assert sent == server.predict.call_args.kwargs["task_config"]
        assert sent.predict_params.model_dump(mode="json", exclude_none=True) == {
            "output_type": output_type,
            **predict_kwargs,
        }

    @ESTIMATORS
    def test_sends_the_data_predict_sends(self, estimator_cls, y):
        # Text that predict() cleans before it sends it.
        X = pd.DataFrame({"a": [0.0, 2.0, 4.0], "text": ["x, y", "a  b", "c"]})
        model = _fitted(estimator_cls, y)

        with _offline_server() as server:
            model.predict(X)
            model.submit_predict(X)

        pd.testing.assert_frame_equal(
            server.submit.call_args.args[0], server.predict.call_args.args[0]
        )

    def test_needs_a_fit(self):
        with _offline_server() as server:
            with pytest.raises(NotFittedError):
                TabPFNClassifier().submit_predict(X_TEST)

        server.init.assert_not_called()
        server.submit.assert_not_called()

    def test_checks_the_data_as_predict_does_before_submitting(self, caplog):
        model = _fitted()
        with _offline_server():
            model.submit_predict(X_TEST)

        with _offline_server(settings=_settings(max_cols=1)) as server:
            with pytest.raises(ValueError, match="number of test columns"):
                model.submit_predict(X_TEST)

        server.submit.assert_not_called()
        assert _pending_id(model) == PREDICTION_ID
        assert _dropped(caplog) == []

    def test_refuses_a_full_output_one_response_cannot_cover(self):
        model = _fitted(TabPFNRegressor, Y_TRAIN_REGRESSION)
        settings = _settings(test_set_max_rows_w_full_regression_output=2)

        with _offline_server(settings=settings) as server:
            with pytest.raises(
                ValueError,
                match=r"too large for one background prediction, which covers at "
                r"most 2 rows; use predict\(\), which splits it",
            ):
                model.submit_predict(X_TEST, output_type="full")

        server.submit.assert_not_called()
        assert _pending_id(model) is None

    def test_submits_a_full_output_one_response_covers(self):
        model = _fitted(TabPFNRegressor, Y_TRAIN_REGRESSION)
        settings = _settings(test_set_max_rows_w_full_regression_output=3)

        with _offline_server(settings=settings) as server:
            model.submit_predict(X_TEST, output_type="full")

        server.submit.assert_called_once()

    @pytest.mark.parametrize(
        "failure",
        [
            RuntimeError("Fail to call submit_predict_job: [HTTP 429] Too many."),
            KeyboardInterrupt(),
        ],
        ids=["refused", "interrupted"],
    )
    def test_a_submit_that_fails_keeps_the_earlier_pending_prediction(
        self, failure, caplog
    ):
        model = _fitted()

        with _offline_server() as server:
            model.submit_predict(X_TEST)
            server.submit.side_effect = failure
            with pytest.raises(type(failure)):
                model.submit_predict(X_TEST, output_type="probas")
            model.wait_for_predict()

        assert server.wait.call_args.args == (PREDICTION_ID,)
        assert _dropped(caplog) == []

    @ESTIMATORS
    def test_a_new_submit_replaces_the_pending_prediction(
        self, estimator_cls, y, caplog
    ):
        model = _fitted(estimator_cls, y)

        with _offline_server() as server:
            model.submit_predict(X_TEST)
            server.submit.return_value = NEW_PREDICTION_ID
            model.submit_predict(X_TEST)
            model.wait_for_predict()

        assert server.wait.call_args.args == (NEW_PREDICTION_ID,)
        [dropped] = _dropped(caplog)
        assert dropped.startswith(
            f"Prediction {PREDICTION_ID} is no longer pending on this "
            f"{estimator_cls.__name__}: submit_predict() started a new one."
        )
        assert dropped.endswith(RESUME_HINT)

    @pytest.mark.parametrize(
        "duplicate",
        [lambda model: pickle.loads(pickle.dumps(model)), copy.deepcopy],
        ids=["pickle", "deepcopy"],
    )
    def test_a_pickled_or_copied_model_keeps_the_pending_prediction(self, duplicate):
        model = _fitted()
        with _offline_server():
            model.submit_predict(X_TEST)

        with _offline_server() as server:
            duplicate(model).wait_for_predict()

        assert server.wait.call_args.args == (PREDICTION_ID,)

    def test_a_clone_holds_no_pending_prediction(self):
        model = _fitted()
        with _offline_server():
            model.submit_predict(X_TEST)

        cloned = clone(model)
        assert isinstance(cloned, TabPFNClassifier)
        with _offline_server() as server:
            with pytest.raises(RuntimeError, match="no prediction to wait for"):
                cloned.wait_for_predict()

        server.wait.assert_not_called()


class TestWaitForPredict:
    @OUTPUTS
    @NO_CRITERION
    def test_returns_what_predict_returns(
        self, estimator_cls, y, output_type, predict_kwargs
    ):
        predicted, waited = _fitted(estimator_cls, y), _fitted(estimator_cls, y)

        with (
            _offline_server(lambda: _y_pred(X_TEST, output_type)),
            _without_criterion_dependencies(),
        ):
            expected = _predict(predicted, output_type, **predict_kwargs)
            waited.submit_predict(X_TEST, output_type=output_type, **predict_kwargs)
            actual = waited.wait_for_predict()

        _assert_same(actual, expected)
        assert waited._last_meta == METADATA
        assert waited.get_timings()["predict"] == TIMINGS

    @ESTIMATORS
    def test_waits_for_the_prediction_submit_predict_started(self, estimator_cls, y):
        model = _fitted(estimator_cls, y)
        with _offline_server():
            model.submit_predict(X_TEST)

        with _offline_server() as server:
            model.wait_for_predict()

        # Authenticates: a pickled model carries its pending prediction into a
        # new process.
        server.init.assert_called_once()
        assert server.wait.call_args.args == (PREDICTION_ID,)
        assert server.wait.call_args.kwargs == {"client_options": model.client_options}
        server.submit.assert_not_called()
        server.predict.assert_not_called()
        assert _pending_id(model) is None

    @ESTIMATORS
    @pytest.mark.parametrize("state", ["unfitted", "fitted", "waited_for"])
    def test_without_a_pending_prediction_there_is_nothing_to_wait_for(
        self, estimator_cls, y, state
    ):
        model = estimator_cls() if state == "unfitted" else _fitted(estimator_cls, y)
        with _offline_server() as server:
            if state == "waited_for":
                model.submit_predict(X_TEST).wait_for_predict()
            server.init.reset_mock()
            server.wait.reset_mock()

            with pytest.raises(
                RuntimeError,
                match=rf"^This {estimator_cls.__name__} has no prediction to wait "
                r"for\. Call submit_predict\(\) first\.$",
            ):
                model.wait_for_predict()

        server.init.assert_not_called()
        server.wait.assert_not_called()

    @ESTIMATORS
    @pytest.mark.parametrize(
        "error",
        [
            PredictFailedError(
                f"Prediction {PREDICTION_ID} failed: boom (error code INTERNAL_ERROR)"
            ),
            RefitRequiredError(NEEDS_REFITTING),
            PredictionGoneError(
                f"The server does not have prediction {PREDICTION_ID}: Fail to "
                "call get_predict_result: [HTTP 410] This prediction's result is "
                "no longer stored."
            ),
        ],
        ids=["failed", "refit_required", "gone"],
    )
    def test_a_prediction_with_nothing_left_to_wait_for_is_no_longer_pending(
        self, estimator_cls, y, error, caplog
    ):
        model = _fitted(estimator_cls, y)

        with _offline_server() as server:
            model.submit_predict(X_TEST)
            server.wait.side_effect = error
            with pytest.raises(type(error)) as raised:
                model.wait_for_predict()
            with pytest.raises(RuntimeError, match="no prediction to wait for"):
                model.wait_for_predict()

        assert raised.value is error
        server.wait.assert_called_once()
        # No drop warning, whose resume hint would fail.
        assert _dropped(caplog) == []
        # The fit stays: fitting again is for the caller to decide.
        assert model.model_id_ == FIT_ID
        check_is_fitted(model)

    @ESTIMATORS
    @pytest.mark.parametrize(
        "stop",
        [
            TimeoutError(f"Prediction {PREDICTION_ID} did not reach a terminal state."),
            KeyboardInterrupt(),
            # The prediction runs on while the client cannot poll for it.
            RuntimeError(
                "Fail to call get_predict_result: [HTTP 401] Requires authentication."
            ),
            # After a rollback, which a roll-forward with the prediction can end.
            RuntimeError(
                f"Cannot wait for prediction {PREDICTION_ID}: this TabPFN server "
                "does not run predictions as background jobs right now. Wait for "
                "it again later: a server that runs them again may still "
                "complete it."
            ),
        ],
        ids=["timeout", "interrupt", "unauthorized", "route_missing"],
    )
    def test_a_wait_that_stops_keeps_the_prediction_pending(
        self, estimator_cls, y, stop
    ):
        model = _fitted(estimator_cls, y)

        with _offline_server() as server:
            model.submit_predict(X_TEST)
            answer = server.wait.side_effect
            server.wait.side_effect = stop
            with pytest.raises(type(stop)):
                model.wait_for_predict()
            assert _pending_id(model) == PREDICTION_ID

            server.wait.side_effect = answer
            model.wait_for_predict()

        assert [c.args for c in server.wait.call_args_list] == [(PREDICTION_ID,)] * 2
        assert _pending_id(model) is None

    @ESTIMATORS
    @pytest.mark.parametrize(
        "make_answer",
        [
            lambda: httpx.Response(
                404,
                text="<html><body>Not Found</body></html>",
                headers={"content-type": "text/html"},
            ),
            lambda: httpx.Response(404, json={"detail": "Not Found"}),
            lambda: httpx.Response(
                404, json={"message": "no Route matched with those values"}
            ),
        ],
        ids=["html", "starlette_default", "gateway"],
    )
    def test_a_404_without_the_servers_code_keeps_the_prediction_pending(
        self, estimator_cls, y, make_answer, mock_server, caplog
    ):
        model = _fitted(estimator_cls, y)
        with _offline_server():
            model.submit_predict(X_TEST)
        poll_route = _router(mock_server).post("/tabpfn/get_predict_result")
        poll_route.mock(return_value=make_answer())

        with (
            patch("tabpfn_client.estimator.init"),
            patch.object(ServiceClient, "get_settings", return_value=None),
            # A fake clock, so that a 404 the client retried by mistake fails
            # the test at once instead of polling for the 2-hour default.
            patch("tabpfn_client.client.time", _FakeTime()),
            pytest.raises(RuntimeError) as raised,
        ):
            model.wait_for_predict()

        # Only the server's NOT_FOUND code says that it does not have the
        # prediction; a 404 from a proxy, say, tells nothing about it, so the
        # model keeps it.
        assert type(raised.value) is RuntimeError
        assert poll_route.call_count == 1
        assert _pending_id(model) == PREDICTION_ID
        assert _dropped(caplog) == []

    def test_the_criterion_warning_points_at_the_line_that_asked(self):
        model = _fitted(TabPFNRegressor, Y_TRAIN_REGRESSION)

        with (
            _offline_server(lambda: _y_pred(X_TEST, "full")),
            _without_criterion_dependencies(),
        ):
            with pytest.warns(UserWarning, match="Optional dependencies") as predicted:
                model.predict(X_TEST, output_type="full")
            model.submit_predict(X_TEST, output_type="full")
            with pytest.warns(UserWarning, match="Optional dependencies") as waited:
                model.wait_for_predict()

        assert [w.filename for w in predicted] == [__file__]
        assert [w.filename for w in waited] == [__file__]


class TestPredict:
    @ESTIMATORS
    @pytest.mark.parametrize("api_mode", list(ApiMode))
    @pytest.mark.parametrize(
        ("params", "thinking"),
        [
            ({}, False),
            ({"thinking_mode": True}, True),
            ({"thinking_effort": "high"}, True),
        ],
        ids=["standard", "thinking_mode", "thinking_effort"],
    )
    def test_passes_what_the_prediction_is_routed_by(
        self, estimator_cls, y, api_mode, params, thinking
    ):
        model = _fitted(estimator_cls, y, api_mode=api_mode, **params)

        with _offline_server() as server:
            model.predict(X_TEST)

        kwargs = server.predict.call_args.kwargs
        assert kwargs["api_mode"] == api_mode
        assert kwargs["n_train_rows"] == len(X_TRAIN)
        assert kwargs["thinking"] is thinking

    @ESTIMATORS
    def test_does_not_know_the_train_rows_of_an_assigned_model_id(
        self, estimator_cls, y
    ):
        model = estimator_cls()
        model.model_id_ = FIT_ID

        with _offline_server() as server:
            model.predict(X_TEST)

        assert server.predict.call_args.kwargs["n_train_rows"] is None

    @ESTIMATORS
    def test_keeps_nothing_but_its_metadata_and_timings(self, estimator_cls, y):
        model = _fitted(estimator_cls, y)
        before = dict(vars(model))

        with _offline_server():
            model.predict(X_TEST)

        missing = object()
        changed = {k for k, v in vars(model).items() if before.get(k, missing) is not v}
        assert changed == {"_last_meta", "_last_timings"}

    @ESTIMATORS
    def test_leaves_a_pending_prediction_alone(self, estimator_cls, y):
        model = _fitted(estimator_cls, y)

        with _offline_server() as server:
            model.submit_predict(X_TEST)
            model.predict(X_TEST)
            model.wait_for_predict()

        assert server.wait.call_args.args == (PREDICTION_ID,)

    @NO_CRITERION
    def test_splits_a_full_output_into_one_request_per_chunk_in_order(self):
        X = np.arange(10, dtype=float).reshape(5, 2)
        model = _fitted(TabPFNRegressor, Y_TRAIN_REGRESSION, api_mode=ApiMode.ASYNC)
        settings = _settings(test_set_max_rows_w_full_regression_output=2)

        with _offline_server(settings=settings) as server:
            server.predict.side_effect = lambda X_rows, **kwargs: PredictionResult(
                y_pred=_y_pred(X_rows, "full"), timings=TIMINGS
            )
            output = model.predict(X, output_type="full")

        calls = server.predict.call_args_list
        assert [c.args[0][:, 0].tolist() for c in calls] == [[0, 2], [4, 6], [8]]
        for call in calls:
            assert call.kwargs["api_mode"] == ApiMode.ASYNC
            assert call.kwargs["n_train_rows"] == len(X_TRAIN)
            assert call.kwargs["thinking"] is False
        np.testing.assert_array_equal(output["mean"], X[:, 0])
        assert model.get_timings()["predict"] == {k: 3 * v for k, v in TIMINGS.items()}

    @NO_CRITERION
    def test_returns_the_unknown_fields_of_a_full_output_in_one_request(self):
        def full_output() -> dict[str, Any]:
            return {**_y_pred(X_TEST, "full"), "surprise": np.ones(3)}

        model = _fitted(TabPFNRegressor, Y_TRAIN_REGRESSION)

        with _offline_server(full_output):
            predicted = model.predict(X_TEST, output_type="full")
            waited = model.submit_predict(X_TEST, output_type="full").wait_for_predict()

        np.testing.assert_array_equal(predicted["surprise"], np.ones(3))
        np.testing.assert_array_equal(waited["surprise"], np.ones(3))


class TestFit:
    @ESTIMATORS
    @pytest.mark.parametrize("refit", ["fit", "submit_fit"])
    def test_a_new_fit_drops_the_pending_prediction(
        self, estimator_cls, y, refit, caplog
    ):
        model = _fitted(estimator_cls, y)

        with _offline_server() as server:
            model.submit_predict(X_TEST)
            getattr(model, refit)(X_TRAIN, y)
            with pytest.raises(RuntimeError, match="no prediction to wait for"):
                model.wait_for_predict()

        server.wait.assert_not_called()
        [dropped] = _dropped(caplog)
        assert dropped.startswith(
            f"Prediction {PREDICTION_ID} is no longer pending on this "
            f"{estimator_cls.__name__}: a new fit replaced the one it was made "
            "with."
        )
        assert dropped.endswith(RESUME_HINT)

    @pytest.mark.parametrize("refit", ["fit", "submit_fit"])
    @pytest.mark.parametrize(
        ("y", "error"),
        [(Y_TRAIN, RuntimeError), (np.array([0, 1, np.nan] * 3), ValueError)],
        ids=["refused", "invalid"],
    )
    def test_a_refit_that_fails_keeps_the_pending_prediction(
        self, refit, y, error, caplog
    ):
        model = _fitted()

        with _offline_server() as server:
            model.submit_predict(X_TEST)
            server.fit.side_effect = RuntimeError("the server failed the fit")
            with pytest.raises(error):
                getattr(model, refit)(X_TRAIN, y)
            model.wait_for_predict()

        assert server.wait.call_args.args == (PREDICTION_ID,)
        assert _dropped(caplog) == []


@pytest.fixture
def mock_server() -> Iterator[MockTabPFNServer]:
    with MockTabPFNServer() as server:
        yield server


def _router(mock_server: MockTabPFNServer) -> respx.MockRouter:
    assert mock_server.router is not None
    return mock_server.router


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sent(route: respx.Route) -> list[dict[str, Any]]:
    return [json.loads(call.request.content) for call in route.calls]


class TestWithServer:
    dummy_token = "dummy_token"

    def setup_method(self):
        reset()
        ServiceClient.reset_authorization()
        ServiceClient._api_settings = None

    def teardown_method(self):
        Config.is_initialized = False
        ServiceClient.reset_authorization()
        ServiceClient._api_settings = None
        ServiceClient._api_settings_ts = 0.0
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    def test_a_model_fitted_in_one_run_predicts_as_a_background_job_in_a_later_one(
        self,
        mock_server: MockTabPFNServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        # init() reads the environment, where a developer's poll timeout would
        # bound the polls below.
        monkeypatch.delenv("TABPFN_CLIENT_ASYNC_POLL_TIMEOUT", raising=False)
        fit_id = "00000000-0000-0000-0000-000000000002"
        upload_id = "00000000-0000-0000-0000-000000000003"
        prediction_id = str(PREDICTION_ID)
        probabilities = [[0.25, 0.25, 0.5], [0.5, 0.25, 0.25], [0.25, 0.5, 0.25]]
        path = tmp_path / "model.json"
        UserAuthenticationClient.CACHED_TOKEN_FILE.parent.mkdir(
            parents=True, exist_ok=True
        )
        UserAuthenticationClient.CACHED_TOKEN_FILE.write_text(self.dummy_token)

        router = _router(mock_server)
        auth_route = router.get(mock_server.endpoints.protected_root.path)
        auth_route.respond(200)
        router.get(mock_server.endpoints.retrieve_greeting_messages.path).respond(
            200, json={"messages": []}
        )
        settings = _api_settings_payload()
        settings["async_settings"] |= {
            "predict_use_above": {"train_rows": 50_000, "test_rows": 100_000},
            "thinking_predict_use_above": {"train_rows": 25_000, "test_rows": 10_000},
        }
        router.get("/tabpfn/get_settings").respond(200, json=settings)
        router.post("/tabpfn/prepare_train_set_upload").respond(
            409,
            json={
                "message": "duplicate",
                "error_code": "DUPLICATE_TRAIN_SET_UPLOAD",
                "train_set_upload_id": "00000000-0000-0000-0000-000000000001",
            },
        )
        router.post("/tabpfn/fit").respond(
            200, json={"fitted_train_set_id": fit_id, "status": "completed"}
        )
        router.post("/tabpfn/prepare_test_set_upload").respond(
            409,
            json={
                "message": "duplicate",
                "error_code": "DUPLICATE_TEST_SET_UPLOAD",
                "test_set_upload_id": upload_id,
            },
        )
        # No /tabpfn/predict route: predicting through it fails the test.
        submit_route = router.post("/tabpfn/submit_predict_job")
        submit_route.respond(200, json={"fitted_test_set_id": prediction_id})
        completed = {
            "prediction": probabilities,
            "metadata": {
                "task": "classification",
                "package_version": "0.0.0",
                "tabpfn_config": {},
                "test_set_num_rows": 3,
                "test_set_num_cols": 2,
            },
            "timings": TIMINGS,
            "fitted_test_set_id": prediction_id,
            "status": "completed",
            "retry_in_secs": None,
            "error": None,
            "error_code": None,
        }
        result_route = router.post("/tabpfn/get_predict_result")
        result_route.side_effect = [
            httpx.Response(
                200,
                json={
                    "fitted_test_set_id": prediction_id,
                    "status": "pending",
                    "retry_in_secs": 0.5,
                },
            ),
            # Keepalive pings precede a result the server is slow to stream.
            httpx.Response(
                200,
                content=b"  \n " + json.dumps(completed).encode(),
                headers={"content-type": "application/json"},
            ),
        ]

        with patch("tabpfn_client.client.time.sleep") as sleep:
            # A run that fits the model, saves it and ends.
            init(use_server=True)
            TabPFNClassifier().fit(X_TRAIN, Y_TRAIN).save_model(path)

            # A later run, in a new process: loading asks nothing, and
            # submit_predict() returns once the server accepted the prediction.
            Config.is_initialized = False
            ServiceClient.reset_authorization()
            ServiceClient._api_settings = None
            clf = TabPFNClassifier.load_model(path)
            assert auth_route.call_count == 1
            clf.submit_predict(X_TEST, output_type="probas")
            assert result_route.call_count == 0
            assert clf.save_model() == _read(path), "a pending prediction isn't saved"
            result = clf.wait_for_predict()

        np.testing.assert_array_equal(result, probabilities)
        assert auth_route.call_count == 2, "each run authenticates once"
        [submitted] = _sent(submit_route)
        assert submitted.keys() == {
            "test_set_upload_id",
            "fitted_train_set_id",
            "task_config",
        }
        assert submitted["test_set_upload_id"] == upload_id
        assert submitted["fitted_train_set_id"] == fit_id
        assert submitted["task_config"]["predict_params"] == {"output_type": "probas"}
        assert _sent(result_route) == [{"fitted_test_set_id": prediction_id}] * 2
        sleep.assert_called_once_with(0.5)
        assert clf.get_timings()["predict"] == TIMINGS
        with pytest.raises(RuntimeError, match="no prediction to wait for"):
            clf.wait_for_predict()
