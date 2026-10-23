#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""`submit_fit()` / `wait_for_fit()`: a fit started without waiting for it, the
pending model it leaves, and how a later run saves, loads and waits for it; and
the thinking fits that always run as background jobs.

Hermetic by default: `init` and the `InferenceClient` boundary are patched. The
mock-server tests at the end run the real HTTP layer for a fit submitted in one
run and waited for in a later one, and for thinking fits.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID

import httpx
import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from tabpfn_client import FitFailedError, FittedModelNotFoundError, init, reset
from tabpfn_client.api_models import (
    FitResponse,
    FitStatus,
    FitTimings,
    GetFitStatusResponse,
)
from tabpfn_client.client import PredictionResult, ServiceClient
from tabpfn_client.config import Config
from tabpfn_client.constants import CACHE_DIR
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.models import ApiMode
from tabpfn_client.service_wrapper import InferenceClient, UserAuthenticationClient
from tests.mock_tabpfn_server import with_mock_server

FIT_ID = UUID("00000000-0000-0000-0000-000000000abc")
NEW_FIT_ID = UUID("00000000-0000-0000-0000-000000000def")
TIMINGS = {
    "elapsed_s": 42.0,
    "queue_wait_s": 2.0,
    "train_set_transform_s": 10.0,
    "fit_s": 30.0,
}

X_TRAIN = np.arange(18, dtype=float).reshape(9, 2)
Y_TRAIN = np.array([0, 1, 2] * 3)
Y_TRAIN_REGRESSION = np.arange(9.0)
X_TEST = np.ones((3, 2))

ESTIMATORS = pytest.mark.parametrize(
    ("estimator_cls", "y"),
    [(TabPFNClassifier, Y_TRAIN), (TabPFNRegressor, Y_TRAIN_REGRESSION)],
    ids=["classifier", "regressor"],
)

FIT_ERRORS = pytest.mark.parametrize(
    "error",
    [
        FitFailedError("Fit failed: boom"),
        FittedModelNotFoundError("Fitted model not found"),
    ],
    ids=["failed", "gone"],
)


def _completed(fit_id: UUID = FIT_ID) -> GetFitStatusResponse:
    return GetFitStatusResponse.model_validate(
        {"fitted_train_set_id": str(fit_id), "status": "completed", "timings": TIMINGS}
    )


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _offline_server() -> Generator[SimpleNamespace]:
    """Run the estimators with no network: no auth, no settings, mocked inference.

    The server accepts every fit as `FIT_ID`: one request comes back completed
    with `TIMINGS`, a background job comes back pending, and waiting for a fit
    finds it completed, and a prediction is a row of zeros per
    test row.
    """
    use_server_before = Config.use_server
    Config.use_server = True

    def fit_with_result(*args: Any, api_mode: ApiMode, **kwargs: Any) -> FitResponse:
        # A background job comes back pending; one request, completed.
        if api_mode == ApiMode.ASYNC:
            return FitResponse(fitted_train_set_id=FIT_ID, status=FitStatus.PENDING)
        return FitResponse(
            fitted_train_set_id=FIT_ID,
            status=FitStatus.COMPLETED,
            timings=FitTimings.model_validate(TIMINGS),
        )

    try:
        with (
            patch("tabpfn_client.estimator.init") as init_,
            patch.object(ServiceClient, "get_settings", return_value=None),
            patch.object(
                InferenceClient, "fit_with_result", side_effect=fit_with_result
            ) as fit,
            patch.object(
                InferenceClient,
                "wait_for_fit",
                side_effect=lambda fit_id, **kwargs: _completed(fit_id),
            ) as wait,
            patch.object(
                InferenceClient,
                "predict",
                side_effect=lambda X, **kwargs: PredictionResult(
                    y_pred=np.zeros(len(X), dtype=int)
                ),
            ) as predict,
        ):
            yield SimpleNamespace(init=init_, fit=fit, wait=wait, predict=predict)
    finally:
        Config.use_server = use_server_before


def _submitted(
    estimator_cls: type[TabPFNClassifier | TabPFNRegressor] = TabPFNClassifier,
    y: np.ndarray = Y_TRAIN,
    **params: Any,
) -> Any:
    """Return an estimator whose fit `submit_fit()` started and nobody waited for."""
    with _offline_server():
        return estimator_cls(**params).submit_fit(X_TRAIN, y)


def _fitted(
    estimator_cls: type[TabPFNClassifier | TabPFNRegressor] = TabPFNClassifier,
    y: np.ndarray = Y_TRAIN,
) -> Any:
    """Return an estimator with a completed fit."""
    with _offline_server():
        return estimator_cls().fit(X_TRAIN, y)


class TestSubmitFit:
    @ESTIMATORS
    def test_returns_once_the_server_accepted_the_fit(self, estimator_cls, y):
        model = estimator_cls()

        with _offline_server() as server:
            assert model.submit_fit(X_TRAIN, y) is model

        assert server.fit.call_args.kwargs["wait"] is False
        server.wait.assert_not_called()
        assert model.model_id_ == FIT_ID
        assert model._n_train_rows == 9
        assert model.get_timings()["fit"] is None
        with pytest.raises(NotFittedError):
            check_is_fitted(model)

    @ESTIMATORS
    @pytest.mark.parametrize("api_mode", list(ApiMode))
    def test_runs_the_fit_as_a_background_job_whatever_the_api_mode(
        self, estimator_cls, y, api_mode
    ):
        with _offline_server() as server:
            estimator_cls(api_mode=api_mode).submit_fit(X_TRAIN, y)

        assert server.fit.call_args.kwargs["api_mode"] == ApiMode.ASYNC

    def test_checks_the_data_before_submitting(self):
        with _offline_server() as server:
            with pytest.raises(ValueError, match="NaN"):
                TabPFNClassifier().submit_fit(X_TRAIN, np.array([0, 1, np.nan] * 3))

        server.fit.assert_not_called()


class TestPredict:
    @ESTIMATORS
    def test_raises_on_a_pending_fit_without_a_request(self, estimator_cls, y):
        model = _submitted(estimator_cls, y)
        methods = [
            getattr(model, m, None)
            for m in ("predict", "predict_proba", "submit_predict")
        ]

        with _offline_server() as server:
            for predict in (m for m in methods if m is not None):
                with pytest.raises(NotFittedError, match=r"wait_for_fit\(\)"):
                    predict(X_TEST)

        server.init.assert_not_called()
        server.wait.assert_not_called()
        server.predict.assert_not_called()

    def test_an_unfitted_model_raises_without_authenticating(self):
        with _offline_server() as server:
            with pytest.raises(NotFittedError):
                TabPFNClassifier().predict(X_TEST)

        server.init.assert_not_called()


class TestWaitForFit:
    @ESTIMATORS
    def test_waits_for_the_fit_then_predicts(self, estimator_cls, y):
        model = _submitted(estimator_cls, y)

        with _offline_server() as server:
            assert model.wait_for_fit() is model
            model.predict(X_TEST)

        assert server.wait.call_args.args == (FIT_ID,)
        server.fit.assert_not_called()
        assert server.predict.call_args.kwargs["fitted_train_set_id"] == FIT_ID
        check_is_fitted(model)
        assert model.get_timings()["fit"] == TIMINGS

    def test_asks_nothing_once_the_fit_has_completed(self):
        clf = _submitted()

        with _offline_server() as server:
            clf.wait_for_fit()
            clf.wait_for_fit()
            fitted = TabPFNClassifier().fit(X_TRAIN, Y_TRAIN)
            assert fitted.wait_for_fit() is fitted

        server.wait.assert_called_once()

    def test_an_unfitted_model_has_nothing_to_wait_for(self):
        with _offline_server() as server:
            with pytest.raises(NotFittedError, match=r"submit_fit\(\)"):
                TabPFNClassifier().wait_for_fit()

        server.wait.assert_not_called()

    @FIT_ERRORS
    def test_a_fit_the_server_lost_leaves_the_model_unfitted(self, error):
        clf = _submitted()

        with _offline_server() as server:
            server.wait.side_effect = error
            with pytest.raises(type(error)):
                clf.wait_for_fit()

        assert not hasattr(clf, "model_id_")
        assert not hasattr(clf, "classes_")
        for nothing_to_do in (
            check_is_fitted,
            type(clf).save_model,
            type(clf).wait_for_fit,
        ):
            with pytest.raises(NotFittedError):
                nothing_to_do(clf)

    @pytest.mark.parametrize(
        "stop",
        [
            KeyboardInterrupt(),
            TimeoutError(),
            # Polling failed: the fit itself may still complete.
            RuntimeError(
                f"Polling fit {FIT_ID} failed 4 times in a row: [HTTP 500] boom"
            ),
        ],
        ids=["interrupt", "timeout", "poll_failures"],
    )
    def test_a_wait_that_stops_keeps_the_fit_pending(self, stop):
        clf = _submitted()

        with _offline_server() as server:
            server.wait.side_effect = stop
            with pytest.raises(type(stop)):
                clf.wait_for_fit()
            assert clf.save_model()["fit_status"] == "pending"

            server.wait.side_effect = lambda fit_id, **kwargs: _completed(fit_id)
            clf.wait_for_fit()

        assert server.wait.call_count == 2
        check_is_fitted(clf)


class TestFit:
    @ESTIMATORS
    @pytest.mark.parametrize("pending", ["submitted", "loaded"])
    def test_starts_a_new_fit_on_a_pending_model(self, estimator_cls, y, pending):
        model = _submitted(estimator_cls, y, api_mode=ApiMode.SYNC)
        if pending == "loaded":
            model = estimator_cls.load_model(model.save_model())

        with _offline_server() as server:
            server.fit.side_effect = lambda *args, **kwargs: FitResponse(
                fitted_train_set_id=NEW_FIT_ID,
                status=FitStatus.COMPLETED,
                timings=FitTimings.model_validate(TIMINGS),
            )
            assert model.fit(X_TRAIN, y) is model
            model.predict(X_TEST)

        server.fit.assert_called_once()
        assert server.fit.call_args.kwargs["api_mode"] == ApiMode.SYNC
        server.wait.assert_not_called()
        assert model.model_id_ == NEW_FIT_ID
        assert server.predict.call_args.kwargs["fitted_train_set_id"] == NEW_FIT_ID
        assert model.save_model()["fit_status"] == "completed"

    @ESTIMATORS
    @pytest.mark.parametrize("api_mode", list(ApiMode))
    @pytest.mark.parametrize(
        "thinking",
        [{"thinking_mode": True}, {"thinking_effort": "high"}],
        ids=["thinking_mode", "thinking_effort"],
    )
    def test_a_thinking_fit_runs_as_a_background_job_whatever_the_api_mode(
        self, estimator_cls, y, api_mode, thinking
    ):
        with _offline_server() as server:
            model = estimator_cls(api_mode=api_mode, **thinking).fit(X_TRAIN, y)

        assert server.fit.call_args.kwargs["api_mode"] == ApiMode.ASYNC
        server.wait.assert_called_once_with(FIT_ID, client_options=model.client_options)
        assert model.get_timings()["fit"] == TIMINGS
        check_is_fitted(model)

    @ESTIMATORS
    @pytest.mark.parametrize(
        "stop",
        [
            KeyboardInterrupt(),
            TimeoutError(),
            # Polling failed: the fit itself may still complete.
            RuntimeError(
                f"Polling fit {NEW_FIT_ID} failed 4 times in a row: [HTTP 500] boom"
            ),
        ],
        ids=["interrupt", "timeout", "poll_failures"],
    )
    def test_a_poll_that_stops_leaves_the_model_pending_on_the_new_fit(
        self, estimator_cls, y, stop
    ):
        model = _fitted(estimator_cls, y)

        with _offline_server() as server:
            server.fit.side_effect = lambda *args, **kwargs: FitResponse(
                fitted_train_set_id=NEW_FIT_ID, status=FitStatus.PENDING
            )
            server.wait.side_effect = stop
            with pytest.raises(type(stop)):
                model.fit(X_TRAIN, y)

            assert server.fit.call_args.kwargs["wait"] is False
            assert model.model_id_ == NEW_FIT_ID
            assert model.save_model()["fit_status"] == "pending"
            with pytest.raises(NotFittedError, match=r"wait_for_fit\(\)"):
                model.predict(X_TEST)

            server.wait.side_effect = lambda fit_id, **kwargs: _completed(fit_id)
            model.wait_for_fit()

        assert server.wait.call_args_list[-1].args == (NEW_FIT_ID,)
        check_is_fitted(model)

    @ESTIMATORS
    @FIT_ERRORS
    def test_a_refit_that_fails_while_polled_leaves_the_model_unfitted(
        self, estimator_cls, y, error
    ):
        model = _fitted(estimator_cls, y)

        with _offline_server() as server:
            server.fit.side_effect = lambda *args, **kwargs: FitResponse(
                fitted_train_set_id=NEW_FIT_ID, status=FitStatus.PENDING
            )
            server.wait.side_effect = error
            with pytest.raises(type(error)):
                model.fit(X_TRAIN, y)

        # Neither the failed fit nor the one it replaced.
        assert not hasattr(model, "model_id_")
        with pytest.raises(NotFittedError):
            check_is_fitted(model)

    @ESTIMATORS
    @pytest.mark.parametrize("api_mode", list(ApiMode))
    def test_other_fits_run_in_the_api_mode(self, estimator_cls, y, api_mode):
        with _offline_server() as server:
            estimator_cls(api_mode=api_mode).fit(X_TRAIN, y)

        assert server.fit.call_args.kwargs["api_mode"] == api_mode

    @ESTIMATORS
    @pytest.mark.parametrize(
        ("fit", "params"),
        [("fit", {"thinking_mode": True}), ("submit_fit", {})],
        ids=["thinking_fit", "submit_fit"],
    )
    def test_an_invalid_api_mode_fails_before_the_upload(
        self, estimator_cls, y, fit, params
    ):
        model = estimator_cls(api_mode=cast(Any, "ASYNC"), **params)

        with _offline_server() as server:
            with pytest.raises(ValueError, match="is not a valid ApiMode"):
                getattr(model, fit)(X_TRAIN, y)

        server.fit.assert_not_called()


class TestSaveAndLoad:
    def test_saves_a_pending_fit(self, tmp_path):
        path = tmp_path / "model.json"

        record = _submitted().save_model(path)

        assert record == _read(path)
        assert record["fit_status"] == "pending"
        assert record["model_id"] == str(FIT_ID)
        assert record["classes"] == [0, 1, 2]
        assert record["n_train_rows"] == 9

    def test_loads_a_pending_fit_as_pending_without_a_request(self, tmp_path):
        path = tmp_path / "model.json"
        _submitted().save_model(path)

        with _offline_server() as server:
            loaded = TabPFNClassifier.load_model(path)
            with pytest.raises(NotFittedError, match=r"wait_for_fit\(\)"):
                loaded.predict(X_TEST)

        server.init.assert_not_called()
        server.wait.assert_not_called()
        server.predict.assert_not_called()
        assert loaded.model_id_ == FIT_ID
        np.testing.assert_array_equal(loaded.classes_, [0, 1, 2])
        assert loaded.save_model() == _read(path)

    @ESTIMATORS
    def test_a_later_run_waits_for_the_loaded_fit(self, tmp_path, estimator_cls, y):
        path = tmp_path / "model.json"
        _submitted(estimator_cls, y).save_model(path)
        before = path.read_text()

        with _offline_server() as server:
            loaded = estimator_cls.load_model(path)
            loaded.wait_for_fit()
            # A model loaded in a fresh process authenticates before waiting.
            server.init.assert_called_once()
            loaded.predict(X_TEST)

        assert server.wait.call_args.args == (FIT_ID,)
        server.fit.assert_not_called()
        assert server.predict.call_args.kwargs["fitted_train_set_id"] == FIT_ID
        assert loaded.get_timings()["fit"] == TIMINGS
        # Nothing rewrites the file; saving again records the completed fit.
        assert path.read_text() == before
        assert loaded.save_model(path)["fit_status"] == "completed"


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
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    # `with_mock_server` hides the test's signature, so pytest cannot pass
    # `tmp_path` to the test itself.
    @pytest.fixture(autouse=True)
    def _tmp_path(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    @with_mock_server()
    def test_a_fit_submitted_in_one_run_is_waited_for_in_a_later_one(self, mock_server):
        fit_id = "00000000-0000-0000-0000-000000000002"
        path = self.tmp_path / "model.json"
        UserAuthenticationClient.CACHED_TOKEN_FILE.parent.mkdir(
            parents=True, exist_ok=True
        )
        UserAuthenticationClient.CACHED_TOKEN_FILE.write_text(self.dummy_token)

        auth_route = mock_server.router.get(mock_server.endpoints.protected_root.path)
        auth_route.respond(200)
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})
        upload_route = mock_server.router.post("/tabpfn/prepare_train_set_upload")
        upload_route.respond(
            409,
            json={
                "message": "duplicate",
                "error_code": "DUPLICATE_TRAIN_SET_UPLOAD",
                "train_set_upload_id": "00000000-0000-0000-0000-000000000001",
            },
        )
        # No sync /tabpfn/fit route: submitting through it fails the test.
        submit_route = mock_server.router.post("/tabpfn/submit_fit_job")
        submit_route.respond(200, json={"fitted_train_set_id": fit_id})
        status_route = mock_server.router.post("/tabpfn/get_fit_status")
        status_route.side_effect = [
            httpx.Response(
                200,
                json={
                    "fitted_train_set_id": fit_id,
                    "status": "pending",
                    "retry_in_secs": 5,
                },
            ),
            httpx.Response(
                200,
                json={
                    "fitted_train_set_id": fit_id,
                    "status": "completed",
                    "timings": TIMINGS,
                },
            ),
        ]
        mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            409,
            json={
                "message": "duplicate",
                "error_code": "DUPLICATE_TEST_SET_UPLOAD",
                "test_set_upload_id": "00000000-0000-0000-0000-000000000003",
            },
        )
        predict_route = mock_server.router.post(mock_server.endpoints.predict.path)
        predict_route.respond(
            200,
            json={
                "prediction": [1, 0, 1],
                "metadata": {
                    "task": "classification",
                    "package_version": "0.3.0",
                    "tabpfn_config": {},
                    "test_set_num_rows": 3,
                    "test_set_num_cols": 2,
                },
            },
        )

        with (
            patch.object(ServiceClient, "get_settings", return_value=None),
            patch("tabpfn_client.client.time.sleep") as sleep,
        ):
            # A run that submits the fit, saves it and ends.
            init(use_server=True)
            TabPFNClassifier(thinking_mode=True).submit_fit(
                X_TRAIN, Y_TRAIN
            ).save_model(path)
            assert status_route.call_count == 0
            assert _read(path)["fit_status"] == "pending"
            assert _read(path)["model_id"] == fit_id

            # A later run, in a new process: loading and predicting ask nothing,
            # and wait_for_fit() polls until the fit has completed.
            Config.is_initialized = False
            ServiceClient.reset_authorization()
            clf = TabPFNClassifier.load_model(path)
            with pytest.raises(NotFittedError):
                clf.predict(X_TEST)
            assert status_route.call_count == 0
            assert auth_route.call_count == 1
            clf.wait_for_fit()
            result = clf.predict(X_TEST)

        np.testing.assert_array_equal(result, [1, 0, 1])
        assert upload_route.call_count == 1
        assert submit_route.call_count == 1
        assert status_route.call_count == 2
        sleep.assert_called_once_with(5)
        assert auth_route.call_count == 2, "each run authenticates once"
        predict_request = json.loads(predict_route.calls.last.request.content)
        assert predict_request["fitted_train_set_id"] == fit_id
        assert clf.get_timings()["fit"] == TIMINGS

    @with_mock_server()
    def test_a_thinking_fit_holds_no_request_open_whatever_the_api_mode(
        self, mock_server
    ):
        fit_id = "00000000-0000-0000-0000-000000000002"
        # Signed in already, so `init()` asks nothing; teardown signs out.
        Config.is_initialized = True
        mock_server.router.post("/tabpfn/prepare_train_set_upload").respond(
            409,
            json={
                "message": "duplicate",
                "error_code": "DUPLICATE_TRAIN_SET_UPLOAD",
                "train_set_upload_id": "00000000-0000-0000-0000-000000000001",
            },
        )
        # No sync /tabpfn/fit route: fitting through it fails the test.
        submit_route = mock_server.router.post("/tabpfn/submit_fit_job")
        submit_route.respond(200, json={"fitted_train_set_id": fit_id})
        status_route = mock_server.router.post("/tabpfn/get_fit_status")
        status_route.respond(
            200,
            json={
                "fitted_train_set_id": fit_id,
                "status": "completed",
                "timings": TIMINGS,
            },
        )

        with patch.object(ServiceClient, "get_settings", return_value=None):
            fitted = [
                TabPFNClassifier(thinking_mode=True, api_mode=api_mode).fit(
                    X_TRAIN, Y_TRAIN
                )
                for api_mode in ApiMode
            ]

        assert submit_route.call_count == status_route.call_count == len(ApiMode)
        for clf in fitted:
            check_is_fitted(clf)
            assert clf.get_timings()["fit"] == TIMINGS
