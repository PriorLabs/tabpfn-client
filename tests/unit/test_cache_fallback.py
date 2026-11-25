#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""The warning the client logs when the server could not use the fit's cache.

The server then fits the model again for that prediction, and reports a
`cache_outcome` of `"fallback"` in its metadata. Hermetic: the `InferenceClient`
boundary is patched, as in `test_submit_predict`.
"""

import logging
import warnings
from collections.abc import Callable
from typing import Any
from unittest.mock import patch
from uuid import UUID

import numpy as np
import pytest
from pydantic import TypeAdapter

from tabpfn_client.api_models import Metadata
from tabpfn_client.client import ServiceClient
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.models import PredictionResult
from tabpfn_client.service_wrapper import InferenceClient
from tests.unit.test_submit_predict import (
    NO_CRITERION,
    OUTPUTS,
    TIMINGS,
    X_TEST,
    Y_TRAIN,
    Y_TRAIN_REGRESSION,
    _fitted,
    _offline_server,
    _predict,
    _settings,
    _without_criterion_dependencies,
    _y_pred,
)

FALLBACK = {"test_set_num_rows": 3, "cache_outcome": "fallback"}
# The words the user reads, as the estimators' logger records them.
FALLBACK_RECORD = (
    "tabpfn_client.estimator",
    logging.WARNING,
    "The saved cache of this fitted model could not be used for this prediction, "
    "so TabPFN fit the model again on the server. The prediction took longer than "
    "usual.",
)
# Names of the server's internals, which users must not see.
INTERNAL_NAMES = ("general pool", "pool", "kv", "lane", "gapi", "worker")

# Each estimator's `predict()` without arguments.
DEFAULT_OUTPUTS = pytest.mark.parametrize(
    ("estimator_cls", "y", "output_type"),
    [
        (TabPFNClassifier, Y_TRAIN, "preds"),
        (TabPFNRegressor, Y_TRAIN_REGRESSION, "mean"),
    ],
    ids=["classifier", "regressor"],
)
VIA = pytest.mark.parametrize("via", ["predict", "wait_for_predict"])


class _ArrayClassifier(TabPFNClassifier):
    """A dependant's classifier, whose `_predict` returns the finished array."""

    def _predict(self, X, output_type):
        return np.full(len(X), output_type)


def _answer(
    metadata: dict[str, Any], output_type: str
) -> Callable[..., PredictionResult]:
    """A server that answers every prediction of `X_TEST` with `metadata`."""

    def answer(
        *args: Any,
        on_accepted: Callable[[UUID | None], None] | None = None,
        **kwargs: Any,
    ) -> PredictionResult:
        if on_accepted is not None:
            on_accepted(None)
        return PredictionResult(
            y_pred=_y_pred(X_TEST, output_type), metadata=metadata, timings=TIMINGS
        )

    return answer


def _predict_via(model: Any, via: str, output_type: str, **kwargs: Any) -> Any:
    """Predict `X_TEST` as `_predict` does, or with `submit_predict()` and then
    `wait_for_predict()`."""
    if via == "wait_for_predict":
        model.submit_predict(X_TEST, output_type=output_type, **kwargs)
        return model.wait_for_predict()
    return _predict(model, output_type, **kwargs)


def _logged(caplog: pytest.LogCaptureFixture) -> list[tuple[str, int, str]]:
    """The records logged at WARNING or above, as (logger, level, message)."""
    return [t for t in caplog.record_tuples if t[1] >= logging.WARNING]


@OUTPUTS
@VIA
@NO_CRITERION
def test_a_prediction_that_could_not_use_the_cache_logs_once(
    estimator_cls, y, output_type, predict_kwargs, via, caplog
):
    model = _fitted(estimator_cls, y)

    with _offline_server() as server, _without_criterion_dependencies():
        server.predict.side_effect = _answer(FALLBACK, output_type)
        server.wait.side_effect = _answer(FALLBACK, output_type)
        _predict_via(model, via, output_type, **predict_kwargs)

    assert _logged(caplog) == [FALLBACK_RECORD]
    assert model._last_meta == FALLBACK


@DEFAULT_OUTPUTS
@VIA
@pytest.mark.parametrize(
    "metadata",
    [
        {"cache_outcome": "hit"},
        {"cache_outcome": "miss"},
        {"cache_outcome": "not_requested"},
        {"cache_outcome": None},
        # A value a newer server may send.
        {"cache_outcome": "stale"},
        # An older server, which does not report it.
        {"test_set_num_rows": 3},
        {},
    ],
    ids=["hit", "miss", "not_requested", "none", "unknown", "unreported", "empty"],
)
def test_a_prediction_that_did_not_fall_back_logs_nothing(
    estimator_cls, y, output_type, via, metadata, caplog
):
    model = _fitted(estimator_cls, y)

    with _offline_server() as server:
        server.predict.side_effect = _answer(metadata, output_type)
        server.wait.side_effect = _answer(metadata, output_type)
        _predict_via(model, via, output_type)

    assert _logged(caplog) == []


@DEFAULT_OUTPUTS
@VIA
def test_a_prediction_that_could_not_use_the_cache_returns_under_warnings_as_errors(
    estimator_cls, y, output_type, via
):
    model = _fitted(estimator_cls, y)

    with _offline_server() as server, warnings.catch_warnings():
        # The server has billed the prediction by now: nothing may lose it.
        warnings.simplefilter("error")
        server.predict.side_effect = _answer(FALLBACK, output_type)
        server.wait.side_effect = _answer(FALLBACK, output_type)
        predicted = _predict_via(model, via, output_type)

    np.testing.assert_array_equal(predicted, _y_pred(X_TEST, output_type))


@NO_CRITERION
@pytest.mark.parametrize(
    ("outcomes", "logged"),
    [
        (["hit", "fallback", "hit"], 1),
        (["fallback"] * 3, 1),
        (["hit"] * 3, 0),
    ],
    ids=["one_request", "every_request", "no_request"],
)
def test_a_split_full_output_logs_once_if_any_request_could_not_use_the_cache(
    outcomes, logged, caplog
):
    X = np.arange(10, dtype=float).reshape(5, 2)
    model = _fitted(TabPFNRegressor, Y_TRAIN_REGRESSION)
    settings = _settings(test_set_max_rows_w_full_regression_output=2)

    def predict_chunks(x_chunks: Any, **kwargs: Any) -> list[PredictionResult]:
        results = []
        # Strict: the 5 rows go in 3 requests.
        for chunk, outcome in zip(x_chunks, outcomes, strict=True):
            results.append(
                PredictionResult(
                    y_pred=_y_pred(chunk, "full"), metadata={"cache_outcome": outcome}
                )
            )
        return results

    with (
        _offline_server(settings=settings),
        patch.object(InferenceClient, "predict_chunks", side_effect=predict_chunks),
        _without_criterion_dependencies(),
    ):
        model.predict(X, output_type="full")

    assert _logged(caplog) == [FALLBACK_RECORD] * logged


@pytest.mark.parametrize(
    ("estimator_cls", "y", "task"),
    [
        (TabPFNClassifier, Y_TRAIN, "classification"),
        (TabPFNRegressor, Y_TRAIN_REGRESSION, "regression"),
    ],
    ids=["classifier", "regressor"],
)
@pytest.mark.parametrize(
    ("cache_outcome", "logged"),
    [("fallback", 1), ("stale", 0), (None, 0)],
    ids=["fallback", "unknown", "unreported"],
)
def test_reads_the_outcome_from_the_metadata_the_server_sends(
    estimator_cls, y, task, cache_outcome, logged, caplog
):
    metadata = TypeAdapter(Metadata).validate_python(
        {
            "task": task,
            "package_version": "0.0.0",
            "tabpfn_config": {},
            "test_set_num_rows": 3,
            "test_set_num_cols": 2,
            "cache_outcome": cache_outcome,
        }
    )
    result = ServiceClient._prediction_result([0, 1, 2], metadata, None, None)
    model = _fitted(estimator_cls, y)

    with _offline_server() as server:
        server.predict.side_effect = None
        server.predict.return_value = result
        model.predict(X_TEST)

    assert _logged(caplog) == [FALLBACK_RECORD] * logged


def test_the_log_names_none_of_the_servers_internals(caplog):
    model = _fitted()

    with _offline_server() as server:
        server.predict.side_effect = _answer(FALLBACK, "preds")
        model.predict(X_TEST)

    [(_, _, message)] = _logged(caplog)
    assert [name for name in INTERNAL_NAMES if name in message.lower()] == []


def test_a_subclass_whose_predict_returns_an_array_still_predicts():
    model = _ArrayClassifier()

    np.testing.assert_array_equal(model.predict(X_TEST), ["preds"] * 3)
    np.testing.assert_array_equal(model.predict_proba(X_TEST), ["probas"] * 3)
