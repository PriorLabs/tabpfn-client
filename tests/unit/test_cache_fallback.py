#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""`cache_fallback_strategy`, and the warning the client logs when the server
could not use the fit's cache.

The server then fits the model again for that prediction, and reports a
`cache_outcome` of `"fallback"` in its metadata. Hermetic: the `InferenceClient`
boundary is patched, as in `test_submit_predict`.
"""

import logging
import pickle
import warnings
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from pydantic import TypeAdapter
from sklearn.base import clone

from tabpfn_client.api_models import Metadata
from tabpfn_client.client import ServiceClient
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.models import PredictionResult
from tabpfn_client.service_wrapper import InferenceClient
from tests.unit.test_submit_predict import (
    ESTIMATORS,
    NO_CRITERION,
    OUTPUTS,
    X_TEST,
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


def _logged(caplog: pytest.LogCaptureFixture) -> list[tuple[str, int, str]]:
    """The records logged at WARNING or above, as (logger, level, message)."""
    return [t for t in caplog.record_tuples if t[1] >= logging.WARNING]


@OUTPUTS
@pytest.mark.parametrize("via", ["predict", "wait_for_predict"])
@NO_CRITERION
def test_a_prediction_that_could_not_use_the_cache_logs_once(
    estimator_cls, y, output_type, predict_kwargs, via, caplog
):
    model = _fitted(estimator_cls, y)

    with (
        _offline_server(lambda: _y_pred(X_TEST, output_type), metadata=FALLBACK),
        _without_criterion_dependencies(),
    ):
        if via == "wait_for_predict":
            model.submit_predict(X_TEST, output_type=output_type, **predict_kwargs)
            model.wait_for_predict()
        else:
            _predict(model, output_type, **predict_kwargs)

    assert _logged(caplog) == [FALLBACK_RECORD]
    assert model._last_meta == FALLBACK


@ESTIMATORS
@pytest.mark.parametrize(
    "metadata",
    [
        {"cache_outcome": "hit"},
        # A value a newer server may send.
        {"cache_outcome": "stale"},
        # An older server, which does not report it.
        {},
    ],
    ids=["hit", "unknown", "unreported"],
)
def test_a_prediction_that_did_not_fall_back_logs_nothing(
    estimator_cls, y, metadata, caplog
):
    model = _fitted(estimator_cls, y)

    with _offline_server(metadata=metadata):
        model.predict(X_TEST)

    assert _logged(caplog) == []


def test_a_prediction_that_could_not_use_the_cache_returns_under_warnings_as_errors():
    model = _fitted()

    with _offline_server(metadata=FALLBACK), warnings.catch_warnings():
        # The server has billed the prediction by now: nothing may lose it.
        warnings.simplefilter("error")
        predicted = model.predict(X_TEST)

    np.testing.assert_array_equal(predicted, np.zeros(3))


@NO_CRITERION
@pytest.mark.parametrize(
    ("outcomes", "logged"),
    [
        # Two requests, neither the first nor the last: one log, not one each.
        (["hit", "fallback", "fallback", "hit"], 1),
        (["hit", "miss", "not_requested", "hit"], 0),
    ],
    ids=["fallback", "no_fallback"],
)
def test_a_split_full_output_logs_once_if_any_request_could_not_use_the_cache(
    outcomes, logged, caplog
):
    X = np.arange(14, dtype=float).reshape(7, 2)
    model = _fitted(TabPFNRegressor, Y_TRAIN_REGRESSION)
    settings = _settings(test_set_max_rows_w_full_regression_output=2)

    def predict_chunks(x_chunks: Any, **kwargs: Any) -> list[PredictionResult]:
        results = []
        # Strict: the 7 rows go in 4 requests.
        for chunk, outcome in zip(x_chunks, outcomes, strict=True):
            y_pred = _y_pred(chunk, "full")
            results.append(PredictionResult(y_pred, {"cache_outcome": outcome}))
        return results

    with (
        _offline_server(settings=settings),
        patch.object(InferenceClient, "predict_chunks", side_effect=predict_chunks),
        _without_criterion_dependencies(),
    ):
        model.predict(X, output_type="full")

    assert _logged(caplog) == [FALLBACK_RECORD] * logged


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize(
    ("cache_outcome", "logged"),
    [("fallback", 1), (None, 0)],
    ids=["fallback", "unreported"],
)
def test_reads_the_outcome_from_the_metadata_the_server_sends(
    task, cache_outcome, logged, caplog
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
    model = _fitted()

    with _offline_server(metadata=result.metadata):
        model.predict(X_TEST)

    assert _logged(caplog) == [FALLBACK_RECORD] * logged


def test_the_log_names_none_of_the_servers_internals():
    internal_names = ("general pool", "pool", "kv", "lane", "gapi", "worker")
    message = FALLBACK_RECORD[2].lower()

    assert [name for name in internal_names if name in message] == []


def test_a_subclass_whose_predict_returns_an_array_still_predicts():
    # A dependant overrides `_predict` to return the finished array.
    class ArrayClassifier(TabPFNClassifier):
        def _predict(self, X, output_type):
            return np.full(len(X), output_type)

    model = ArrayClassifier()

    np.testing.assert_array_equal(model.predict(X_TEST), ["preds"] * 3)
    np.testing.assert_array_equal(model.predict_proba(X_TEST), ["probas"] * 3)


@ESTIMATORS
def test_get_params_and_clone_keep_the_strategy(estimator_cls, y):
    model = estimator_cls(cache_fallback_strategy="refit")

    assert estimator_cls().cache_fallback_strategy == "error"
    assert model.get_params()["cache_fallback_strategy"] == "refit"
    assert clone(model).cache_fallback_strategy == "refit"


@ESTIMATORS
def test_an_older_pickle_without_the_strategy_predicts_with_the_default(
    estimator_cls, y
):
    model = _fitted(estimator_cls, y)
    # Estimators pickled by v0.7.1 or earlier have no `cache_fallback_strategy`.
    del model.cache_fallback_strategy
    restored = pickle.loads(pickle.dumps(model))

    with _offline_server() as server:
        restored.predict(X_TEST)

    assert restored.get_params()["cache_fallback_strategy"] == "error"
    assert server.predict.call_args.kwargs["cache_fallback_strategy"] == "error"


@ESTIMATORS
@pytest.mark.parametrize("via", ["predict", "submit_predict"])
def test_an_invalid_strategy_raises_before_any_request(estimator_cls, y, via):
    model = _fitted(estimator_cls, y, cache_fallback_strategy="retry")

    with (
        _offline_server() as server,
        pytest.raises(ValueError, match="cache_fallback_strategy must be one of"),
    ):
        getattr(model, via)(X_TEST)

    server.init.assert_not_called()
    server.predict.assert_not_called()
    server.submit.assert_not_called()
