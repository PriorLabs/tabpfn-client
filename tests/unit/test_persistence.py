#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""`save_model()` / `load_model()`: the record they exchange, the fitted state
it restores, and what a loaded estimator does on `predict()`.

Hermetic by default: `init` and the `InferenceClient` / `ServiceClient`
boundaries are patched. The mock-server tests at the end run the real HTTP
layer to cover the "fresh process" path, where authentication happens on the
first `predict()`.
"""

from __future__ import annotations

import json
import pickle
import shutil
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch
from uuid import UUID

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from tabpfn_client import FittedModelNotFoundError, init, reset
from tabpfn_client.api_models import GetSettingsResponse
from tabpfn_client.client import PredictionResult, ServiceClient
from tabpfn_client.config import Config
from tabpfn_client.constants import CACHE_DIR
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.service_wrapper import InferenceClient, UserAuthenticationClient
from tests.mock_tabpfn_server import with_mock_server

MODEL_ID = UUID("00000000-0000-0000-0000-000000000abc")
N_TRAIN_ROWS = 120


def _fitted_regressor(**params: Any) -> TabPFNRegressor:
    reg = TabPFNRegressor(**params)
    reg.model_id_ = MODEL_ID
    reg._n_train_rows = N_TRAIN_ROWS
    return reg


def _fitted_classifier(classes: list[Any], **params: Any) -> TabPFNClassifier:
    clf = TabPFNClassifier(**params)
    clf.model_id_ = MODEL_ID
    clf._n_train_rows = N_TRAIN_ROWS
    clf.classes_ = np.asarray(classes)
    return clf


def _model_params(estimator: TabPFNClassifier | TabPFNRegressor) -> dict[str, Any]:
    """Constructor params minus the transport-only `client_options`."""
    return {
        k: v
        for k, v in estimator.get_params(deep=False).items()
        if k != "client_options"
    }


def _api_settings_payload(predict_row_pairs_budget: int) -> dict[str, Any]:
    limit = {
        "train_set_max_rows": 100_000,
        "train_set_max_cells": 100_000_000,
        "test_set_max_rows": 100_000,
        "test_set_max_cells": 100_000_000,
        "test_set_max_rows_w_full_regression_output": 100_000,
        "max_cols": 2_000,
        "max_classes": 10,
        "predict_row_pairs_budget": predict_row_pairs_budget,
    }
    return {
        "default_model_version": "v2.5",
        "max_model_limit": limit,
        "model_limits": {"v2.5": limit},
        "dataset_max_size_bytes": 100_000_000,
        "async_settings": {
            "use_above_trainset_size_bytes": 50 * 1024 * 1024,
            "poll_timeout_secs": 7200.0,
        },
    }


@contextmanager
def _offline_server(prediction: PredictionResult | None = None):
    """Run fit/predict with no network: no auth, no settings, canned inference."""
    use_server_before = Config.use_server
    Config.use_server = True
    try:
        with (
            patch("tabpfn_client.estimator.init"),
            patch.object(ServiceClient, "get_settings", return_value=None),
            patch.object(InferenceClient, "fit", return_value=MODEL_ID) as fit,
            patch.object(
                InferenceClient, "predict", return_value=prediction
            ) as predict,
        ):
            yield fit, predict
    finally:
        Config.use_server = use_server_before


class TestRecord:
    def test_save_requires_a_fit(self):
        for Est in (TabPFNClassifier, TabPFNRegressor):
            with pytest.raises(NotFittedError):
                Est().save_model()

    def test_record_holds_exactly_the_fitted_state(self):
        reg = _fitted_regressor(n_estimators=4, api_mode="async")

        record = reg.save_model()

        assert set(record) == {
            "tabpfn_client_version",
            "task",
            "model_id",
            "params",
            "n_train_rows",
            "classes",
        }
        assert record["task"] == "regression"
        assert record["model_id"] == str(MODEL_ID)
        assert record["n_train_rows"] == N_TRAIN_ROWS
        assert record["classes"] is None
        assert record["params"] == _model_params(reg)
        assert "client_options" not in record["params"]
        json.dumps(record)  # JSON-serialisable as returned

    def test_none_valued_params_survive(self):
        # `random_state=None` asks for a fresh seed each run and differs from
        # the default of 0, so a record must keep explicit Nones rather than
        # let the constructor default take over on load.
        record = _fitted_regressor(random_state=None).save_model()
        assert record["params"]["random_state"] is None
        assert TabPFNRegressor.load_model(record).get_params()["random_state"] is None

    def test_file_round_trip(self, tmp_path):
        path = tmp_path / "model.json"
        clf = _fitted_classifier(["cat", "dog"], n_estimators=3, thinking_mode=True)

        returned = clf.save_model(path)
        assert json.loads(path.read_text()) == returned

        for source in (path, str(path), returned):
            loaded = TabPFNClassifier.load_model(source)
            check_is_fitted(loaded)
            assert loaded.model_id_ == MODEL_ID
            assert loaded._n_train_rows == N_TRAIN_ROWS
            np.testing.assert_array_equal(loaded.classes_, np.array(["cat", "dog"]))
            assert _model_params(loaded) == _model_params(clf)

    def test_directly_assigned_id_saves_without_classes_or_row_count(self):
        # Reusing a fit by assigning `model_id_` leaves no classes or row count
        # behind; the record must still round-trip what there is.
        clf = TabPFNClassifier()
        clf.model_id_ = MODEL_ID

        record = clf.save_model()
        assert record["classes"] is None
        assert record["n_train_rows"] is None

        loaded = TabPFNClassifier.load_model(record)
        check_is_fitted(loaded)
        assert not hasattr(loaded, "classes_")
        assert loaded._n_train_rows is None


class TestLoadRejects:
    def test_a_model_of_the_other_task(self):
        with pytest.raises(
            ValueError, match="classification model into TabPFNRegressor"
        ):
            TabPFNRegressor.load_model(_fitted_classifier([0, 1]).save_model())
        with pytest.raises(ValueError, match="regression model into TabPFNClassifier"):
            TabPFNClassifier.load_model(_fitted_regressor().save_model())

    def test_malformed_sources(self, tmp_path):
        not_json = tmp_path / "model.json"
        not_json.write_text("{not json")
        with pytest.raises(ValueError, match="not a model file"):
            TabPFNRegressor.load_model(not_json)

        with pytest.raises(ValueError, match="not a model record"):
            TabPFNRegressor.load_model({"task": "regression"})  # no model_id

        unknown_task = _fitted_regressor().save_model()
        unknown_task["task"] = "timeseries"
        with pytest.raises(ValueError, match="not a model record"):
            TabPFNRegressor.load_model(unknown_task)

        with pytest.raises(FileNotFoundError):
            TabPFNRegressor.load_model(tmp_path / "missing.json")

    def test_params_this_version_does_not_know(self):
        record = _fitted_regressor().save_model()
        record["tabpfn_client_version"] = "99.0.0"
        record["params"]["hyperspace_mode"] = True

        with pytest.raises(ValueError) as excinfo:
            TabPFNRegressor.load_model(record)
        assert "99.0.0" in str(excinfo.value)
        assert "hyperspace_mode" in str(excinfo.value)

    def test_unknown_top_level_fields_are_ignored(self):
        # A newer tabpfn-client may add fields to the record; an older one
        # should still load everything it understands.
        record = _fitted_regressor().save_model()
        record["fitted_at"] = "2026-09-03T00:00:00Z"
        assert TabPFNRegressor.load_model(record).model_id_ == MODEL_ID


class TestLoadedEstimator:
    def test_predicts_against_the_saved_fit_without_fitting(self):
        loaded = TabPFNRegressor.load_model(
            _fitted_regressor(n_estimators=4).save_model()
        )
        prediction = PredictionResult(y_pred={"mean": np.zeros(3)}, metadata={})

        with _offline_server(prediction) as (fit, predict):
            loaded.predict(np.random.randn(3, 2))

        fit.assert_not_called()
        kwargs = predict.call_args.kwargs
        assert kwargs["fitted_train_set_id"] == MODEL_ID
        assert kwargs["task_config"].tabpfn_config.n_estimators == 4

    def test_row_budget_is_enforced_after_loading(self):
        # The server caps n_train_rows * n_test_rows per call. The saved row
        # count keeps that check on the client after loading, so an oversized
        # test set fails before anything is uploaded.
        loaded = TabPFNRegressor.load_model(_fitted_regressor().save_model())
        settings = GetSettingsResponse.model_validate(
            _api_settings_payload(predict_row_pairs_budget=N_TRAIN_ROWS * 2)
        )

        with (
            patch("tabpfn_client.estimator.init"),
            patch.object(ServiceClient, "get_settings", return_value=settings),
            pytest.raises(ValueError, match="exceeds the maximum of 2"),
        ):
            loaded.predict(np.random.randn(3, 2))

    def test_clone_starts_unfitted(self):
        loaded = TabPFNClassifier.load_model(
            _fitted_classifier([0, 1], n_estimators=4).save_model()
        )
        cloned = clone(loaded)
        assert isinstance(cloned, TabPFNClassifier)
        with pytest.raises(NotFittedError):
            check_is_fitted(cloned)
        assert cloned.get_params()["n_estimators"] == 4

    def test_refit_replaces_the_loaded_state(self):
        loaded = TabPFNClassifier.load_model(_fitted_classifier([0, 1]).save_model())
        new_id = UUID("00000000-0000-0000-0000-000000000def")
        X = np.random.randn(9, 2)
        y = np.array([0, 1, 2] * 3)

        with _offline_server() as (fit, _):
            fit.return_value = new_id
            loaded.fit(X, y)

        assert loaded.model_id_ == new_id
        assert loaded._n_train_rows == 9
        np.testing.assert_array_equal(loaded.classes_, [0, 1, 2])


class TestPickle:
    def test_fit_keeps_the_row_count_not_the_data(self):
        X = np.random.randn(2_000, 50)  # 800 kB of float64
        y = np.random.randn(2_000)
        reg = TabPFNRegressor()

        with _offline_server():
            reg.fit(X, y)

        assert reg._n_train_rows == 2_000
        assert len(pickle.dumps(reg)) < 10_000

    def test_pickle_round_trip_stays_fitted(self):
        restored = pickle.loads(pickle.dumps(_fitted_classifier(["a", "b"])))
        check_is_fitted(restored)
        assert restored.model_id_ == MODEL_ID
        assert restored._n_train_rows == N_TRAIN_ROWS
        np.testing.assert_array_equal(restored.classes_, ["a", "b"])


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

    @with_mock_server()
    def test_predict_reports_a_model_the_server_no_longer_has(self, mock_server):
        mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            404,
            json={
                "message": "Fitted train set not found",
                "error_code": "NOT_FOUND",
                "trace_id": "00000000-0000-0000-0000-00000000beef",
            },
        )
        loaded = TabPFNRegressor.load_model(_fitted_regressor().save_model())

        with (
            patch("tabpfn_client.estimator.init"),
            patch.object(ServiceClient, "get_settings", return_value=None),
            pytest.raises(FittedModelNotFoundError) as excinfo,
        ):
            loaded.predict(np.random.randn(3, 2))

        message = str(excinfo.value)
        assert str(MODEL_ID) in message
        assert "fit()" in message
        assert "00000000-0000-0000-0000-00000000beef" in message

    @with_mock_server()
    def test_fit_save_load_predict_across_runs(self, mock_server):
        fitted_id = "00000000-0000-0000-0000-000000000002"
        # The token cached by an earlier login is all a later run has.
        UserAuthenticationClient.CACHED_TOKEN_FILE.parent.mkdir(
            parents=True, exist_ok=True
        )
        UserAuthenticationClient.CACHED_TOKEN_FILE.write_text(self.dummy_token)

        auth_route = mock_server.router.get(mock_server.endpoints.protected_root.path)
        auth_route.respond(200)
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})
        mock_server.router.get("/tabpfn/get_settings").respond(
            200, json=_api_settings_payload(predict_row_pairs_budget=10**12)
        )
        mock_server.router.post("/tabpfn/prepare_train_set_upload").respond(
            409,
            json={
                "message": "duplicate",
                "error_code": "DUPLICATE_TRAIN_SET_UPLOAD",
                "train_set_upload_id": "00000000-0000-0000-0000-000000000001",
            },
        )
        mock_server.router.post(mock_server.endpoints.fit.path).respond(
            200, json={"fitted_train_set_id": fitted_id, "status": "completed"}
        )
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
        X_train = np.random.randn(20, 2)
        y_train = np.array([0, 1] * 10)
        X_test = np.random.randn(3, 2)

        # First run: fit and save.
        init(use_server=True)
        clf = TabPFNClassifier(n_estimators=10).fit(X_train, y_train)
        record = clf.save_model()
        assert record["model_id"] == fitted_id
        assert record["n_train_rows"] == 20

        # Later run: nothing survives but the record, and the client has to
        # authenticate again, which the first predict() takes care of.
        Config.is_initialized = False
        ServiceClient.reset_authorization()
        ServiceClient._api_settings = None
        later = TabPFNClassifier.load_model(record)

        result = later.predict(X_test)

        np.testing.assert_array_equal(result, [1, 0, 1])
        np.testing.assert_array_equal(later.classes_, [0, 1])
        assert auth_route.call_count == 2, "each run authenticates once"
        predict_request = json.loads(predict_route.calls.last.request.content)
        assert predict_request["fitted_train_set_id"] == fitted_id
        assert predict_request["task_config"]["tabpfn_config"]["n_estimators"] == 10
