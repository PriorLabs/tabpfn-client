"""Server-reported timings reach `FitResult`, `PredictionResult` and the estimators."""

from __future__ import annotations

import time
import unittest
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch
from uuid import UUID

import httpx
import numpy as np
import pytest

from tabpfn_client.api_models import (
    ClassifierConfig,
    ClassifierFitTaskConfig,
    GetSettingsResponse,
)
from tabpfn_client.client import ServiceClient
from tabpfn_client.config import Config
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor, _combine_timings
from tabpfn_client.models import FitResult, PredictionResult
from tabpfn_client.service_wrapper import InferenceClient
from tests.mock_tabpfn_server import with_mock_server
from tests.unit.test_client import _api_settings_payload, _fast_poll_settings

FIT_ID = "00000000-0000-0000-0000-000000000002"
FIT_TIMINGS = {
    "elapsed_s": 3.0,
    "queue_wait_s": 1.0,
    "train_set_transform_s": 0.5,
    "fit_s": 1.5,
}
PREDICT_TIMINGS = {
    "test_set_transform_queue_wait_s": 0.25,
    "test_set_transform_s": 0.5,
    "predict_queue_wait_s": 0.75,
    "predict_s": 2.0,
}
N_ROWS = 20


def _upload_info(url: str) -> dict[str, Any]:
    return {
        "signed_urls": [url],
        "expires_at": 1_700_000_000.0,
        "required_headers": {},
    }


class TestClientTimings(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = rng.rand(N_ROWS, 3)
        self.y = np.arange(N_ROWS) % 2
        ServiceClient.reset_authorization()
        ServiceClient._api_settings = GetSettingsResponse(**_api_settings_payload())
        ServiceClient._api_settings_ts = time.monotonic()
        ServiceClient.authorize("dummy_token")

    def tearDown(self):
        ServiceClient.reset_authorization()
        ServiceClient._api_settings = None
        ServiceClient._api_settings_ts = 0.0

    @staticmethod
    def _route_train_upload(mock_server) -> None:
        mock_server.router.post("/tabpfn/prepare_train_set_upload").respond(
            200,
            json={
                "train_set_upload_id": "00000000-0000-0000-0000-000000000001",
                "x_train_info": _upload_info("https://upload.example/x_train"),
                "y_train_info": _upload_info("https://upload.example/y_train"),
            },
        )

    def _call(self, method):
        with (
            patch.object(ServiceClient, "_upload_to_gcs"),
            patch.object(
                ServiceClient,
                "_resolve_async_settings",
                return_value=_fast_poll_settings(),
            ),
            patch("tabpfn_client.client.time.sleep"),
        ):
            return method(
                self.X,
                self.y,
                tabpfn_systems=["preprocessing", "text"],
                task_config=ClassifierFitTaskConfig(),
            )

    @with_mock_server()
    def test_completed_fit_returns_its_timings(self, mock_server):
        self._route_train_upload(mock_server)
        mock_server.router.post("/tabpfn/fit").respond(
            200,
            json={
                "fitted_train_set_id": FIT_ID,
                "status": "completed",
                "timings": FIT_TIMINGS,
            },
        )

        result = self._call(ServiceClient.fit_with_result)

        self.assertEqual(
            result, FitResult(fitted_train_set_id=UUID(FIT_ID), timings=FIT_TIMINGS)
        )

    @with_mock_server()
    def test_pending_fit_takes_timings_from_the_final_status(self, mock_server):
        self._route_train_upload(mock_server)
        mock_server.router.post("/tabpfn/fit").respond(
            200,
            json={
                "fitted_train_set_id": FIT_ID,
                "status": "pending",
                "timings": {"elapsed_s": 0.5},
            },
        )
        mock_server.router.post("/tabpfn/get_fit_status").respond(
            200,
            json={
                "fitted_train_set_id": FIT_ID,
                "status": "completed",
                "timings": FIT_TIMINGS,
            },
        )

        result = self._call(ServiceClient.fit_with_result)

        self.assertEqual(result.timings, FIT_TIMINGS)

    @with_mock_server()
    def test_fit_against_a_server_without_timings(self, mock_server):
        self._route_train_upload(mock_server)
        mock_server.router.post("/tabpfn/fit").respond(
            200, json={"fitted_train_set_id": FIT_ID, "status": "completed"}
        )

        self.assertIsNone(self._call(ServiceClient.fit_with_result).timings)
        self.assertEqual(self._call(ServiceClient.fit), UUID(FIT_ID))

    @with_mock_server()
    def test_predict_returns_timings_when_the_server_reports_them(self, mock_server):
        mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            200,
            json={
                "test_set_upload_id": "00000000-0000-0000-0000-000000000003",
                "x_test_info": _upload_info("https://upload.example/x_test"),
            },
        )
        body = {
            "prediction": [0] * N_ROWS,
            "metadata": {
                "task": "classification",
                "package_version": "0.0.0",
                "tabpfn_config": {},
                "test_set_num_rows": N_ROWS,
                "test_set_num_cols": 3,
            },
        }
        mock_server.router.post("/tabpfn/predict").side_effect = [
            httpx.Response(200, json={**body, "timings": PREDICT_TIMINGS}),
            httpx.Response(200, json=body),
        ]

        with patch.object(ServiceClient, "_upload_to_gcs"):
            reported, unreported = (
                ServiceClient.predict(
                    fitted_train_set_id=UUID(FIT_ID),
                    x_test=self.X,
                    task_config=ClassifierConfig(),
                )
                for _ in range(2)
            )

        self.assertEqual(reported.timings, PREDICT_TIMINGS)
        self.assertIsNone(unreported.timings)


@contextmanager
def _offline(fit_result: FitResult, prediction: PredictionResult):
    use_server_before = Config.use_server
    Config.use_server = True
    try:
        with (
            patch("tabpfn_client.estimator.init"),
            patch.object(ServiceClient, "get_settings", return_value=None),
            patch.object(InferenceClient, "fit_with_result", return_value=fit_result),
            patch.object(InferenceClient, "predict", return_value=prediction),
        ):
            yield
    finally:
        Config.use_server = use_server_before


@pytest.mark.parametrize("Est", [TabPFNClassifier, TabPFNRegressor])
def test_estimator_exposes_fit_and_predict_timings(Est):
    X = np.random.RandomState(0).rand(N_ROWS, 3)
    y = (np.arange(N_ROWS) % 2).astype(float)
    y_pred = np.zeros(N_ROWS) if Est is TabPFNRegressor else np.zeros(N_ROWS, dtype=int)
    estimator = Est()
    assert estimator.get_timings() == {"fit": None, "predict": None}

    with _offline(
        FitResult(fitted_train_set_id=UUID(FIT_ID), timings=FIT_TIMINGS),
        PredictionResult(y_pred=y_pred, timings=PREDICT_TIMINGS),
    ):
        estimator.fit(X, y if Est is TabPFNRegressor else y.astype(int))
        assert estimator.get_timings() == {"fit": FIT_TIMINGS, "predict": None}
        estimator.predict(X)

    assert estimator.fit_timings_ == FIT_TIMINGS
    assert estimator.last_predict_timings == PREDICT_TIMINGS
    assert estimator.get_timings() == {"fit": FIT_TIMINGS, "predict": PREDICT_TIMINGS}


def test_estimator_restored_without_fit_has_no_timings():
    estimator = TabPFNRegressor()
    estimator.model_id_ = UUID(FIT_ID)
    assert estimator.get_timings() == {"fit": None, "predict": None}


def test_chunked_prediction_timings_add_up_per_stage():
    assert _combine_timings([None, None]) is None
    assert _combine_timings(
        [
            {"predict_s": 1.0, "test_set_transform_s": None},
            {"predict_s": 2.5, "test_set_transform_s": 0.5},
            None,
        ]
    ) == {"predict_s": 3.5, "test_set_transform_s": 0.5}
