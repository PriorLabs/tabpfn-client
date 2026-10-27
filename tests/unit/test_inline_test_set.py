"""Inline test sets: the predict request carries the test set, and the client
uploads it if the server refuses."""

import base64
import json
import time
import unittest
from unittest.mock import patch
from uuid import UUID

import httpx
import numpy as np

from tabpfn_client.api_models import (
    ClassifierConfig,
    ClassifierTabPFNConfig,
    FitMode,
    GetSettingsResponse,
)
from tabpfn_client.client import ServiceClient
from tabpfn_client.models import ApiMode
from tabpfn_client.errors import FittedModelNotFoundError, PredictFailedError
from tests.mock_tabpfn_server import with_mock_server
from tests.unit.test_client import _api_settings_payload

FIT_ID = UUID("00000000-0000-0000-0000-000000000002")
PREDICTION_ID = "00000000-0000-0000-0000-000000000004"
N_ROWS = 4
PREDICT_BODY = {
    "prediction": [0] * N_ROWS,
    "metadata": {
        "task": "classification",
        "package_version": "0.0.0",
        "tabpfn_config": {},
        "test_set_num_rows": N_ROWS,
        "test_set_num_cols": 3,
    },
}
PREPARE_BODY = {
    "test_set_upload_id": "00000000-0000-0000-0000-000000000003",
    "x_test_info": {
        "signed_urls": ["https://upload.example/x_test"],
        "expires_at": 1_700_000_000.0,
        "required_headers": {},
    },
}


def _settings(inline_max_bytes: int | None) -> dict:
    payload = _api_settings_payload()
    if inline_max_bytes is not None:
        payload["inline_test_set_max_parquet_bytes"] = inline_max_bytes
    return payload


def _sent(route) -> list[dict]:
    return [json.loads(call.request.content) for call in route.calls]


class TestInlineTestSet(unittest.TestCase):
    def setUp(self):
        self.X = np.random.RandomState(0).rand(N_ROWS, 3)
        self._cache_settings(5_000_000)

    def tearDown(self):
        ServiceClient._api_settings = None
        ServiceClient._api_settings_ts = 0.0

    def _cache_settings(self, inline_max_bytes: int | None) -> None:
        ServiceClient._api_settings = GetSettingsResponse(**_settings(inline_max_bytes))
        ServiceClient._api_settings_ts = time.monotonic()

    def _predict(
        self, X=None, api_mode=ApiMode.SYNC, task_config=None, n_train_rows=None
    ):
        with patch.object(ServiceClient, "_upload_to_gcs") as upload:
            ServiceClient.predict(
                fitted_train_set_id=FIT_ID,
                x_test=self.X if X is None else X,
                task_config=ClassifierConfig() if task_config is None else task_config,
                api_mode=api_mode,
                n_train_rows=n_train_rows,
            )
        return upload

    @with_mock_server()
    def test_inline_predict_skips_prepare_and_upload(self, mock_server):
        # The mock server fails any unrouted call, prepare included.
        predict = mock_server.router.post("/tabpfn/predict").respond(
            200, json=PREDICT_BODY
        )

        upload = self._predict()

        upload.assert_not_called()
        (sent,) = _sent(predict)
        self.assertIn("x_test_parquet_base64", sent)
        self.assertNotIn("test_set_upload_id", sent)

    def _uploads(self, mock_server) -> bool:
        prepare = mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            200, json=PREPARE_BODY
        )
        predict = mock_server.router.post("/tabpfn/predict").respond(
            200, json=PREDICT_BODY
        )
        self._predict()
        (sent,) = _sent(predict)
        return prepare.call_count == 1 and "x_test_parquet_base64" not in sent

    @with_mock_server()
    def test_limit_counts_parquet_bytes(self, mock_server):
        predict = mock_server.router.post("/tabpfn/predict").respond(
            200, json=PREDICT_BODY
        )
        self._predict()
        (sent,) = _sent(predict)
        length = len(base64.b64decode(sent["x_test_parquet_base64"]))

        self._cache_settings(length)
        self._predict()
        self.assertIn("x_test_parquet_base64", _sent(predict)[-1])

        self._cache_settings(length - 1)
        mock_server.router.routes.clear()
        self.assertTrue(self._uploads(mock_server))

    @with_mock_server()
    def test_switched_off_server_gets_an_upload(self, mock_server):
        self._cache_settings(0)
        self.assertTrue(self._uploads(mock_server))

    @with_mock_server()
    def test_server_without_inline_gets_an_upload(self, mock_server):
        self._cache_settings(None)
        self.assertTrue(self._uploads(mock_server))

    @with_mock_server()
    def test_refusal_uploads_that_call_and_refetches_settings(self, mock_server):
        # The cached limit is stale. The server has turned inline test sets off.
        settings = mock_server.router.get("/tabpfn/get_settings").respond(
            200, json=_settings(0)
        )
        prepare = mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            200, json=PREPARE_BODY
        )
        predict = mock_server.router.post("/tabpfn/predict")
        predict.side_effect = [
            httpx.Response(
                403, json={"message": "no", "error_code": "INLINE_TEST_SET_NOT_ENABLED"}
            ),
            httpx.Response(200, json=PREDICT_BODY),
            httpx.Response(200, json=PREDICT_BODY),
        ]

        self._predict()
        self._predict()

        self.assertEqual(settings.call_count, 1)
        self.assertEqual(prepare.call_count, 2)
        inline, *uploads = _sent(predict)
        self.assertIn("x_test_parquet_base64", inline)
        self.assertEqual(len(uploads), 2)
        for sent in uploads:
            self.assertNotIn("x_test_parquet_base64", sent)

    @with_mock_server()
    def test_too_large_uploads_only_that_call(self, mock_server):
        # A proxy, or a limit that is lower for a short time, refuses one request.
        mock_server.router.get("/tabpfn/get_settings").respond(
            200, json=_settings(5_000_000)
        )
        prepare = mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            200, json=PREPARE_BODY
        )
        predict = mock_server.router.post("/tabpfn/predict")
        predict.side_effect = [
            httpx.Response(413, text="Request Entity Too Large"),
            httpx.Response(200, json=PREDICT_BODY),
            httpx.Response(200, json=PREDICT_BODY),
        ]

        self._predict()
        self._predict()

        self.assertEqual(prepare.call_count, 1)
        refused, uploaded, inline_again = _sent(predict)
        self.assertIn("x_test_parquet_base64", refused)
        self.assertNotIn("x_test_parquet_base64", uploaded)
        self.assertIn("x_test_parquet_base64", inline_again)

    @with_mock_server()
    def test_server_rolled_back_below_inline_gets_an_upload(self, mock_server):
        # That server reads the request as an upload predict with no upload id,
        # and its message repeats the request body.
        prepare = mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            200, json=PREPARE_BODY
        )
        predict = mock_server.router.post("/tabpfn/predict")
        predict.side_effect = [
            httpx.Response(
                422,
                json={
                    "message": "[{'type': 'missing', 'loc': ('body', 'test_set_upload_id'), "
                    "'input': {'x_test_parquet_base64': 'SECRET'}}]",
                    "error_code": "REQUEST_VALIDATION_ERROR",
                },
            ),
            httpx.Response(200, json=PREDICT_BODY),
        ]

        self._predict()

        self.assertEqual(prepare.call_count, 1)
        _, uploaded = _sent(predict)
        self.assertNotIn("x_test_parquet_base64", uploaded)
        # The next predict asks the server for its settings again.
        self.assertIsNone(ServiceClient._api_settings)

    @with_mock_server()
    def test_other_errors_are_raised(self, mock_server):
        predict = mock_server.router.post("/tabpfn/predict")
        predict.side_effect = [
            httpx.Response(
                422, json={"message": "Not a Parquet file.", "error_code": "USER_ERROR"}
            ),
            httpx.Response(
                422,
                json={
                    "message": "Bad task_config.",
                    "error_code": "REQUEST_VALIDATION_ERROR",
                },
            ),
            httpx.Response(
                403, json={"message": "Forbidden.", "error_code": "USER_ERROR"}
            ),
        ]

        for message in ("Not a Parquet file", "Bad task_config", "Forbidden"):
            with self.assertRaisesRegex(RuntimeError, message):
                self._predict()

    @with_mock_server()
    def test_failed_prediction_with_not_found_code_is_a_failed_prediction(
        self, mock_server
    ):
        mock_server.router.post("/tabpfn/predict").respond(
            404,
            json={
                "message": "Gone.",
                "error_code": "NOT_FOUND",
                "fitted_test_set_id": PREDICTION_ID,
            },
        )

        with self.assertRaises(PredictFailedError) as raised:
            self._predict()
        self.assertNotIsInstance(raised.exception, FittedModelNotFoundError)

    @with_mock_server()
    def test_unknown_fit_raises_fitted_model_not_found(self, mock_server):
        mock_server.router.post("/tabpfn/predict").respond(
            404, json={"message": "Fitted train set not found"}
        )

        with self.assertRaises(FittedModelNotFoundError):
            self._predict()

    def _completed_result(self, mock_server):
        return mock_server.router.post("/tabpfn/get_predict_result").respond(
            200,
            json={
                **PREDICT_BODY,
                "fitted_test_set_id": PREDICTION_ID,
                "status": "completed",
            },
        )

    @with_mock_server()
    def test_background_job_uploads(self, mock_server):
        # The server takes inline test sets on a single predict request only.
        prepare = mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            200, json=PREPARE_BODY
        )
        submit = mock_server.router.post("/tabpfn/submit_predict_job").respond(
            200, json={"fitted_test_set_id": PREDICTION_ID}
        )
        self._completed_result(mock_server)

        self._predict(api_mode=ApiMode.ASYNC)

        self.assertEqual(prepare.call_count, 1)
        (sent,) = _sent(submit)
        self.assertNotIn("x_test_parquet_base64", sent)

    @with_mock_server()
    def test_inline_predict_still_running_is_polled(self, mock_server):
        mock_server.router.post("/tabpfn/predict").respond(
            202,
            json={
                "fitted_test_set_id": PREDICTION_ID,
                "status": "pending",
                "retry_in_secs": 0.0,
            },
        )
        result = self._completed_result(mock_server)

        upload = self._predict()

        upload.assert_not_called()
        self.assertEqual(result.call_count, 1)

    @with_mock_server()
    def test_auto_on_a_cached_fit_goes_inline(self, mock_server):
        # A KV-cache fit predicts with one request whatever its train rows.
        predict = mock_server.router.post("/tabpfn/predict").respond(
            200, json=PREDICT_BODY
        )

        self._predict(
            api_mode=ApiMode.AUTO,
            task_config=ClassifierConfig(
                tabpfn_config=ClassifierTabPFNConfig(fit_mode=FitMode.FIT_WITH_CACHE)
            ),
            n_train_rows=500_000,
        )

        (sent,) = _sent(predict)
        self.assertIn("x_test_parquet_base64", sent)

    @with_mock_server()
    def test_auto_on_a_large_fit_uploads_to_a_job(self, mock_server):
        prepare = mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            200, json=PREPARE_BODY
        )
        mock_server.router.post("/tabpfn/submit_predict_job").respond(
            200, json={"fitted_test_set_id": PREDICTION_ID}
        )
        self._completed_result(mock_server)

        self._predict(api_mode=ApiMode.AUTO, n_train_rows=500_000)

        self.assertEqual(prepare.call_count, 1)
