import time
import unittest
from contextlib import contextmanager
from typing import Any, cast
from uuid import UUID
from unittest.mock import Mock, patch

import httpx
import numpy as np
from pydantic import ValidationError
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn_client.client import (
    _MIN_RETRY_INTERVAL_SECS,
    GetSettingsResponse,
    RetryableServerError,
    ServiceClient,
)
from tabpfn_client.api_models import (
    ClassifierConfig,
    DuplicateTrainSetErrorResponse,
    FitResponse,
    FitStatus,
    GetFitStatusResponse,
    RegressorConfig,
    RegressorOutputType,
    RegressorPredictParams,
    AsyncSettings,
    ClassifierFitTaskConfig,
)
from tests.mock_tabpfn_server import with_mock_server


def _api_settings_payload(
    max_cells=100_000_000,
    max_cols=2_000,
    max_size_bytes=100_000_000,
    max_classes=10,
    max_rows=None,
    test_max_cells=None,
) -> dict[str, Any]:
    max_rows = max_cells if max_rows is None else max_rows
    test_max_cells = max_cells if test_max_cells is None else test_max_cells
    model_limit = {
        "train_set_max_rows": max_rows,
        "train_set_max_cells": max_cells,
        "test_set_max_rows": max_rows,
        "test_set_max_cells": test_max_cells,
        "test_set_max_rows_w_full_regression_output": max_rows,
        "max_cols": max_cols,
        "max_classes": max_classes,
        "predict_row_pairs_budget": 250_000 * 1_000_000,
    }
    return {
        "default_model_version": "v2.5",
        "max_model_limit": model_limit,
        "model_limits": {"v2.5": model_limit},
        "dataset_max_size_bytes": max_size_bytes,
        "async_settings": {
            "use_above_trainset_size_bytes": 50 * 1024 * 1024,
            "poll_timeout_secs": 7200.0,
        },
    }


def _fast_poll_settings() -> AsyncSettings:
    return AsyncSettings(
        use_above_trainset_size_bytes=50 * 1024 * 1024,
        poll_timeout_secs=7200.0,
    )


class TestServiceClient(unittest.TestCase):
    def setUp(self):
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = cast(
            "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]",
            train_test_split(X, y, test_size=0.33, random_state=42),
        )

        ServiceClient.reset_authorization()
        ServiceClient._api_settings = GetSettingsResponse(
            **_api_settings_payload(),
        )
        ServiceClient._api_settings_ts = time.monotonic()

    def tearDown(self):
        ServiceClient.reset_authorization()
        ServiceClient._api_settings = None
        ServiceClient._api_settings_ts = 0.0

    @staticmethod
    def _upload_info(url: str) -> dict:
        return {
            "signed_urls": [url],
            "expires_at": 1_700_000_000.0,
            "required_headers": {"x-test-header": "1"},
        }

    def _prepare_train_set_upload_response(self, train_set_upload_id: str) -> dict:
        return {
            "train_set_upload_id": train_set_upload_id,
            "x_train_info": self._upload_info("https://upload.example/x_train"),
            "y_train_info": self._upload_info("https://upload.example/y_train"),
        }

    def _prepare_test_set_upload_response(self, test_set_upload_id: str) -> dict:
        return {
            "test_set_upload_id": test_set_upload_id,
            "x_test_info": self._upload_info("https://upload.example/x_test"),
        }

    def _predict_response(self, prediction) -> dict:
        return {
            "prediction": prediction,
            "metadata": {
                "task": "classification",
                "package_version": "0.3.0rc1",
                "tabpfn_config": {},
                "test_set_num_rows": len(self.X_test),
                "test_set_num_cols": self.X_test.shape[1],
            },
        }

    @with_mock_server()
    def test_try_connection(self, mock_server):
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        self.assertTrue(ServiceClient.try_connection())

    @with_mock_server()
    def test_try_connection_with_invalid_server(self, mock_server):
        mock_server.router.get(mock_server.endpoints.root.path).respond(404)
        self.assertFalse(ServiceClient.try_connection())

    @with_mock_server()
    def test_try_connection_with_outdated_client_raises_runtime_error(
        self, mock_server
    ):
        mock_server.router.get(mock_server.endpoints.root.path).respond(
            426, json={"message": "Client version too old. ..."}
        )
        with self.assertRaises(RuntimeError) as cm:
            ServiceClient.try_connection()
        self.assertTrue(str(cm.exception).startswith("Client version too old."))

    @with_mock_server()
    def test_validate_email(self, mock_server):
        mock_server.router.post(mock_server.endpoints.validate_email.path).respond(
            200, json={"message": "dummy_message"}
        )
        self.assertEqual(ServiceClient.validate_email("dummy_email"), (True, ""))

    @with_mock_server()
    def test_validate_email_invalid(self, mock_server):
        mock_server.router.post(mock_server.endpoints.validate_email.path).respond(
            401, json={"message": "dummy_message"}
        )
        self.assertEqual(
            ServiceClient.validate_email("dummy_email"),
            (False, "dummy_message"),
        )

    @with_mock_server()
    def test_register_user(self, mock_server):
        mock_server.router.post(mock_server.endpoints.register.path).respond(
            200, json={"message": "dummy_message", "token": "DUMMY_TOKEN"}
        )
        self.assertEqual(
            ServiceClient.register(
                "dummy_email",
                "dummy_password",
                "dummy_password",
                "dummy_validation",
                {
                    "company": "dummy_company",
                    "use_case": "dummy_usecase",
                    "role": "dummy_role",
                    "contact_via_email": False,
                },
            ),
            (True, "dummy_message", "DUMMY_TOKEN"),
        )

    @with_mock_server()
    def test_register_user_with_invalid_email(self, mock_server):
        mock_server.router.post(mock_server.endpoints.register.path).respond(
            401, json={"message": "dummy_message", "token": None}
        )
        self.assertEqual(
            ServiceClient.register(
                "dummy_email",
                "dummy_password",
                "dummy_password",
                "dummy_validation",
                {
                    "company": "dummy_company",
                    "use_case": "dummy_usecase",
                    "role": "dummy_role",
                    "contact_via_email": False,
                },
            ),
            (False, "dummy_message", None),
        )

    @with_mock_server()
    def test_invalid_auth_token(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(401)
        self.assertFalse(ServiceClient.is_auth_token_outdated("fake_token"))

    @with_mock_server()
    def test_valid_auth_token(self, mock_server):
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        self.assertTrue(ServiceClient.is_auth_token_outdated("true_token"))

    @with_mock_server()
    def test_send_reset_password_email(self, mock_server):
        mock_server.router.post(
            mock_server.endpoints.send_reset_password_email.path
        ).respond(200, json={"message": "Password reset email sent!"})
        self.assertEqual(
            ServiceClient.send_reset_password_email("test"),
            (True, "Password reset email sent!"),
        )

    @with_mock_server()
    def test_send_verification_email(self, mock_server):
        mock_server.router.post(
            mock_server.endpoints.send_verification_email.path
        ).respond(200, json={"message": "Verification Email sent!"})
        self.assertEqual(
            ServiceClient.send_verification_email("test"),
            (True, "Verification Email sent!"),
        )

    @with_mock_server()
    def test_retrieve_greeting_messages(self, mock_server):
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": ["message_1", "message_2"]})
        self.assertEqual(
            ServiceClient.retrieve_greeting_messages(), ["message_1", "message_2"]
        )

    @with_mock_server()
    def test_predict_with_valid_train_set_and_test_set(self, mock_server):
        mock_server.router.post("/tabpfn/prepare_train_set_upload").respond(
            200,
            json=self._prepare_train_set_upload_response(
                "00000000-0000-0000-0000-000000000001"
            ),
        )
        mock_server.router.post("/tabpfn/fit").respond(
            200,
            json={
                "fitted_train_set_id": "00000000-0000-0000-0000-000000000002",
                "status": "completed",
            },
        )
        mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            200,
            json=self._prepare_test_set_upload_response(
                "00000000-0000-0000-0000-000000000003"
            ),
        )
        mock_server.router.post("/tabpfn/predict").respond(
            200,
            json=self._predict_response([1, 0, 1]),
        )

        ServiceClient.authorize("dummy_token")

        with patch.object(ServiceClient, "_upload_to_gcs") as mock_upload:
            fitted_train_set_id = ServiceClient.fit(
                self.X_train,
                self.y_train,
                tabpfn_systems=["preprocessing", "text"],
                task_config=ClassifierFitTaskConfig(),
            )
            pred = ServiceClient.predict(
                fitted_train_set_id=fitted_train_set_id,
                x_test=self.X_test,
                task_config=ClassifierConfig(),
            )

        self.assertEqual(
            fitted_train_set_id, UUID("00000000-0000-0000-0000-000000000002")
        )
        assert isinstance(pred.y_pred, np.ndarray)
        self.assertTrue(np.array_equal(pred.y_pred, [1, 0, 1]))
        self.assertEqual(pred.metadata["task"], "classification")
        self.assertEqual(mock_upload.call_count, 3)

    def test_raise_on_error_no_op_on_success(self):
        response = Mock()
        response.status_code = 200
        self.assertIsNone(ServiceClient._raise_on_error(response, "test"))
        # The body must stay untouched on success so streaming responses
        # (e.g. download) remain unread.
        response.read.assert_not_called()
        response.json.assert_not_called()

    def test_validate_response(self):
        response = Mock()

        response.status_code = 426
        response.json.return_value = {"message": "Client version too old."}
        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._validate_response(
                response, "test", success_model=FitResponse
            )
        self.assertEqual(str(cm.exception), "Client version too old.")

        response.status_code = 400
        response.json.return_value = {"message": "Some other error"}
        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._validate_response(
                response, "test", success_model=FitResponse
            )
        self.assertTrue(str(cm.exception).startswith("Fail to call test"))

    def test_validate_response_streamed_error_envelope(self):
        # Long-running endpoints can emit a chunked 200 whose body is
        # {"_streamed_error": True, "message": "..."} when the fit fails
        # mid-stream. The handler must surface the message, not let the
        # success-schema validation turn it into a misleading pydantic error.
        response = Mock()
        response.status_code = 200
        response.is_closed = True
        response.json.return_value = {
            "_streamed_error": True,
            "message": "thinking fits require at least 500 rows of training data; got 225.",
        }
        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._validate_response(response, "fit", success_model=FitResponse)
        self.assertIn("streamed", str(cm.exception))
        self.assertIn("500 rows", str(cm.exception))

    def test_validate_response_streamed_error_without_message_uses_reason_phrase(self):
        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._validate_response(
                self._http_response(200, json={"_streamed_error": True}),
                "fit",
                success_model=FitResponse,
            )
        self.assertEqual(str(cm.exception), "Fail to call fit with error: streamed, OK")

    def test_check_version(self):
        response = Mock()
        response.status_code = 426
        response.json.return_value = {"message": "Client version too old."}
        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._check_version(response)
        self.assertEqual(str(cm.exception), "Client version too old.")

        # Any status other than 426 is the caller's business.
        response.status_code = 400
        response.json.return_value = {"message": "Some other error"}
        self.assertIsNone(ServiceClient._check_version(response))

    # -- Tests pinning the _validate_response contract. --

    @staticmethod
    def _http_response(status_code: int, **kwargs) -> httpx.Response:
        return httpx.Response(
            status_code,
            request=httpx.Request("POST", "http://testserver/test"),
            **kwargs,
        )

    def test_raise_on_error_retryable_statuses_raise_retryable_error(self):
        for status in (408, 502, 503, 504):
            with self.assertRaises(RetryableServerError) as cm:
                ServiceClient._raise_on_error(self._http_response(status), "test")
            self.assertIn(f"[HTTP {status}]", str(cm.exception))

    def test_raise_on_error_includes_trace_id(self):
        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._raise_on_error(
                self._http_response(
                    500, json={"message": "boom", "trace_id": "abc-123"}
                ),
                "test",
            )
        self.assertEqual(
            str(cm.exception),
            "Fail to call test: [HTTP 500] boom. Report trace ID: abc-123.",
        )

    def test_raise_on_error_5xx_with_non_json_body(self):
        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._raise_on_error(
                self._http_response(500, text="<html>oops</html>"), "test"
            )
        self.assertEqual(
            str(cm.exception), "Fail to call test: [HTTP 500] Internal Server Error."
        )

    def test_raise_on_error_message_falls_back_to_reason_phrase(self):
        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._raise_on_error(self._http_response(403), "test")
        self.assertEqual(str(cm.exception), "Fail to call test: [HTTP 403] Forbidden.")

    def test_validate_response_registered_error_model_is_returned(self):
        # Statuses with a registered error model (e.g. the 409 dedup path)
        # are returned to the caller instead of raising.
        body = {
            "message": "duplicate train set",
            "train_set_upload_id": "00000000-0000-0000-0000-000000000001",
        }
        parsed = ServiceClient._validate_response(
            self._http_response(409, json=body),
            "test",
            success_model=FitResponse,
            error_models={409: DuplicateTrainSetErrorResponse},
        )
        self.assertIsInstance(parsed, DuplicateTrainSetErrorResponse)
        assert isinstance(parsed, DuplicateTrainSetErrorResponse)
        self.assertEqual(parsed.message, "duplicate train set")

    def test_validate_response_error_model_invalid_body_falls_back_to_generic_error(
        self,
    ):
        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._validate_response(
                self._http_response(409, json={"unexpected": "shape"}),
                "test",
                success_model=FitResponse,
                error_models={409: DuplicateTrainSetErrorResponse},
            )
        self.assertIn("[HTTP 409]", str(cm.exception))

    def test_validate_response_success_with_invalid_schema_raises(self):
        # A success body that does not match the success model surfaces the
        # pydantic error as-is.
        with self.assertRaises(ValidationError):
            ServiceClient._validate_response(
                self._http_response(200, json={"unexpected": "shape"}),
                "test",
                success_model=FitResponse,
            )

    def test_check_version_ignores_server_errors(self):
        r = ServiceClient._check_version(
            self._http_response(500, json={"message": "boom"})
        )
        self.assertIsNone(r)

    @with_mock_server()
    def test_fit_calls_prepare_and_fit_each_time(self, mock_server):
        prepare_route = mock_server.router.post("/tabpfn/prepare_train_set_upload")
        prepare_route.respond(
            200,
            json=self._prepare_train_set_upload_response(
                "00000000-0000-0000-0000-000000000001"
            ),
        )
        fit_route = mock_server.router.post("/tabpfn/fit")
        fit_route.respond(
            200,
            json={
                "fitted_train_set_id": "00000000-0000-0000-0000-000000000002",
                "status": "completed",
            },
        )

        ServiceClient.authorize("dummy_access_token")

        with patch.object(ServiceClient, "_upload_to_gcs"):
            fitted_train_set_id_1 = ServiceClient.fit(
                self.X_train,
                self.y_train,
                tabpfn_systems=["preprocessing", "text"],
                task_config=ClassifierFitTaskConfig(),
            )
            fitted_train_set_id_2 = ServiceClient.fit(
                self.X_train,
                self.y_train,
                tabpfn_systems=["preprocessing", "text"],
                task_config=ClassifierFitTaskConfig(),
            )

        self.assertEqual(fitted_train_set_id_1, fitted_train_set_id_2)
        self.assertEqual(prepare_route.call_count, 2)
        self.assertEqual(fit_route.call_count, 2)

    def test_fit_rejects_invalid_api_mode_before_any_request(self):
        # `api_mode` is coerced through the ApiMode enum up front, so a typo
        # fails fast with a clear ValueError instead of leaving `use_async`
        # unbound after the train set was already uploaded. Deliberately no
        # mock server: the error must fire before any request is attempted.
        for bad_mode in ("SYNC", "asdasd"):
            with self.assertRaises(ValueError, msg=bad_mode):
                ServiceClient.fit(
                    self.X_train,
                    self.y_train,
                    tabpfn_systems=["preprocessing", "text"],
                    task_config=ClassifierFitTaskConfig(),
                    # cast: deliberately smuggle a bad value past the ApiMode
                    # annotation to exercise the runtime check.
                    api_mode=cast(Any, bad_mode),
                )

    @with_mock_server()
    def test_fit_async_mode_polls_until_completed(self, mock_server):
        # In async mode POST /tabpfn/fit returns immediately with status=pending;
        # the client must poll GET /tabpfn/fit/{id} until a terminal state.
        import httpx

        fitted_train_set_id = "00000000-0000-0000-0000-000000000002"
        mock_server.router.post("/tabpfn/prepare_train_set_upload").respond(
            200,
            json=self._prepare_train_set_upload_response(
                "00000000-0000-0000-0000-000000000001"
            ),
        )
        mock_server.router.post("/tabpfn/fit").respond(
            200,
            json={"fitted_train_set_id": fitted_train_set_id, "status": "pending"},
        )
        status_route = mock_server.router.post("/tabpfn/get_fit_status")
        # First poll still pending, second poll completed — exercises the loop.
        status_route.side_effect = [
            httpx.Response(
                200,
                json={
                    "fitted_train_set_id": fitted_train_set_id,
                    "status": "pending",
                    # A 0 hint is clamped to the loop's floor; sleep is
                    # mocked below to keep the test fast.
                    "retry_in_secs": 0,
                },
            ),
            httpx.Response(
                200,
                json={
                    "fitted_train_set_id": fitted_train_set_id,
                    "status": "completed",
                },
            ),
        ]

        ServiceClient.authorize("dummy_access_token")

        with (
            patch.object(ServiceClient, "_upload_to_gcs"),
            patch.object(
                ServiceClient,
                "_resolve_async_settings",
                return_value=_fast_poll_settings(),
            ),
            patch("tabpfn_client.client.time.sleep") as mock_sleep,
        ):
            result = ServiceClient.fit(
                self.X_train,
                self.y_train,
                tabpfn_systems=["preprocessing", "text"],
                task_config=ClassifierFitTaskConfig(),
            )

        self.assertEqual(result, UUID(fitted_train_set_id))
        self.assertEqual(status_route.call_count, 2)
        # The server's 0 hint was clamped to the floor instead of
        # busy-polling with no sleep at all.
        mock_sleep.assert_called_once_with(_MIN_RETRY_INTERVAL_SECS)

    @with_mock_server()
    def test_fit_async_mode_raises_on_failed(self, mock_server):
        # A failed fit is reported as HTTP 200 + status=failed on the polling
        # endpoint, so the client must inspect the status field and raise.
        fitted_train_set_id = "00000000-0000-0000-0000-000000000002"
        mock_server.router.post("/tabpfn/prepare_train_set_upload").respond(
            200,
            json=self._prepare_train_set_upload_response(
                "00000000-0000-0000-0000-000000000001"
            ),
        )
        mock_server.router.post("/tabpfn/fit").respond(
            200,
            json={"fitted_train_set_id": fitted_train_set_id, "status": "pending"},
        )
        mock_server.router.post("/tabpfn/get_fit_status").respond(
            200,
            json={
                "fitted_train_set_id": fitted_train_set_id,
                "status": "failed",
                "error": "boom",
            },
        )

        ServiceClient.authorize("dummy_access_token")

        with (
            patch.object(ServiceClient, "_upload_to_gcs"),
            patch.object(
                ServiceClient,
                "_resolve_async_settings",
                return_value=_fast_poll_settings(),
            ),
        ):
            with self.assertRaises(RuntimeError) as cm:
                ServiceClient.fit(
                    self.X_train,
                    self.y_train,
                    tabpfn_systems=["preprocessing", "text"],
                    task_config=ClassifierFitTaskConfig(),
                )

        self.assertIn("boom", str(cm.exception))

    @with_mock_server()
    def test_predict_with_same_test_set_calls_prepare_and_predict_each_time(
        self, mock_server
    ):
        prepare_route = mock_server.router.post("/tabpfn/prepare_test_set_upload")
        prepare_route.respond(
            200,
            json=self._prepare_test_set_upload_response(
                "00000000-0000-0000-0000-000000000003"
            ),
        )
        predict_route = mock_server.router.post("/tabpfn/predict")
        predict_route.respond(
            200,
            json=self._predict_response([1, 0, 1]),
        )

        fitted_train_set_id = UUID("00000000-0000-0000-0000-000000000002")

        with patch.object(ServiceClient, "_upload_to_gcs"):
            pred_1 = ServiceClient.predict(
                fitted_train_set_id=fitted_train_set_id,
                x_test=self.X_test,
                task_config=ClassifierConfig(),
            )
            pred_2 = ServiceClient.predict(
                fitted_train_set_id=fitted_train_set_id,
                x_test=self.X_test,
                task_config=ClassifierConfig(),
            )

        assert isinstance(pred_1.y_pred, np.ndarray)
        assert isinstance(pred_2.y_pred, np.ndarray)
        self.assertTrue(np.array_equal(pred_1.y_pred, pred_2.y_pred))
        self.assertEqual(prepare_route.call_count, 2)
        self.assertEqual(predict_route.call_count, 2)

    def test_get_settings_uses_cache(self):
        ServiceClient._api_settings = None
        ServiceClient._api_settings_ts = 0.0

        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = _api_settings_payload(
            max_cells=123,
            max_cols=12,
            max_size_bytes=456,
            max_classes=7,
        )

        with patch.object(
            ServiceClient.httpx_client, "get", return_value=response
        ) as m:
            first = ServiceClient.get_settings()
            second = ServiceClient.get_settings()

        assert first is not None
        self.assertEqual(first.dataset_max_size_bytes, 456)
        self.assertIs(first, second)
        self.assertEqual(m.call_count, 1)

    def test_get_settings_returns_stale_value_on_failure(self):
        stale = GetSettingsResponse(
            **_api_settings_payload(
                max_cells=100,
                max_cols=20,
                max_size_bytes=300,
                max_classes=4,
            ),
        )
        ServiceClient._api_settings = stale
        ServiceClient._api_settings_ts = time.monotonic() - 1_900

        with patch.object(
            ServiceClient.httpx_client, "get", side_effect=RuntimeError("boom")
        ):
            result = ServiceClient.get_settings()

        self.assertIs(result, stale)


class TestServiceClientPredictionNormalization(unittest.TestCase):
    def tearDown(self):
        ServiceClient.reset_authorization()
        ServiceClient._api_settings = None
        ServiceClient._api_settings_ts = 0.0

    @staticmethod
    def _upload_info(url: str) -> dict:
        return {
            "signed_urls": [url],
            "expires_at": 1_700_000_000.0,
            "required_headers": {"x-test-header": "1"},
        }

    def _prepare_test_set_upload_response(self, test_set_upload_id: str) -> dict:
        return {
            "test_set_upload_id": test_set_upload_id,
            "x_test_info": self._upload_info("https://upload.example/x_test"),
        }

    @staticmethod
    def _predict_response(prediction) -> dict:
        return {
            "prediction": prediction,
            "metadata": {
                "task": "regression",
                "package_version": "0.3.0rc1",
                "tabpfn_config": {},
                "test_set_num_rows": 2,
                "test_set_num_cols": 1,
            },
        }

    @with_mock_server()
    def test_predict_converts_none_in_dict_prediction_to_nan(self, mock_server):
        mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            200,
            json=self._prepare_test_set_upload_response(
                "00000000-0000-0000-0000-000000000003"
            ),
        )
        mock_server.router.post("/tabpfn/predict").respond(
            200,
            json=self._predict_response(
                {
                    "borders": [0.0, None, 2.0],
                    "logits": [[1.0, None], [None, 4.0]],
                }
            ),
        )

        with patch.object(ServiceClient, "get_settings", return_value=None):
            with patch.object(ServiceClient, "_upload_to_gcs"):
                pred = ServiceClient.predict(
                    fitted_train_set_id=UUID("00000000-0000-0000-0000-000000000002"),
                    x_test=np.array([[1.0], [2.0]]),
                    task_config=RegressorConfig(
                        predict_params=RegressorPredictParams(
                            output_type=RegressorOutputType.FULL
                        )
                    ),
                )

        y_pred = cast("dict[str, np.ndarray]", pred.y_pred)
        self.assertTrue(np.issubdtype(y_pred["borders"].dtype, np.floating))
        self.assertTrue(np.issubdtype(y_pred["logits"].dtype, np.floating))
        np.testing.assert_allclose(
            y_pred["borders"],
            np.array([0.0, np.nan, 2.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            y_pred["logits"],
            np.array([[1.0, np.nan], [np.nan, 4.0]]),
            equal_nan=True,
        )


class _FakeTime:
    """Deterministic stand-in for the `time` module inside the polling loop:
    the clock only advances when the loop sleeps."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, secs: float) -> None:
        self.sleeps.append(secs)
        self.now += secs


def _fit_status(
    status: FitStatus,
    retry_in_secs: float | None = None,
    error: str | None = None,
) -> GetFitStatusResponse:
    return GetFitStatusResponse(
        fitted_train_set_id=TestWaitForFit.FIT_ID,
        status=status,
        retry_in_secs=retry_in_secs,
        error=error,
    )


class TestWaitForFit(unittest.TestCase):
    """Pin the `_wait_for_fit` polling contract with a fake clock: the first
    poll is immediate, sleeps run between polls following the server's
    `retry_in_secs` hint (5s fallback until the first hint), and the deadline
    is a hard bound — the loop gives up as soon as the next poll could no
    longer start before the deadline."""

    FIT_ID = UUID("00000000-0000-0000-0000-000000000002")
    FALLBACK_INTERVAL = 5.0
    MIN_INTERVAL = _MIN_RETRY_INTERVAL_SECS

    @contextmanager
    def _patched(self, status_outcomes, poll_timeout: float = 100.0):
        """Patch the loop's clock, deadline settings, and status calls.

        `status_outcomes` is a Mock side_effect: a list of responses and/or
        exceptions, a single exception (raised on every call), or a callable.
        """
        fake_time = _FakeTime()
        settings = AsyncSettings(
            use_above_trainset_size_bytes=50 * 1024 * 1024,
            poll_timeout_secs=poll_timeout,
        )
        with (
            patch("tabpfn_client.client.time", fake_time),
            patch.object(
                ServiceClient, "_resolve_async_settings", return_value=settings
            ),
            patch.object(
                ServiceClient, "_get_fit_status", side_effect=status_outcomes
            ) as mock_status,
        ):
            yield fake_time, mock_status

    def test_completed_immediately_polls_once_without_sleeping(self):
        with self._patched([_fit_status(FitStatus.COMPLETED)]) as (fake_time, mock):
            ServiceClient._wait_for_fit(self.FIT_ID)
        self.assertEqual(mock.call_count, 1)
        self.assertEqual(fake_time.sleeps, [])

    def test_pending_sleeps_fallback_interval_between_polls(self):
        with self._patched(
            [_fit_status(FitStatus.PENDING), _fit_status(FitStatus.COMPLETED)]
        ) as (fake_time, mock):
            ServiceClient._wait_for_fit(self.FIT_ID)
        self.assertEqual(mock.call_count, 2)
        self.assertEqual(fake_time.sleeps, [self.FALLBACK_INTERVAL])

    def test_server_retry_hint_sets_interval(self):
        with self._patched(
            [
                _fit_status(FitStatus.PENDING, retry_in_secs=1.5),
                _fit_status(FitStatus.PENDING, retry_in_secs=2.5),
                _fit_status(FitStatus.COMPLETED),
            ]
        ) as (fake_time, _):
            ServiceClient._wait_for_fit(self.FIT_ID)
        self.assertEqual(fake_time.sleeps, [1.5, 2.5])

    def test_zero_or_negative_retry_hint_clamped_to_floor(self):
        # A zero or negative hint (server bug / clock skew) must neither
        # busy-poll the endpoint nor crash time.sleep with a ValueError;
        # both are clamped to the loop's floor.
        with self._patched(
            [
                _fit_status(FitStatus.PENDING, retry_in_secs=0.0),
                _fit_status(FitStatus.PENDING, retry_in_secs=-30.0),
                _fit_status(FitStatus.COMPLETED),
            ]
        ) as (fake_time, _):
            ServiceClient._wait_for_fit(self.FIT_ID)
        self.assertEqual(fake_time.sleeps, [self.MIN_INTERVAL, self.MIN_INTERVAL])

    def test_server_retry_hint_persists_when_later_responses_omit_it(self):
        with self._patched(
            [
                _fit_status(FitStatus.PENDING, retry_in_secs=2.0),
                _fit_status(FitStatus.PENDING),  # no hint: keep the last one
                _fit_status(FitStatus.COMPLETED),
            ]
        ) as (fake_time, _):
            ServiceClient._wait_for_fit(self.FIT_ID)
        self.assertEqual(fake_time.sleeps, [2.0, 2.0])

    def test_deadline_is_a_hard_bound_and_raises_timeout(self):
        # timeout 12, interval 5: polls at t=0/5/10. After the t=10 poll only
        # 2s remain — not enough for another full interval — so the loop gives
        # up without sleeping to the deadline and polling once more past it.
        with self._patched(
            lambda **kwargs: _fit_status(FitStatus.PENDING), poll_timeout=12.0
        ) as (fake_time, mock):
            with self.assertRaises(TimeoutError) as cm:
                ServiceClient._wait_for_fit(self.FIT_ID)
        self.assertIn("did not reach a terminal state", str(cm.exception))
        # The fit was genuinely still pending: no poll error to chain.
        self.assertIsNone(cm.exception.__cause__)
        self.assertEqual(fake_time.sleeps, [5.0, 5.0])
        self.assertEqual(mock.call_count, 3)

    def test_failed_fit_raises_runtime_error_without_sleeping(self):
        with self._patched([_fit_status(FitStatus.FAILED, error="boom")]) as (
            fake_time,
            _,
        ):
            with self.assertRaises(RuntimeError) as cm:
                ServiceClient._wait_for_fit(self.FIT_ID)
        self.assertIn("boom", str(cm.exception))
        self.assertEqual(fake_time.sleeps, [])

    def test_transient_errors_retry_until_deadline(self):
        # Each swallowed error is logged, and the TimeoutError chains the last
        # one so the real cause of a failing poll loop is not discarded.
        # timeout 10, interval 5: polls at t=0/5 fail; after the t=5 poll the
        # 5s remaining can't fit another full interval, so the loop gives up.
        with self._patched(httpx.ConnectError("down"), poll_timeout=10.0) as (
            fake_time,
            mock,
        ):
            with (
                self.assertLogs("tabpfn_client.client", level="WARNING") as logs,
                self.assertRaises(TimeoutError) as cm,
            ):
                ServiceClient._wait_for_fit(self.FIT_ID)
        self.assertEqual(fake_time.sleeps, [5.0])
        self.assertEqual(mock.call_count, 2)
        self.assertIsInstance(cm.exception.__cause__, httpx.ConnectError)
        self.assertIn("The last status poll failed: down", str(cm.exception))
        self.assertEqual(len(logs.output), 2)
        self.assertIn(str(self.FIT_ID), logs.output[0])
        self.assertIn("down", logs.output[0])

    def test_transient_error_then_recovery(self):
        with self._patched(
            [httpx.ConnectError("blip"), _fit_status(FitStatus.COMPLETED)]
        ) as (fake_time, mock):
            ServiceClient._wait_for_fit(self.FIT_ID)
        self.assertEqual(mock.call_count, 2)
        self.assertEqual(fake_time.sleeps, [self.FALLBACK_INTERVAL])
