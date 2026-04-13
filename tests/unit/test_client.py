import time
import unittest
from uuid import UUID
from unittest.mock import Mock, patch

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn_client.client import (
    GetDatasetLimitsResponse,
    NeedsRefittingError,
    ServiceClient,
)
from tests.mock_tabpfn_server import with_mock_server


class TestServiceClient(unittest.TestCase):
    def setUp(self):
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        ServiceClient.reset_authorization()
        ServiceClient._dataset_limits = GetDatasetLimitsResponse(
            max_cells=100_000_000,
            max_cols=2_000,
            max_size_bytes=100_000_000,
            max_classes=10,
        )
        ServiceClient._dataset_limits_ts = time.monotonic()

    def tearDown(self):
        ServiceClient.reset_authorization()
        ServiceClient._dataset_limits = None
        ServiceClient._dataset_limits_ts = 0.0

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
                "tabpfn_config": None,
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
            426, json={"detail": "Client version too old. ..."}
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
            401, json={"detail": "dummy_message"}
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
            401, json={"detail": "dummy_message", "token": None}
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
            json={"fitted_train_set_id": "00000000-0000-0000-0000-000000000002"},
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
                task="classification",
            )
            pred = ServiceClient.predict(
                fitted_train_set_id=fitted_train_set_id,
                x_test=self.X_test,
                task="classification",
            )

        self.assertEqual(
            fitted_train_set_id, UUID("00000000-0000-0000-0000-000000000002")
        )
        self.assertTrue(np.array_equal(pred.y_pred, [1, 0, 1]))
        self.assertEqual(pred.metadata["task"], "classification")
        self.assertEqual(mock_upload.call_count, 3)

    def test_validate_response_no_error(self):
        response = Mock()
        response.status_code = 200
        r = ServiceClient._validate_response(response, "test")
        self.assertIsNone(r)

    def test_validate_response(self):
        response = Mock()

        response.status_code = 426
        response.json.return_value = {"detail": "Client version too old."}
        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._validate_response(response, "test")
        self.assertEqual(str(cm.exception), "Client version too old.")

        response.status_code = 400
        response.json.return_value = {"detail": "Some other error"}
        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._validate_response(response, "test")
        self.assertTrue(str(cm.exception).startswith("Fail to call test"))

    def test_validate_response_only_version_check(self):
        response = Mock()
        response.status_code = 426
        response.json.return_value = {"detail": "Client version too old."}
        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._validate_response(response, "test", only_version_check=True)
        self.assertEqual(str(cm.exception), "Client version too old.")

        response.status_code = 400
        response.json.return_value = {"detail": "Some other error"}
        r = ServiceClient._validate_response(response, "test", only_version_check=True)
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
            json={"fitted_train_set_id": "00000000-0000-0000-0000-000000000002"},
        )

        ServiceClient.authorize("dummy_access_token")

        with patch.object(ServiceClient, "_upload_to_gcs"):
            fitted_train_set_id_1 = ServiceClient.fit(
                self.X_train,
                self.y_train,
                task="classification",
            )
            fitted_train_set_id_2 = ServiceClient.fit(
                self.X_train,
                self.y_train,
                task="classification",
            )

        self.assertEqual(fitted_train_set_id_1, fitted_train_set_id_2)
        self.assertEqual(prepare_route.call_count, 2)
        self.assertEqual(fit_route.call_count, 2)

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
                task="classification",
            )
            pred_2 = ServiceClient.predict(
                fitted_train_set_id=fitted_train_set_id,
                x_test=self.X_test,
                task="classification",
            )

        self.assertTrue(np.array_equal(pred_1.y_pred, pred_2.y_pred))
        self.assertEqual(prepare_route.call_count, 2)
        self.assertEqual(predict_route.call_count, 2)

    @with_mock_server()
    def test_predict_with_missing_fitted_train_set_raises_needs_refitting(
        self, mock_server
    ):
        mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            404,
            json={
                "message": "fitted train set missing",
                "error_code": "NOT_FOUND",
                "trace_id": "trace-123",
            },
        )

        with self.assertRaises(NeedsRefittingError):
            ServiceClient.predict(
                fitted_train_set_id=UUID("00000000-0000-0000-0000-000000000002"),
                x_test=self.X_test,
                task="classification",
            )

    def test_get_dataset_limits_uses_cache(self):
        ServiceClient._dataset_limits = None
        ServiceClient._dataset_limits_ts = 0.0

        response = Mock()
        response.raise_for_status = Mock()
        response.json.return_value = {
            "max_cells": 123,
            "max_cols": 12,
            "max_size_bytes": 456,
            "max_classes": 7,
        }

        with patch.object(
            ServiceClient.httpx_client, "get", return_value=response
        ) as m:
            first = ServiceClient.get_dataset_limits()
            second = ServiceClient.get_dataset_limits()

        self.assertEqual(first.max_size_bytes, 456)
        self.assertIs(first, second)
        self.assertEqual(m.call_count, 1)

    def test_get_dataset_limits_returns_stale_value_on_failure(self):
        stale = GetDatasetLimitsResponse(
            max_cells=100,
            max_cols=20,
            max_size_bytes=300,
            max_classes=4,
        )
        ServiceClient._dataset_limits = stale
        ServiceClient._dataset_limits_ts = time.monotonic() - 1_900

        with patch.object(
            ServiceClient.httpx_client, "get", side_effect=RuntimeError("boom")
        ):
            result = ServiceClient.get_dataset_limits()

        self.assertIs(result, stale)


class TestServiceClientPredictionNormalization(unittest.TestCase):
    def tearDown(self):
        ServiceClient.reset_authorization()
        ServiceClient._dataset_limits = None
        ServiceClient._dataset_limits_ts = 0.0

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
                "tabpfn_config": None,
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

        with patch.object(ServiceClient, "get_dataset_limits", return_value=None):
            with patch.object(ServiceClient, "_upload_to_gcs"):
                pred = ServiceClient.predict(
                    fitted_train_set_id=UUID("00000000-0000-0000-0000-000000000002"),
                    x_test=np.array([[1.0], [2.0]]),
                    task="regression",
                    predict_params={"output_type": "full"},
                )

        self.assertTrue(np.issubdtype(pred.y_pred["borders"].dtype, np.floating))
        self.assertTrue(np.issubdtype(pred.y_pred["logits"].dtype, np.floating))
        np.testing.assert_allclose(
            pred.y_pred["borders"],
            np.array([0.0, np.nan, 2.0]),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            pred.y_pred["logits"],
            np.array([[1.0, np.nan], [np.nan, 4.0]]),
            equal_nan=True,
        )
