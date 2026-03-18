import unittest
from uuid import uuid4

import httpx

from tabpfn_client.api_models import (
    DuplicateTrainSetErrorResponse,
    ErrorResponse,
    FitResponse,
    PrepareTestSetUploadResponse,
)
from tabpfn_client.client import NeedsRefittingError, ServiceClient


class TestResponseValidation(unittest.TestCase):
    def test_validate_response_returns_typed_success_model(self):
        response = httpx.Response(
            200,
            json={
                "test_set_upload_id": str(uuid4()),
                "x_test_info": {
                    "signed_urls": ["https://example.com/upload"],
                    "expires_at": 123.0,
                    "required_headers": {"x-goog-meta-test": "1"},
                },
            },
        )

        parsed = ServiceClient._validate_response(
            response,
            "prepare_test_set_upload",
            response_models={200: PrepareTestSetUploadResponse},
        )

        self.assertIsInstance(parsed, PrepareTestSetUploadResponse)

    def test_validate_response_returns_typed_non_200_model(self):
        response = httpx.Response(
            409,
            json={
                "message": "duplicate",
                "error_code": "ALREADY_EXISTS",
                "train_set_upload_id": str(uuid4()),
            },
        )

        parsed = ServiceClient._validate_response(
            response,
            "prepare_train_set_upload",
            response_models={409: DuplicateTrainSetErrorResponse},
        )

        self.assertIsInstance(parsed, DuplicateTrainSetErrorResponse)

    def test_validate_response_runs_handler_for_typed_error_model(self):
        response = httpx.Response(
            404,
            json={
                "message": "train set is gone",
                "error_code": "NOT_FOUND",
            },
        )

        with self.assertRaises(NeedsRefittingError) as cm:
            ServiceClient._validate_response(
                response,
                "predict",
                response_models={404: ErrorResponse},
                handlers={404: ServiceClient._raise_not_found_error},
            )

        self.assertEqual(str(cm.exception), "train set is gone")

    def test_validate_response_invalid_success_schema_raises_runtime_error(self):
        response = httpx.Response(200, json={"unexpected": "payload"})

        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._validate_response(
                response,
                "fit",
                response_models={200: FitResponse},
            )

        self.assertIn("invalid response schema", str(cm.exception))

    def test_validate_response_invalid_non_200_schema_raises_http_status_error(self):
        request = httpx.Request("POST", "https://example.com/test")
        response = httpx.Response(
            409,
            json={"unexpected": "payload"},
            request=request,
        )

        with self.assertRaises(httpx.HTTPStatusError):
            ServiceClient._validate_response(
                response,
                "prepare_train_set_upload",
                response_models={409: DuplicateTrainSetErrorResponse},
            )

    def test_validate_response_200_without_model_fails_closed(self):
        response = httpx.Response(200, json={"ok": True})

        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._validate_response(
                response,
                "fit",
                response_models={409: DuplicateTrainSetErrorResponse},
            )

        self.assertIn("no response model configured for status 200", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
