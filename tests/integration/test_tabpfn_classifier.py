import unittest
from typing import cast
from unittest.mock import patch

import httpx
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

from tabpfn_client import init, reset
from tabpfn_client import TabPFNClassifier
from tests.mock_tabpfn_server import with_mock_server
from tabpfn_client.service_wrapper import UserAuthenticationClient


class TestTabPFNClassifier(unittest.TestCase):
    def setUp(self):
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = cast(
            "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]",
            train_test_split(X, y, test_size=0.33, random_state=42),
        )

    def tearDown(self):
        reset()

    @with_mock_server()
    def test_use_remote_tabpfn_classifier(self, mock_server):
        # create dummy token file
        token_file = UserAuthenticationClient.CACHED_TOKEN_FILE
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("dummy token")

        # mock connection and authentication
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})
        init(use_server=True)

        tabpfn = TabPFNClassifier()

        # mock fitting
        mock_server.router.post("/tabpfn/prepare_train_set_upload").respond(
            409,
            json={
                "message": "duplicate",
                "error_code": "DUPLICATE_TRAIN_SET",
                "train_set_upload_id": "00000000-0000-0000-0000-000000000001",
            },
        )
        mock_server.router.post(mock_server.endpoints.fit.path).respond(
            200,
            json={
                "fitted_train_set_id": "00000000-0000-0000-0000-000000000002",
                "status": "completed",
            },
        )
        tabpfn.fit(self.X_train, self.y_train)

        mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            409,
            json={
                "message": "duplicate",
                "error_code": "DUPLICATE_TEST_SET",
                "test_set_upload_id": "00000000-0000-0000-0000-000000000003",
            },
        )

        prediction = np.random.randint(
            0, len(np.unique(self.y_train)), len(self.X_test)
        ).tolist()
        mock_server.router.post(mock_server.endpoints.predict.path).respond(
            200,
            json={
                "prediction": prediction,
                "metadata": {
                    "task": "classification",
                    "package_version": "0.3.0rc1",
                    "tabpfn_config": {},
                    "test_set_num_rows": len(self.X_test),
                    "test_set_num_cols": self.X_test.shape[1],
                },
            },
        )
        pred = tabpfn.predict(self.X_test)
        self.assertEqual(pred.shape[0], self.X_test.shape[0])

    @with_mock_server()
    def test_use_remote_tabpfn_classifier_async_mode(self, mock_server):
        # End-to-end async-mode flow: POST /tabpfn/fit returns status=pending and
        # the client polls GET /tabpfn/fit/{id} until the fit is completed before
        # predict can run.
        token_file = UserAuthenticationClient.CACHED_TOKEN_FILE
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("dummy token")

        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})
        init(use_server=True)

        tabpfn = TabPFNClassifier()

        fitted_train_set_id = "00000000-0000-0000-0000-000000000002"
        mock_server.router.post("/tabpfn/prepare_train_set_upload").respond(
            409,
            json={
                "message": "duplicate",
                "error_code": "DUPLICATE_TRAIN_SET",
                "train_set_upload_id": "00000000-0000-0000-0000-000000000001",
            },
        )
        # Async fit: returns immediately with status=pending ...
        mock_server.router.post(mock_server.endpoints.fit.path).respond(
            200,
            json={"fitted_train_set_id": fitted_train_set_id, "status": "pending"},
        )
        # ... and the client polls until the fit reaches a terminal state.
        status_route = mock_server.router.get(f"/tabpfn/fit/{fitted_train_set_id}")
        status_route.side_effect = [
            httpx.Response(
                200,
                json={"fitted_train_set_id": fitted_train_set_id, "status": "pending"},
            ),
            httpx.Response(
                200,
                json={
                    "fitted_train_set_id": fitted_train_set_id,
                    "status": "completed",
                },
            ),
        ]

        with patch(
            "tabpfn_client.client.TABPFN_CLIENT_FIT_POLL_INTERVAL", 0
        ):
            tabpfn.fit(self.X_train, self.y_train)

        self.assertEqual(status_route.call_count, 2)

        mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            409,
            json={
                "message": "duplicate",
                "error_code": "DUPLICATE_TEST_SET",
                "test_set_upload_id": "00000000-0000-0000-0000-000000000003",
            },
        )
        prediction = np.random.randint(
            0, len(np.unique(self.y_train)), len(self.X_test)
        ).tolist()
        mock_server.router.post(mock_server.endpoints.predict.path).respond(
            200,
            json={
                "prediction": prediction,
                "metadata": {
                    "task": "classification",
                    "package_version": "0.3.0rc1",
                    "tabpfn_config": {},
                    "test_set_num_rows": len(self.X_test),
                    "test_set_num_cols": self.X_test.shape[1],
                },
            },
        )
        pred = tabpfn.predict(self.X_test)
        self.assertEqual(pred.shape[0], self.X_test.shape[0])
