import unittest

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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
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
            200, json={"fitted_train_set_id": "00000000-0000-0000-0000-000000000002"}
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
                    "tabpfn_config": None,
                    "test_set_num_rows": len(self.X_test),
                    "test_set_num_cols": self.X_test.shape[1],
                },
            },
        )
        pred = tabpfn.predict(self.X_test)
        self.assertEqual(pred.shape[0], self.X_test.shape[0])
