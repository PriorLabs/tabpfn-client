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
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})
        init(use_server=True)

        tabpfn = TabPFNClassifier()

        # mock fitting
        mock_server.router.post(mock_server.endpoints.fit.path).respond(
            200, json={"train_set_uid": "5"}
        )
        tabpfn.fit(self.X_train, self.y_train)

        # mock prediction with SSE
        probas = np.random.rand(len(self.X_test), len(np.unique(self.y_train))).tolist()
        mock_server.router.post(mock_server.endpoints.predict.path).respond(
            200,
            content=f'data: {{"event": "result", "data": {{"classification": {probas}, "test_set_uid": "6"}}}}\n\n',
            headers={"Content-Type": "text/event-stream"},
        )
        pred = tabpfn.predict(self.X_test)
        self.assertEqual(pred.shape[0], self.X_test.shape[0])
