import os
import unittest
from unittest.mock import Mock, patch

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import json

from tabpfn_client.client import ServiceClient, ModelType
from tabpfn_client.constants import CACHE_DIR
from tests.mock_tabpfn_server import with_mock_server


class TestServiceClient(unittest.TestCase):
    def setUp(self):
        # setup data
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33, random_state=42
        )

        ServiceClient.dataset_uid_cache_manager.file_path = (
            CACHE_DIR / "test_dataset_cache"
        )
        ServiceClient.dataset_uid_cache_manager.cache = (
            ServiceClient.dataset_uid_cache_manager.load_cache()
        )

    def tearDown(self):
        try:
            os.remove(CACHE_DIR / "test_dataset_cache")
        except OSError:
            pass

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
        self.assertTrue(ServiceClient.validate_email("dummy_email")[0])

    @with_mock_server()
    def test_validate_email_invalid(self, mock_server):
        mock_server.router.post(mock_server.endpoints.validate_email.path).respond(
            401, json={"detail": "dummy_message"}
        )
        self.assertFalse(ServiceClient.validate_email("dummy_email")[0])
        self.assertEqual(
            "dummy_message", ServiceClient.validate_email("dummy_email")[1]
        )

    @with_mock_server()
    def test_register_user(self, mock_server):
        mock_server.router.post(mock_server.endpoints.register.path).respond(
            200, json={"message": "dummy_message", "token": "DUMMY_TOKEN"}
        )
        self.assertTrue(
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
            )[0]
        )

    @with_mock_server()
    def test_register_user_with_invalid_email(self, mock_server):
        mock_server.router.post(mock_server.endpoints.register.path).respond(
            401, json={"detail": "dummy_message", "token": None}
        )
        self.assertFalse(
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
            )[0]
        )

    @with_mock_server()
    def test_register_user_with_invalid_validation_link(self, mock_server):
        mock_server.router.post(mock_server.endpoints.register.path).respond(
            401, json={"detail": "dummy_message", "token": None}
        )
        self.assertFalse(
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
            )[0]
        )

    @with_mock_server()
    def test_register_user_with_limit_reached(self, mock_server):
        mock_server.router.post(mock_server.endpoints.register.path).respond(
            401, json={"detail": "dummy_message", "token": "DUMMY_TOKEN"}
        )
        self.assertFalse(
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
            )[0]
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
        dummy_json = {"train_set_uid": "5"}
        mock_server.router.post(mock_server.endpoints.fit.path).respond(
            200, json=dummy_json
        )
        ServiceClient.authorize("dummy_token")
        ServiceClient.fit(self.X_train, self.y_train, model_type=ModelType.TABPFN)

        dummy_result = {"test_set_uid": "dummy_uid", "classification": [1, 2, 3]}
        mock_server.router.post(mock_server.endpoints.predict.path).respond(
            200,
            content=f"data: {json.dumps({'event': 'result', 'data': dummy_result})}\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

        pred = ServiceClient.predict(
            train_set_uid=dummy_json["train_set_uid"],
            x_test=self.X_test,
            model_type=ModelType.TABPFN,
            task="classification",
        )
        self.assertTrue(np.array_equal(pred.y_pred, dummy_result["classification"]))

    def test_validate_response_no_error(self):
        response = Mock()
        response.status_code = 200
        r = ServiceClient._validate_response(response, "test")
        self.assertIsNone(r)

    def test_validate_response(self):
        response = Mock()
        # Test for "Client version too old." error
        response.status_code = 426
        response.json.return_value = {"detail": "Client version too old."}
        with self.assertRaises(RuntimeError) as cm:
            ServiceClient._validate_response(response, "test")
        self.assertEqual(str(cm.exception), "Client version too old.")

        # Test for "Some other error" which is translated to a generic failure message
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

        # Errors that have nothing to do with client version should be skipped.
        response = Mock()
        response.status_code = 400
        response.json.return_value = {"detail": "Some other error"}
        r = ServiceClient._validate_response(response, "test", only_version_check=True)
        self.assertIsNone(r)

    @with_mock_server()
    def test_fit_with_caching(self, mock_server):
        """
        Test that calling fit with the same training set multiple times uses the cache and
        only calls the fit endpoint once.
        """
        ServiceClient.authorize("dummy_access_token")

        # Mock the fit endpoint to return a fixed train_set_uid
        with patch.object(
            ServiceClient.httpx_client,
            "stream",
            wraps=ServiceClient.httpx_client.stream,
        ) as mock_stream:
            # Set up the mock streaming response for fit
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_lines = Mock(
                return_value=[
                    '{"status": "complete", "train_set_uid": "dummy_train_set_uid"}'
                ]
            )
            mock_response.__enter__ = Mock(return_value=mock_response)
            mock_response.__exit__ = Mock(return_value=None)
            mock_stream.return_value = mock_response

            # First upload
            train_set_uid1 = ServiceClient.fit(
                self.X_train, self.y_train, model_type=ModelType.TABPFN
            )

            # Second upload with the same data
            train_set_uid2 = ServiceClient.fit(
                self.X_train, self.y_train, model_type=ModelType.TABPFN
            )

            # The train_set_uid should be the same due to caching
            self.assertEqual(train_set_uid1, train_set_uid2)

            # The fit endpoint should have been called only once
            mock_stream.assert_called_once()

    def test_predict_with_caching(self):
        """
        Test that making predictions with the same test set uses the cache and
        avoids re-uploading the test set.
        """
        ServiceClient.authorize("dummy_access_token")

        # Mock the fit and predict endpoints
        with (
            patch.object(
                ServiceClient.httpx_client,
                "post",
                wraps=ServiceClient.httpx_client.post,
            ) as mock_post,
            patch.object(
                ServiceClient.httpx_client,
                "stream",
                wraps=ServiceClient.httpx_client.stream,
            ) as mock_stream,
        ):
            # Mock responses
            def side_effect(*args, **kwargs):
                if kwargs.get("url") == ServiceClient.server_endpoints.fit.path:
                    # Provide a context-managed streaming response for fit
                    response = Mock()
                    response.status_code = 200
                    response.iter_lines = Mock(
                        return_value=[
                            '{"status": "complete", "train_set_uid": "dummy_train_set_uid"}'
                        ]
                    )
                    response.__enter__ = Mock(return_value=response)
                    response.__exit__ = Mock(return_value=None)
                    return response
                elif kwargs.get("url") == ServiceClient.server_endpoints.predict.path:
                    response = Mock()
                    response.status_code = 200
                    response.headers = {"Content-Type": "text/event-stream"}
                    response.iter_bytes = Mock(
                        return_value=iter(
                            [
                                'data: {"event": "result", "data": {"classification": [1, 2, 3], "test_set_uid": "dummy_test_set_uid"}}\n\n'.encode()
                            ]
                        )
                    )
                    response.__enter__ = Mock(return_value=response)
                    response.__exit__ = Mock(return_value=None)
                    return response
                else:
                    return Mock(status_code=404)

            mock_post.side_effect = side_effect
            mock_stream.side_effect = side_effect

            # Upload train set
            train_set_uid = ServiceClient.fit(
                self.X_train, self.y_train, model_type=ModelType.TABPFN
            )

            # First prediction
            pred1 = ServiceClient.predict(
                train_set_uid=train_set_uid,
                x_test=self.X_test,
                model_type=ModelType.TABPFN,
                task="classification",
            )

            # Second prediction with the same test set
            pred2 = ServiceClient.predict(
                train_set_uid=train_set_uid,
                x_test=self.X_test,
                model_type=ModelType.TABPFN,
                task="classification",
            )

            # The predictions should be the same
            self.assertTrue(np.array_equal(pred1.y_pred, pred2.y_pred))

            # The predict endpoint should have been called twice
            self.assertEqual(
                mock_post.call_count + mock_stream.call_count, 3
            )  # 1 for fit, 2 for predict

            # Check that the test set was uploaded only once (first predict call)
            upload_calls = [
                c
                for c in mock_stream.call_args_list
                if (
                    c.kwargs.get("url") == ServiceClient.server_endpoints.predict.path
                    and "files" in c.kwargs
                )
            ]
            self.assertEqual(len(upload_calls), 1)

    def test_predict_with_invalid_cached_uids(self):
        """
        Test that when the cached UIDs are invalid, the client re-uploads the datasets
        and retries the prediction.
        """
        ServiceClient.authorize("dummy_access_token")

        # Mock the fit and predict endpoints
        with patch.object(
            ServiceClient.httpx_client,
            "stream",
            wraps=ServiceClient.httpx_client.stream,
        ) as mock_stream:
            # Mock responses with side effects to simulate invalid cached UIDs
            def side_effect(*args, **kwargs):
                if kwargs.get("url") == ServiceClient.server_endpoints.fit.path:
                    # Provide a context-managed streaming response for fit
                    response = Mock()
                    response.status_code = 200
                    response.iter_lines = Mock(
                        return_value=[
                            '{"status": "complete", "train_set_uid": "dummy_train_set_uid"}'
                        ]
                    )
                    response.__enter__ = Mock(return_value=response)
                    response.__exit__ = Mock(return_value=None)
                    return response
                elif kwargs.get("url") == ServiceClient.server_endpoints.predict.path:
                    # Simulate invalid UID on first call, success on second
                    if side_effect.call_count == 2:
                        response = Mock()
                        response.status_code = 400
                        response.json.return_value = {
                            "detail": "Invalid train or test set uid"
                        }
                        response.__enter__ = Mock(return_value=response)
                        response.__exit__ = Mock(return_value=None)
                        return response
                    else:
                        # Successful prediction after re-upload
                        response = Mock()
                        response.status_code = 200
                        response.headers = {"Content-Type": "text/event-stream"}
                        response.iter_bytes = Mock(
                            return_value=iter(
                                [
                                    'data: {"event": "result", "data": {"classification": [1, 2, 3], "test_set_uid": "new_dummy_test_set_uid"}}\n\n'.encode()
                                ]
                            )
                        )
                        response.__enter__ = Mock(return_value=response)
                        response.__exit__ = Mock(return_value=None)
                        return response
                else:
                    return Mock(status_code=404)

            side_effect.call_count = 0

            def side_effect_counter(*args, **kwargs):
                side_effect.call_count += 1
                return side_effect(*args, **kwargs)

            mock_stream.side_effect = side_effect_counter

            # Upload train set
            train_set_uid = ServiceClient.fit(
                self.X_train, self.y_train, model_type=ModelType.TABPFN
            )

            # Attempt prediction, which should fail and trigger retry
            pred = ServiceClient.predict(
                train_set_uid=train_set_uid,
                x_test=self.X_test,
                model_type=ModelType.TABPFN,
                task="classification",
                X_train=self.X_train,
                y_train=self.y_train,
            )

            # The predictions should be as expected
            self.assertTrue(np.array_equal(pred.y_pred, [1, 2, 3]))

            # The predict endpoint should have been called twice due to retry
            self.assertEqual(
                mock_stream.call_count, 4
            )  # 1 fit + 2 predict + 1 re-upload

            # Ensure that fit was called again (re-upload)
            upload_calls = [
                call
                for call in mock_stream.call_args_list
                if call.kwargs.get("url") == ServiceClient.server_endpoints.fit.path
            ]
            self.assertEqual(len(upload_calls), 2)

    def test_dataset_cache_manager(self):
        """
        Test the DatasetCacheManager's basic functionality: adding, retrieving,
        and deleting dataset UIDs based on hashes.
        """
        # Create a fresh cache manager
        cache_manager = ServiceClient.dataset_uid_cache_manager

        # Mock dataset hashes and UIDs
        dataset_1 = "data1"
        dataset_uid_1 = "uid1"
        dataset_2 = "data2"
        dataset_uid_2 = "uid2"

        # Get hash by trying to get dataset_uid from cache
        _, dataset_hash_1 = cache_manager.get_dataset_uid(dataset_1)
        _, dataset_hash_2 = cache_manager.get_dataset_uid(dataset_2)

        # Add datasets to cache
        cache_manager.add_dataset_uid(dataset_hash_1, dataset_uid_1)
        cache_manager.add_dataset_uid(dataset_hash_2, dataset_uid_2)

        # Retrieve datasets from cache
        retrieved_uid_1, _ = cache_manager.get_dataset_uid(dataset_1)
        retrieved_uid_2, _ = cache_manager.get_dataset_uid(dataset_2)
        self.assertEqual(retrieved_uid_1, dataset_uid_1)
        self.assertEqual(retrieved_uid_2, dataset_uid_2)

        # Delete a dataset by UID
        deleted_hash = cache_manager.delete_uid(dataset_uid_1)
        self.assertEqual(deleted_hash, dataset_hash_1)

        # Ensure the deleted dataset is no longer in the cache
        self.assertIsNone(cache_manager.get_dataset_uid(dataset_1)[0])

    def test_cache_limit(self):
        """
        Test that the cache does not exceed its limit and evicts the oldest entries.
        """
        cache_manager = ServiceClient.dataset_uid_cache_manager
        cache_manager.cache_limit = 3  # Set a small limit for testing

        # Add more datasets than the cache limit
        for i in range(5):
            cache_manager.add_dataset_uid(f"hash{i}", f"uid{i}")

        # The cache should only contain the last 3 added datasets
        expected_hashes = ["hash2", "hash3", "hash4"]
        actual_hashes = list(cache_manager.cache.keys())

        self.assertEqual(actual_hashes, expected_hashes)
        self.assertEqual(len(cache_manager.cache), 3)
