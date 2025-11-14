import unittest
from unittest.mock import patch
import shutil
import json

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.exceptions import NotFittedError


from tabpfn_client import init, reset
from tabpfn_client.estimator import TabPFNClassifier
from tabpfn_client.service_wrapper import UserAuthenticationClient, InferenceClient
from tests.mock_tabpfn_server import with_mock_server
from tabpfn_client.constants import CACHE_DIR
from tabpfn_client import config
from tabpfn_client.client import PredictionResult


class TestTabPFNClassifierInit(unittest.TestCase):
    dummy_token = "dummy_token"

    def setUp(self):
        # set up dummy data
        reset()
        X, y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33
        )

    def tearDown(self):
        # remove cache dir
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    @patch("tabpfn_client.browser_auth.webbrowser.open", return_value=False)
    @patch("tabpfn_client.prompt_agent.PromptAgent.prompt_and_set_token")
    @with_mock_server()
    def test_init_remote_classifier(
        self,
        mock_server,
        mock_prompt_and_set_token,
        mock_webbrowser_open,
    ):
        mock_prompt_and_set_token.side_effect = (
            lambda: UserAuthenticationClient.set_token(self.dummy_token)
        )

        # mock server connection
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        mock_server.router.post(mock_server.endpoints.fit.path).respond(
            200, json={"train_set_uid": "5"}
        )

        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})

        mock_predict_response = [1, 0, 1]
        predict_route = mock_server.router.post(mock_server.endpoints.predict.path)
        predict_route.respond(
            200,
            content=f"data: {json.dumps({'event': 'result', 'data': {'classification': mock_predict_response, 'test_set_uid': '6'}})}\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

        init(use_server=True)
        self.assertTrue(mock_prompt_and_set_token.called)

        tabpfn = TabPFNClassifier(n_estimators=10)
        self.assertRaises(NotFittedError, tabpfn.predict, self.X_test)
        tabpfn.fit(self.X_train, self.y_train)
        self.assertTrue(mock_prompt_and_set_token.called)
        result = tabpfn.predict(self.X_test)
        print(f"result: {type(result)} {result}")
        self.assertTrue(np.all(mock_predict_response == result))
        # Checking for both %20 and + enconding of spaces
        # since httpx was inconsistent with its spacen encoding scheme
        self.assertTrue(
            "n_estimators%22%3A%2010" in str(predict_route.calls.last.request.url)
            or "n_estimators%22%3A+10" in str(predict_route.calls.last.request.url),
            "check that n_estimators is passed to the server",
        )

    @with_mock_server()
    def test_reuse_saved_access_token(self, mock_server):
        # mock connection and authentication
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})

        # create dummy token file
        token_file = UserAuthenticationClient.CACHED_TOKEN_FILE
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(self.dummy_token)

        # init is called without error
        init(use_server=True)

        # check if access token still exists
        self.assertTrue(UserAuthenticationClient.CACHED_TOKEN_FILE.exists())

    @with_mock_server()
    @patch("tabpfn_client.prompt_agent.PromptAgent.prompt_and_set_token")
    def test_invalid_saved_access_token(self, mock_server, mock_prompt_and_set_token):
        mock_prompt_and_set_token.side_effect = [RuntimeError]

        # mock connection and invalid authentication
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(401)

        # create dummy token file
        token_file = UserAuthenticationClient.CACHED_TOKEN_FILE
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text("invalid_token")

        self.assertRaises(RuntimeError, init, use_server=True)
        self.assertTrue(mock_prompt_and_set_token.called)

    @with_mock_server()
    def test_reset_on_remote_classifier(self, mock_server):
        # create dummy token file
        token_file = UserAuthenticationClient.CACHED_TOKEN_FILE
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(self.dummy_token)

        # init classifier as usual
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})
        init(use_server=True)

        # check if access token is saved
        self.assertTrue(UserAuthenticationClient.CACHED_TOKEN_FILE.exists())

        # reset
        reset()

        # check if access token is deleted
        self.assertFalse(UserAuthenticationClient.CACHED_TOKEN_FILE.exists())

        # check if config is reset
        self.assertFalse(config.Config.is_initialized)

    @patch(
        "tabpfn_client.prompt_agent.PromptAgent.prompt_terms_and_cond",
        return_value=False,
    )
    @patch("tabpfn_client.browser_auth.webbrowser.open", return_value=False)
    @patch("rich.console.Console.input", side_effect=["1"])
    @with_mock_server()  # TODO (leo): investigate why this needs to be the last decorator
    def test_decline_terms_and_cond(
        self,
        mock_server,
        mock_input,
        mock_webbrowser_open,
        mock_prompt_for_terms_and_cond,
    ):
        # mock connection
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)

        self.assertRaises(RuntimeError, init, use_server=True)
        self.assertTrue(mock_prompt_for_terms_and_cond.called)

    @with_mock_server()
    @patch("tabpfn_client.prompt_agent.PromptAgent.prompt_and_set_token")
    @patch(
        "tabpfn_client.prompt_agent.PromptAgent.prompt_terms_and_cond",
        return_value=True,
    )
    def test_cache_based_on_paper_version(
        self, mock_server, mock_prompt_for_terms_and_cond, mock_prompt_and_set_token
    ):
        """
        Check that the dataset cached is different for different paper_version
        and similar for the same paper_version
        """
        # a bit out of place but we don't want to skip init for this test
        mock_prompt_and_set_token.side_effect = (
            lambda: UserAuthenticationClient.set_token(self.dummy_token)
        )

        # mock server connection
        mock_server.router.get(mock_server.endpoints.root.path).respond(200)
        mock_server.router.get(mock_server.endpoints.protected_root.path).respond(200)
        fit_route = mock_server.router.post(mock_server.endpoints.fit.path)
        fit_route.respond(200, json={"train_set_uid": "5"})

        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})

        mock_predict_response = [[1, 0.0], [0.9, 0.1], [0.01, 0.99]]
        predict_route = mock_server.router.post(mock_server.endpoints.predict.path)
        predict_route.respond(
            200,
            content=f"data: {json.dumps({'event': 'result', 'data': {'classification': mock_predict_response, 'test_set_uid': '6'}})}\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

        init(use_server=True)

        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        test_X = np.random.rand(5, 5)

        # Initialize with paper_version=True
        tabpfn_true = TabPFNClassifier(paper_version=True)

        tabpfn_true.fit(X, y)
        tabpfn_true.predict(test_X)

        # Call fit and predict again with the same paper_version
        tabpfn_true.fit(X, y)
        tabpfn_true.predict(test_X)

        # Ensure fit endpoint is not called again
        self.assertEqual(
            fit_route.call_count,
            1,
            "Fit endpoint should not be called again with the same paper_version",
        )

        # Initialize with paper_version=False
        tabpfn_false = TabPFNClassifier(paper_version=False)

        tabpfn_false.fit(X, y)
        tabpfn_false.predict(test_X)

        # check fit is called
        self.assertEqual(
            fit_route.call_count,
            2,
            "Fit endpoint should be called again with a different paper_version",
        )

        # Call fit and predict again with the same paper_version
        tabpfn_false.fit(X, y)
        tabpfn_false.predict(test_X)

        # Ensure fit endpoint is not called again
        self.assertEqual(
            fit_route.call_count,
            2,
            "Fit endpoint should not be called again with the same paper_version",
        )

        # TODO: fix this
        # # Check that different cache entries are created for training set
        # cache_manager = ServiceClient.dataset_uid_cache_manager
        # X_serialized = common_utils.serialize_to_csv_formatted_bytes(X)
        # y_serialized = common_utils.serialize_to_csv_formatted_bytes(y)
        # uid_true_train, hash_true_train = cache_manager.get_dataset_uid(
        #     X_serialized, y_serialized, self.dummy_token, "_".join([])
        # )
        # uid_false_train, hash_false_train = cache_manager.get_dataset_uid(
        #     X_serialized,
        #     y_serialized,
        #     self.dummy_token,
        #     "_".join(["preprocessing", "text"]),
        # )

        # self.assertNotEqual(
        #     hash_true_train,
        #     hash_false_train,
        #     "Cache hash should differ based on paper_version for training set",
        # )

        # # Check that different cache entries are created for test set
        # test_X_serialized = common_utils.serialize_to_csv_formatted_bytes(test_X)
        # uid_true_test, hash_true_test = cache_manager.get_dataset_uid(
        #     test_X_serialized, uid_true_train, self.dummy_token, "_".join([])
        # )
        # uid_false_test, hash_false_test = cache_manager.get_dataset_uid(
        #     test_X_serialized,
        #     uid_false_train,
        #     self.dummy_token,
        #     "_".join(["preprocessing", "text"]),
        # )

        # self.assertNotEqual(
        #     hash_true_test,
        #     hash_false_test,
        #     "Cache hash should differ based on paper_version for test set",
        # )

        # # Verify that the cache entries are used correctly
        # self.assertIsNotNone(
        #     uid_true_train, "Training set cache should be used for paper_version=True"
        # )
        # self.assertIsNotNone(
        #     uid_false_train, "Training set cache should be used for paper_version=False"
        # )
        # self.assertIsNotNone(
        #     uid_true_test, "Test set cache should be used for paper_version=True"
        # )
        # self.assertIsNotNone(
        #     uid_false_test, "Test set cache should be used for paper_version=False"
        # )


class TestTabPFNClassifierInference(unittest.TestCase):
    def setUp(self):
        # skip init
        config.Config.is_initialized = True

    def tearDown(self):
        # undo setUp
        config.reset()

    def test_data_size_check_on_train_with_inconsistent_number_of_samples_raise_error(
        self,
    ):
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 11)
        tabpfn = TabPFNClassifier()

        with self.assertRaises(ValueError):
            tabpfn.fit(X, y)

    def test_data_size_check_on_train_with_oversized_data_raise_error(self):
        X = np.random.randn(50_001, 2001)
        y = np.random.randint(0, 2, 50_001)

        tabpfn = TabPFNClassifier()

        # test oversized columns
        with self.assertRaises(ValueError):
            tabpfn.fit(X[:10], y[:10])

        # test oversized rows
        with self.assertRaises(ValueError):
            tabpfn.fit(X[:, :10], y)

    def test_data_size_check_on_predict_with_oversized_data_raise_error(self):
        test_X = np.random.randn(50_001, 5)
        tabpfn = TabPFNClassifier()

        # skip fitting
        tabpfn.fitted_ = True

        # test oversized rows
        with self.assertRaises(ValueError):
            tabpfn.predict(test_X)

    def test_data_check_on_predict_with_valid_data_pass(self):
        test_X = np.random.randn(10, 5)
        tabpfn = TabPFNClassifier()

        # skip fitting
        tabpfn.fitted_ = True
        tabpfn.classes_ = np.array([0, 1])

        # mock prediction
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = PredictionResult(
                y_pred={"probas": np.random.rand(10, 2)}, metadata={}
            )
            tabpfn.predict(test_X)

    def test_only_allowed_parameters_passed_to_config(self):
        """Test that only allowed parameters are passed to the config."""
        ALLOWED_PARAMS = {
            "n_estimators",
            # TODO: put it back
            # "categorical_features_indices",
            "softmax_temperature",
            "average_before_softmax",
            "ignore_pretraining_limits",
            "inference_precision",
            "random_state",
            "inference_config",
            "model_path",
            "balance_probabilities",
            "paper_version",
            # These are removed in ServiceClient.predict
            "thinking_params",
            "thinking",
        }

        # Create classifier with various parameters
        classifier = TabPFNClassifier(
            n_estimators=5,
            softmax_temperature=0.8,
            paper_version=True,
            random_state=42,
            balance_probabilities=True,
        )

        # Skip fitting
        classifier.fitted_ = True
        classifier.last_train_set_uid = "dummy_uid"

        test_X = np.random.randn(10, 5)

        # Mock predict and capture config
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = PredictionResult(
                y_pred=np.random.rand(10, 2), metadata={}
            )
            classifier.predict(test_X)

            # Get the config that was passed to predict
            actual_config = mock_predict.call_args[1]["tabpfn_config"]

            # Check that only allowed parameters are present
            config_params = set(actual_config.keys())
            unexpected_params = config_params - ALLOWED_PARAMS
            missing_params = ALLOWED_PARAMS - config_params

            self.assertEqual(
                unexpected_params,
                set(),
                f"Found unexpected parameters in config: {unexpected_params}",
            )
            self.assertEqual(
                missing_params,
                set(),
                f"Missing required parameters in config: {missing_params}",
            )

    def test_predict_params_output_type(self):
        """Test that predict_params contains correct output_type."""
        classifier = TabPFNClassifier()
        classifier.fitted_ = True  # Skip fitting
        test_X = np.random.randn(10, 5)

        # Test predict() sets output_type to "preds"
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = PredictionResult(
                y_pred=np.random.rand(10), metadata={}
            )
            classifier.predict(test_X)

            predict_params = mock_predict.call_args[1]["predict_params"]
            self.assertEqual(predict_params, {"output_type": "preds"})

        # Test predict_proba() sets output_type to "probas"
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = PredictionResult(
                y_pred=np.random.rand(10, 2), metadata={}
            )
            classifier.predict_proba(test_X)

            predict_params = mock_predict.call_args[1]["predict_params"]
            self.assertEqual(predict_params, {"output_type": "probas"})

    def test_string_label_predictions(self):
        """Test that string labels in y are preserved in predictions."""
        X = np.random.rand(10, 5)
        test_X = np.random.rand(5, 5)

        # Test with pure string labels
        y_str = np.array(["cat", "dog"] * 5)
        clf_str = TabPFNClassifier()

        # Mock fit and predict
        with patch.object(InferenceClient, "fit") as mock_fit:
            mock_fit.return_value = "dummy_uid"
            clf_str.fit(X, y_str)

            with patch.object(InferenceClient, "predict") as mock_predict:
                mock_predict.return_value = PredictionResult(
                    y_pred=np.array(["cat", "dog"] * 5), metadata={}
                )
                y_pred = clf_str.predict(test_X)

                self.assertTrue(np.all(np.isin(y_pred, ["cat", "dog"])))
                self.assertTrue(
                    y_pred.dtype.kind in {"U", "O"}, "Predictions should be string type"
                )

        # Test with numeric-like strings
        y_num_str = np.array(["0", "1"] * 5)
        clf_num_str = TabPFNClassifier()

        with patch.object(InferenceClient, "fit") as mock_fit:
            mock_fit.return_value = "dummy_uid"
            clf_num_str.fit(X, y_num_str)

            with patch.object(InferenceClient, "predict") as mock_predict:
                mock_predict.return_value = PredictionResult(
                    y_pred=np.array(["0", "1"] * 5), metadata={}
                )
                y_pred = clf_num_str.predict(test_X)

                self.assertTrue(np.all(np.isin(y_pred, ["0", "1"])))
                self.assertTrue(
                    y_pred.dtype.kind in {"U", "O"}, "Predictions should be string type"
                )

    def test_predict_with_uncleaned_text(self):
        """Test predictions with long text (>2500 chars) and text containing commas."""
        # Skip initialization
        tabpfn = TabPFNClassifier()
        tabpfn.fitted_ = True
        tabpfn.classes_ = np.array([0, 1])  # Binary classification

        # Create test data with a mix of numeric and text features
        n_samples = 5
        n_features = 3

        # Create base numeric data
        X_numeric = np.random.randn(n_samples, n_features)

        # Convert to list for easier manipulation
        X = X_numeric.tolist()

        # Add text variations to test
        base_text = "very " * 500  # 2500 characters
        long_text = base_text + " extra text that should be truncated"
        text_with_commas = "very, " * 500  # Same length but with commas
        text_with_spaces = (
            "text\n\n with\t\tweird    spaces\r\nand\n\n\nlinebreaks" * 100
        )  # Text with various whitespace

        # Create variations of the same data with different text
        X_normal = [row + [base_text] for row in X]
        X_long = [row + [long_text] for row in X]
        X_commas = [row + [text_with_commas] for row in X]
        X_spaces = [row + [text_with_spaces] for row in X]

        # Convert to numpy arrays and make copies for comparison
        X_normal_array = np.array(X_normal)
        X_long_array = np.array(X_long)
        X_commas_array = np.array(X_commas)
        X_spaces_array = np.array(X_spaces)

        X_normal_copy = X_normal_array.copy()
        X_long_copy = X_long_array.copy()
        X_commas_copy = X_commas_array.copy()
        X_spaces_copy = X_spaces_array.copy()

        # Mock predictions
        expected_predictions = np.random.randint(0, 2, n_samples)
        expected_probas = np.random.rand(n_samples, 2)
        expected_probas = (
            expected_probas / expected_probas.sum(axis=1)[:, np.newaxis]
        )  # Normalize

        with patch.object(InferenceClient, "predict") as mock_predict:
            # Test predict()
            mock_predict.return_value = PredictionResult(
                y_pred=expected_predictions, metadata={}
            )

            # Test predictions for each variation
            pred_normal = tabpfn.predict(X_normal_array)
            pred_long = tabpfn.predict(X_long_array)
            pred_commas = tabpfn.predict(X_commas_array)
            pred_spaces = tabpfn.predict(X_spaces_array)

            # Verify input arrays were not modified
            np.testing.assert_array_equal(
                X_normal_array,
                X_normal_copy,
                "Input array with normal text was modified during prediction",
            )
            np.testing.assert_array_equal(
                X_long_array,
                X_long_copy,
                "Input array with long text was modified during prediction",
            )
            np.testing.assert_array_equal(
                X_commas_array,
                X_commas_copy,
                "Input array with comma text was modified during prediction",
            )
            np.testing.assert_array_equal(
                X_spaces_array,
                X_spaces_copy,
                "Input array with special spaces was modified during prediction",
            )

            # Verify predictions are returned as expected
            np.testing.assert_array_equal(pred_normal, expected_predictions)
            np.testing.assert_array_equal(pred_long, expected_predictions)
            np.testing.assert_array_equal(pred_commas, expected_predictions)
            np.testing.assert_array_equal(pred_spaces, expected_predictions)

            # Verify that long text (which should be truncated) gives same predictions as normal text
            np.testing.assert_array_equal(pred_normal, pred_long)

            # Test predict_proba()
            mock_predict.return_value = PredictionResult(
                y_pred=expected_probas, metadata={}
            )

            # Test probability predictions for each variation
            proba_normal = tabpfn.predict_proba(X_normal_array)
            proba_long = tabpfn.predict_proba(X_long_array)
            proba_commas = tabpfn.predict_proba(X_commas_array)
            proba_spaces = tabpfn.predict_proba(X_spaces_array)

            # Verify input arrays were not modified during predict_proba
            np.testing.assert_array_equal(
                X_normal_array,
                X_normal_copy,
                "Input array with normal text was modified during probability prediction",
            )
            np.testing.assert_array_equal(
                X_long_array,
                X_long_copy,
                "Input array with long text was modified during probability prediction",
            )
            np.testing.assert_array_equal(
                X_commas_array,
                X_commas_copy,
                "Input array with comma text was modified during probability prediction",
            )
            np.testing.assert_array_equal(
                X_spaces_array,
                X_spaces_copy,
                "Input array with special spaces was modified during probability prediction",
            )

            # Verify probability predictions are returned as expected
            np.testing.assert_array_equal(proba_normal, expected_probas)
            np.testing.assert_array_equal(proba_long, expected_probas)
            np.testing.assert_array_equal(proba_commas, expected_probas)
            np.testing.assert_array_equal(proba_spaces, expected_probas)

            # Verify that long text gives same probability predictions as normal text
            np.testing.assert_array_equal(proba_normal, proba_long)

            # Verify predict and predict_proba were each called 4 times
            self.assertEqual(mock_predict.call_count, 8)

    def test_predict_with_pandas_dataframe(self):
        """Test predictions with pandas DataFrame input, including text columns."""
        import pandas as pd

        # Skip initialization
        tabpfn = TabPFNClassifier()
        tabpfn.fitted_ = True
        tabpfn.classes_ = np.array([0, 1])

        # Create test data
        n_samples = 5

        # Create DataFrame with various column types
        df = pd.DataFrame(
            {
                "numeric1": np.random.randn(n_samples),
                "numeric2": np.random.randint(0, 100, n_samples),
                "text_normal": ["Sample text " * 10] * n_samples,
                "text_long": ["Very long " * 500] * n_samples,  # >2500 chars
                "text_commas": ["Text, with, commas, " * 100] * n_samples,
            }
        )

        # Create a copy for comparison
        df_copy = df.copy()

        # Mock predictions
        expected_predictions = np.random.randint(0, 2, n_samples)
        expected_probas = np.random.rand(n_samples, 2)
        expected_probas = expected_probas / expected_probas.sum(axis=1)[:, np.newaxis]

        with patch.object(InferenceClient, "predict") as mock_predict:
            # Test predict()
            mock_predict.return_value = PredictionResult(
                y_pred=expected_predictions, metadata={}
            )
            pred = tabpfn.predict(df)

            # Verify DataFrame wasn't modified
            pd.testing.assert_frame_equal(
                df, df_copy, "Input DataFrame was modified during prediction"
            )

            # Verify predictions are returned as expected
            np.testing.assert_array_equal(pred, expected_predictions)

            # Test predict_proba()
            mock_predict.return_value = PredictionResult(
                y_pred=expected_probas, metadata={}
            )
            proba = tabpfn.predict_proba(df)

            # Verify DataFrame wasn't modified during predict_proba
            pd.testing.assert_frame_equal(
                df,
                df_copy,
                "Input DataFrame was modified during probability prediction",
            )

            # Verify probability predictions are returned as expected
            np.testing.assert_array_equal(proba, expected_probas)

            # Verify predict and predict_proba were each called once
            self.assertEqual(mock_predict.call_count, 2)

            # Test that column order doesn't matter
            shuffled_columns = list(df.columns)
            np.random.shuffle(shuffled_columns)
            df_shuffled = df[shuffled_columns]
            df_shuffled_copy = df_shuffled.copy()

            # Test predict with shuffled columns
            mock_predict.return_value = PredictionResult(
                y_pred=expected_predictions, metadata={}
            )
            pred_shuffled = tabpfn.predict(df_shuffled)

            # Verify shuffled DataFrame wasn't modified
            pd.testing.assert_frame_equal(
                df_shuffled,
                df_shuffled_copy,
                "Shuffled DataFrame was modified during prediction",
            )

            # Verify predictions match regardless of column order
            np.testing.assert_array_equal(pred, pred_shuffled)

    def test_missing_values_in_y_raise_error(self):
        """Test that missing values in y raise a ValueError."""
        import pandas as pd

        X = np.random.rand(10, 5)

        # Test with None values
        y_none = np.array([0, 1, None, 1, 0, 1, None, 0, 1, 0])
        print("y_none", y_none)
        print("y_none.dtype", y_none.dtype)
        tabpfn = TabPFNClassifier()
        with self.assertRaises(ValueError) as cm:
            tabpfn.fit(X, y_none)
        self.assertIn("contains NaN.", str(cm.exception))

        # Test with np.nan values
        y_nan = np.array([0, 1, np.nan, 1, 0, 1, np.nan, 0, 1, 0], dtype=float)
        with self.assertRaises(ValueError) as cm:
            tabpfn.fit(X, y_nan)
        self.assertIn("contains NaN.", str(cm.exception))

        # Test with pd.NA values
        y_pd_na = pd.Series([0, 1, pd.NA, 1, 0, 1, pd.NA, 0, 1, 0])
        with self.assertRaises(ValueError) as cm:
            tabpfn.fit(X, y_pd_na)
        self.assertIn("contains NaN.", str(cm.exception))

        # Test with multiclass data containing missing values
        y_multiclass_na = pd.Series([0, 1, 2, pd.NA, 1, 0, 2, pd.NA, 1, 0])
        with self.assertRaises(ValueError) as cm:
            tabpfn.fit(X, y_multiclass_na)
        self.assertIn("contains NaN.", str(cm.exception))

    def test_too_many_classes_raise_error(self):
        """Test that having more than 10 classes raises a ValueError."""
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 11, 100)  # 11 classes (0-10)

        tabpfn = TabPFNClassifier()
        with self.assertRaises(ValueError) as cm:
            tabpfn.fit(X, y)
        self.assertIn("exceeds the maximal number of", str(cm.exception))

        # Test with string labels
        y_str = np.array([f"class_{i}" for i in range(11)])[y]
        with self.assertRaises(ValueError) as cm:
            tabpfn.fit(X, y_str)
        self.assertIn("exceeds the maximal number of", str(cm.exception))


class TestTabPFNModelSelection(unittest.TestCase):
    def setUp(self):
        # skip init
        config.Config.is_initialized = True
        config.Config.use_server = True

    def tearDown(self):
        # undo setUp
        config.reset()

    def test_list_available_models_returns_expected_models(self):
        expected_models = [
            "v2.5_default-2",
            "v2.5_default",
            "v2.5_large-features-L",
            "v2.5_large-features-XL",
            "v2.5_large-samples",
            "v2.5_real-large-features",
            "v2.5_real-large-samples-and-features",
            "v2.5_real",
            "v2.5_variant",
            "v2_default",
            "default",
            "gn2p4bpt",
            "llderlii",
            "od3j1g5m",
            "vutqq28w",
            "znskzxi4",
        ]
        self.assertEqual(TabPFNClassifier.list_available_models(), expected_models)

    def test_model_names_that_are_substrings_come_later(self):
        # Mitigation to ensure that model "parsing" in the tabpfn-time-series
        # package continues to work. Long-term we should fix that package as
        # that "parsing" is quite brittle.
        # https://github.com/PriorLabs/tabpfn-time-series/blob/71c22aed9d3f8ec280ffb753d0e87086be3cb7a4/tabpfn_time_series/worker/model_adapters/tabpfn_adapter.py#L18

        model_names = TabPFNClassifier.list_available_models()

        for i in range(len(model_names)):
            possible_substring = model_names[i]
            for j in range(i + 1, len(model_names)):
                model_name = model_names[j]
                self.assertNotIn(possible_substring, model_name)

    def test_validate_model_name_with_valid_model_passes(self):
        # Should not raise any exception
        TabPFNClassifier._validate_model_name("default")
        TabPFNClassifier._validate_model_name("gn2p4bpt")

    def test_validate_model_name_with_invalid_model_raises_error(self):
        with self.assertRaises(ValueError):
            TabPFNClassifier._validate_model_name("invalid_model")

    def test_model_name_to_path_returns_expected_path(self):
        # Test default model path. Server decides.
        expected_default_path = None
        self.assertEqual(
            TabPFNClassifier._model_name_to_path("classification", "default"),
            expected_default_path,
        )

        # Test specific model path
        expected_specific_path = "tabpfn-v2-classifier-gn2p4bpt.ckpt"
        self.assertEqual(
            TabPFNClassifier._model_name_to_path("classification", "gn2p4bpt"),
            expected_specific_path,
        )

        # Test specific v2.5 model path
        expected_specific_path = "tabpfn-v2.5-classifier-v2.5_default.ckpt"
        self.assertEqual(
            TabPFNClassifier._model_name_to_path("classification", "v2.5_default"),
            expected_specific_path,
        )

    def test_model_name_to_path_with_invalid_model_raises_error(self):
        with self.assertRaises(ValueError):
            TabPFNClassifier._model_name_to_path("classification", "invalid_model")

    def test_predict_proba_uses_correct_model_path(self):
        # Setup
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)

        tabpfn = TabPFNClassifier(model_path="gn2p4bpt")

        # Mock the inference client
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = PredictionResult(
                y_pred={"probas": np.random.rand(10, 2)}, metadata={}
            )

            with patch.object(InferenceClient, "fit") as mock_fit:
                mock_fit.return_value = "dummy_uid"

                # Fit and predict
                tabpfn.fit(X, y)
                tabpfn.predict_proba(X)

                # Verify the model path was correctly passed to predict
                predict_kwargs = mock_predict.call_args[1]
                expected_model_path = "tabpfn-v2-classifier-gn2p4bpt.ckpt"

                self.assertEqual(
                    predict_kwargs["tabpfn_config"]["model_path"], expected_model_path
                )

    @patch.object(InferenceClient, "fit", return_value="dummy_uid")
    @patch.object(
        InferenceClient,
        "predict",
        return_value=PredictionResult(
            y_pred={"probas": np.random.rand(10, 2)}, metadata={}
        ),
    )
    def test_paper_version_behavior(self, mock_predict, mock_fit):
        # this just test that it doesn't break,
        # but the actual behavior is easier to test
        # on the server side
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        test_X = np.random.rand(5, 5)

        # Test with paper_version=True
        tabpfn_true = TabPFNClassifier(paper_version=True)
        tabpfn_true.fit(X, y)
        y_pred_true = tabpfn_true.predict(test_X)
        self.assertIsNotNone(y_pred_true)

        # Test with paper_version=False
        tabpfn_false = TabPFNClassifier(paper_version=False)
        tabpfn_false.fit(X, y)
        y_pred_false = tabpfn_false.predict(test_X)
        self.assertIsNotNone(y_pred_false)

    @patch.object(
        InferenceClient,
        "predict",
        return_value=PredictionResult(y_pred=np.random.rand(20, 2), metadata={}),
    )
    @patch.object(InferenceClient, "fit", return_value="dummy_uid")
    def test_cross_validation(self, mock_fit, mock_predict):
        """Test that TabPFNClassifier works with sklearn's cross_val_score,
        even when predict_proba is required."""

        # Create synthetic dataset
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)

        # Initialize classifier
        clf = TabPFNClassifier()

        # Compute cross validation scores
        scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")

        # Basic validation checks
        self.assertEqual(len(scores), 5)  # 5-fold CV should return 5 scores
        self.assertTrue(
            all(0 <= score <= 1 for score in scores)
        )  # ROC AUC scores should be between 0 and 1

        # Verify fit and predict were called appropriate number of times
        # For 5-fold CV, fit should be called 5 times (once per fold)
        self.assertEqual(mock_fit.call_count, 5)

        # predict_proba should be called 5 times (once per fold)
        self.assertEqual(mock_predict.call_count, 5)
