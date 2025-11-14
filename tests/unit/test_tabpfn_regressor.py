import unittest
from unittest.mock import patch
import sys
import types
import builtins

import shutil
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.exceptions import NotFittedError

from tabpfn_client import init, reset
from tabpfn_client.estimator import TabPFNRegressor
from tabpfn_client.service_wrapper import UserAuthenticationClient, InferenceClient
from tests.mock_tabpfn_server import with_mock_server
from tabpfn_client.constants import CACHE_DIR
from tabpfn_client import config
import json
from tabpfn_client.client import PredictionResult


class TestTabPFNRegressorInit(unittest.TestCase):
    dummy_token = "dummy_token"

    def setUp(self):
        # set up dummy data
        reset()
        X, y = load_diabetes(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33
        )

    def tearDown(self):
        # remove cache dir
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    @patch("tabpfn_client.browser_auth.webbrowser.open", return_value=False)
    @patch("tabpfn_client.prompt_agent.PromptAgent.prompt_and_set_token")
    @with_mock_server()
    def test_init_remote_regressor(
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

        mock_predict_responses = {
            "mean": [100, 200, 300],
            "median": [110, 210, 310],
            "mode": [120, 220, 320],
        }
        for metric, response in mock_predict_responses.items():
            predict_route = mock_server.router.post(mock_server.endpoints.predict.path)
            predict_route.respond(
                200,
                content=f"data: {json.dumps({'event': 'result', 'data': {'regression': response, 'test_set_uid': '6'}})}\n\n",
                headers={"Content-Type": "text/event-stream"},
            )

            init(use_server=True)
            self.assertTrue(mock_prompt_and_set_token.called)

        tabpfn = TabPFNRegressor(n_estimators=10)
        self.assertRaises(NotFittedError, tabpfn.predict, self.X_test)
        tabpfn.fit(self.X_train, self.y_train)
        self.assertTrue(mock_prompt_and_set_token.called)

        y_pred = tabpfn.predict(self.X_test, output_type=metric)
        self.assertTrue(np.all(np.array(response) == y_pred))
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
    def test_reset_on_remote_regressor(self, mock_server):
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
    @with_mock_server()
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

        mock_predict_response = {
            "mean": [100, 200, 300],
            "median": [110, 210, 310],
            "mode": [120, 220, 320],
        }
        predict_route = mock_server.router.post(mock_server.endpoints.predict.path)
        predict_route.respond(
            200,
            content=f"data: {json.dumps({'event': 'result', 'data': {'regression': mock_predict_response, 'test_set_uid': '6'}})}\n\n",
            headers={"Content-Type": "text/event-stream"},
        )

        init(use_server=True)

        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        test_X = np.random.rand(5, 5)

        # Initialize with paper_version=True
        tabpfn_true = TabPFNRegressor(paper_version=True)

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
        tabpfn_false = TabPFNRegressor(paper_version=False)

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


class TestTabPFNRegressorInference(unittest.TestCase):
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
        y = np.random.rand(11)
        tabpfn = TabPFNRegressor()

        with self.assertRaises(ValueError):
            tabpfn.fit(X, y)

    def test_data_size_check_on_train_with_oversized_data_raise_error(self):
        X = np.random.randn(50_001, 2001)
        y = np.random.randn(50_001)

        tabpfn = TabPFNRegressor()

        # test oversized columns
        with self.assertRaises(ValueError):
            tabpfn.fit(X[:10], y[:10])

        # test oversized rows
        with self.assertRaises(ValueError):
            tabpfn.fit(X[:, :10], y)

    def test_data_size_check_on_predict_with_oversized_data_raise_error(self):
        test_X = np.random.randn(50_001, 5)
        tabpfn = TabPFNRegressor()

        # skip fitting
        tabpfn.fitted_ = True

        # test oversized rows
        with self.assertRaises(ValueError):
            tabpfn.predict(test_X)

    def test_data_check_on_predict_with_valid_data_pass(self):
        test_X = np.random.randn(10, 5)
        tabpfn = TabPFNRegressor()

        # skip fitting
        tabpfn.fitted_ = True

        # mock prediction
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = PredictionResult(
                y_pred={"mean": np.random.randn(10)}, metadata={}
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
            "paper_version",
            "thinking",
            "thinking_params",
        }

        # Create regressor with various parameters
        regressor = TabPFNRegressor(
            n_estimators=8,
            softmax_temperature=0.9,
            paper_version=True,
            random_state=42,
        )

        # Skip fitting
        regressor.fitted_ = True
        regressor.last_train_set_uid = "dummy_uid"

        test_X = np.random.randn(10, 5)

        # Mock predict and capture config
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = PredictionResult(
                y_pred={"mean": np.random.randn(10)}, metadata={}
            )
            regressor.predict(test_X)

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
        """Test that predict_params contains correct output_type and quantiles."""
        regressor = TabPFNRegressor()
        regressor.fitted_ = True  # Skip fitting
        test_X = np.random.randn(10, 5)

        # Test default predict() sets output_type to "mean"
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = PredictionResult(
                y_pred={"mean": np.random.randn(10)}, metadata={}
            )
            regressor.predict(test_X)

            predict_params = mock_predict.call_args[1]["predict_params"]
            self.assertEqual(predict_params, {"output_type": "mean", "quantiles": None})

        # Test predict() with quantiles
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = PredictionResult(
                y_pred={"quantiles": np.random.randn(10, 3)}, metadata={}
            )
            quantiles = [0.1, 0.5, 0.9]
            regressor.predict(test_X, output_type="quantiles", quantiles=quantiles)

            predict_params = mock_predict.call_args[1]["predict_params"]
            self.assertEqual(
                predict_params, {"output_type": "quantiles", "quantiles": quantiles}
            )

    def test_predict_full_adds_criterion_with_optional_dependencies(self):
        regressor = TabPFNRegressor()
        regressor.fitted_ = True
        regressor.last_train_set_uid = "dummy_uid"
        regressor.last_train_X = np.random.randn(5, 2)
        regressor.last_train_y = np.random.randn(5)

        test_X = np.random.randn(3, 2)
        dummy_output = {"borders": [0.0, 1.0], "mean": np.random.randn(3)}

        class DummyFullSupportBarDistribution:
            def __init__(self, borders):
                self.borders = borders

        module_tabpfn = types.ModuleType("tabpfn")
        module_regressor = types.ModuleType("tabpfn.regressor")
        module_regressor.FullSupportBarDistribution = DummyFullSupportBarDistribution
        module_tabpfn.regressor = module_regressor

        module_torch = types.ModuleType("torch")
        module_torch.tensor = lambda borders: borders

        with patch.dict(
            sys.modules,
            {
                "tabpfn": module_tabpfn,
                "tabpfn.regressor": module_regressor,
                "torch": module_torch,
            },
        ):
            with patch.object(InferenceClient, "predict") as mock_predict:
                mock_predict.return_value = PredictionResult(
                    y_pred=dummy_output.copy(), metadata={}
                )
                output = regressor.predict(test_X, output_type="full")

        self.assertIn("criterion", output)
        self.assertIsInstance(output["criterion"], DummyFullSupportBarDistribution)
        self.assertEqual(output["criterion"].borders, dummy_output["borders"])

    def test_predict_full_missing_optional_dependencies_logs_warning(self):
        regressor = TabPFNRegressor()
        regressor.fitted_ = True
        regressor.last_train_set_uid = "dummy_uid"
        regressor.last_train_X = np.random.randn(5, 2)
        regressor.last_train_y = np.random.randn(5)

        test_X = np.random.randn(3, 2)
        dummy_output = {"borders": [0.0, 1.0], "mean": np.random.randn(3)}

        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = PredictionResult(
                y_pred=dummy_output.copy(), metadata={}
            )

            original_import = builtins.__import__

            def import_side_effect(name, *args, **kwargs):
                if name in {"tabpfn", "tabpfn.regressor", "torch"}:
                    raise ImportError(f"No module named {name}")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=import_side_effect):
                with self.assertLogs(
                    "tabpfn_client.estimator", level="WARNING"
                ) as captured_logs:
                    output = regressor.predict(test_X, output_type="full")

        self.assertNotIn("criterion", output)
        self.assertTrue(
            any("Optional dependencies" in message for message in captured_logs.output)
        )

    def test_predict_with_long_and_comma_text(self):
        """Test predictions with long text (>2500 chars) and text containing commas."""
        # Skip initialization
        tabpfn = TabPFNRegressor()
        tabpfn.fitted_ = True

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
        expected_predictions = np.random.randn(n_samples)
        with patch.object(InferenceClient, "predict") as mock_predict:
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

            # Verify predict was called the same way for all variations
            self.assertEqual(mock_predict.call_count, 4)

    def test_predict_with_pandas_dataframe(self):
        """Test predictions with pandas DataFrame input, including text columns."""
        import pandas as pd

        # Skip initialization
        tabpfn = TabPFNRegressor()
        tabpfn.fitted_ = True

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
        expected_predictions = np.random.randn(n_samples)

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

            # Verify predict was called twice
            self.assertEqual(mock_predict.call_count, 2)

    def test_missing_values_in_y_raise_error(self):
        """Test that missing values in y raise a ValueError."""
        import pandas as pd

        X = np.random.rand(10, 5)

        # Test with None values
        y_none = np.array([1.0, 2.0, None, 3.0, 4.0, 5.0, None, 6.0, 7.0, 8.0])
        tabpfn = TabPFNRegressor()
        with self.assertRaises(ValueError) as cm:
            tabpfn.fit(X, y_none)
        self.assertIn("contains NaN.", str(cm.exception))

        # Test with np.nan values
        y_nan = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0, np.nan, 6.0, 7.0, 8.0])
        with self.assertRaises(ValueError) as cm:
            tabpfn.fit(X, y_nan)
        self.assertIn("contains NaN.", str(cm.exception))

        # Test with pd.NA values
        y_pd_na = pd.Series([1.0, 2.0, pd.NA, 3.0, 4.0, 5.0, pd.NA, 6.0, 7.0, 8.0])
        with self.assertRaises(ValueError) as cm:
            tabpfn.fit(X, y_pd_na)
        self.assertIn("contains NaN.", str(cm.exception))

    @patch.object(
        InferenceClient,
        "predict",
        return_value=PredictionResult(y_pred=np.random.rand(20), metadata={}),
    )
    @patch.object(InferenceClient, "fit", return_value="dummy_uid")
    def test_cross_validation(self, mock_fit, mock_predict):
        """Test that TabPFNRegressor works with sklearn's cross_val_score."""

        # Create synthetic dataset
        X = np.random.rand(100, 10)
        y = np.random.rand(100)

        # Initialize regressor
        reg = TabPFNRegressor()

        # Compute cross validation scores using R² score
        scores = cross_val_score(reg, X, y, cv=5, scoring="r2")

        # Basic validation checks
        self.assertEqual(len(scores), 5)  # 5-fold CV should return 5 scores
        # R² scores are typically <= 1, but can be negative for poor models
        self.assertTrue(all(score <= 1 for score in scores))

        # Verify fit and predict were called appropriate number of times
        # For 5-fold CV, fit should be called 5 times (once per fold)
        self.assertEqual(mock_fit.call_count, 5)

        # predict should be called 5 times (once per fold)
        self.assertEqual(mock_predict.call_count, 5)


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
            "v2.5_default",
            "v2.5_low-skew",
            "v2.5_quantiles",
            "v2.5_real-variant",
            "v2.5_real",
            "v2.5_small-samples",
            "v2.5_variant",
            "v2_default",
            "default",
            "2noar4o2",
            "5wof9ojf",
            "09gpqh39",
            "wyl4o83o",
        ]
        self.assertEqual(TabPFNRegressor.list_available_models(), expected_models)

    def test_model_names_that_are_substrings_come_later(self):
        # Mitigation to ensure that model "parsing" in the tabpfn-time-series
        # package continues to work. Long-term we should fix that package as
        # that "parsing" is quite brittle.
        # https://github.com/PriorLabs/tabpfn-time-series/blob/71c22aed9d3f8ec280ffb753d0e87086be3cb7a4/tabpfn_time_series/worker/model_adapters/tabpfn_adapter.py#L18

        model_names = TabPFNRegressor.list_available_models()

        for i in range(len(model_names)):
            possible_substring = model_names[i]
            for j in range(i + 1, len(model_names)):
                model_name = model_names[j]
                self.assertNotIn(possible_substring, model_name)

    def test_validate_model_name_with_valid_model_passes(self):
        # Should not raise any exception
        TabPFNRegressor._validate_model_name("default")
        TabPFNRegressor._validate_model_name("2noar4o2")

    def test_validate_model_name_with_invalid_model_raises_error(self):
        with self.assertRaises(ValueError):
            TabPFNRegressor._validate_model_name("invalid_model")

    def test_model_name_to_path_returns_expected_path(self):
        # Test default model path. Server decides.
        expected_default_path = None
        self.assertEqual(
            TabPFNRegressor._model_name_to_path("regression", "default"),
            expected_default_path,
        )

        # Test specific model path
        expected_specific_path = "tabpfn-v2-regressor-2noar4o2.ckpt"
        self.assertEqual(
            TabPFNRegressor._model_name_to_path("regression", "2noar4o2"),
            expected_specific_path,
        )

        # Test specific v2.5 model path
        expected_specific_path = "tabpfn-v2.5-regressor-v2.5_default.ckpt"
        self.assertEqual(
            TabPFNRegressor._model_name_to_path("regression", "v2.5_default"),
            expected_specific_path,
        )

    def test_model_name_to_path_with_invalid_model_raises_error(self):
        with self.assertRaises(ValueError):
            TabPFNRegressor._model_name_to_path("regression", "invalid_model")

    def test_predict_uses_correct_model_path(self):
        # Setup
        X = np.random.rand(10, 5)
        y = np.random.rand(10)

        tabpfn = TabPFNRegressor(model_path="2noar4o2")

        # Mock the inference client
        with patch.object(InferenceClient, "predict") as mock_predict:
            mock_predict.return_value = PredictionResult(
                y_pred={"mean": np.random.rand(10)}, metadata={}
            )
            with patch.object(InferenceClient, "fit") as mock_fit:
                mock_fit.return_value = "dummy_uid"

                # Fit and predict
                tabpfn.fit(X, y)
                tabpfn.predict(X)

                # Verify the model path was correctly passed to predict
                predict_kwargs = mock_predict.call_args[1]
                expected_model_path = "tabpfn-v2-regressor-2noar4o2.ckpt"

                self.assertEqual(
                    predict_kwargs["tabpfn_config"]["model_path"], expected_model_path
                )

    @patch.object(InferenceClient, "fit", return_value="dummy_uid")
    @patch.object(
        InferenceClient,
        "predict",
        return_value=PredictionResult(y_pred={"mean": np.random.rand(10)}, metadata={}),
    )
    def test_paper_version_behavior(self, mock_predict, mock_fit):
        # this just tests that it doesn't break,
        # but the actual behavior is easier to test
        # on the server side
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        test_X = np.random.rand(5, 5)

        # Test with paper_version=True
        tabpfn_true = TabPFNRegressor(paper_version=True)
        tabpfn_true.fit(X, y)
        y_pred_true = tabpfn_true.predict(test_X)
        self.assertIsNotNone(y_pred_true)

        # Test with paper_version=False
        tabpfn_false = TabPFNRegressor(paper_version=False)
        tabpfn_false.fit(X, y)
        y_pred_false = tabpfn_false.predict(test_X)
        self.assertIsNotNone(y_pred_false)
