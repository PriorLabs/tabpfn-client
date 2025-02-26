import unittest
from unittest.mock import patch
import numpy as np

from tabpfn_client.mock_prediction import (
    mock_mode,
    check_api_credits,
    get_mock_time,
    get_mock_cost,
    estimate_duration,
    is_mock_mode,
)
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor


class TestMockPrediction(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.X_train = np.random.rand(100, 10)
        self.y_train = np.random.randint(0, 2, 100)
        self.X_test = np.random.rand(20, 10)
        self.config = {"n_estimators": 4}

    def test_mock_mode_behavior(self):
        """Test that mock mode properly tracks time and cost"""
        self.assertFalse(is_mock_mode())

        # Patch get_access_token and init to prevent authentication prompts
        with (
            patch("tabpfn_client.config.get_access_token") as mock_token,
            patch("tabpfn_client.get_access_token") as mock_token_direct,
            patch("tabpfn_client.config.init") as mock_init,
            patch("tabpfn_client.init") as mock_init_direct,
            patch("tabpfn_client.config.Config.is_initialized", True),
        ):
            mock_token.return_value = "fake_token"
            mock_token_direct.return_value = "fake_token"
            mock_init.return_value = None
            mock_init_direct.return_value = None

            with mock_mode():
                self.assertTrue(is_mock_mode())
                initial_time = get_mock_time()
                initial_cost = get_mock_cost()

                # Use TabPFNClassifier for prediction
                clf = TabPFNClassifier(n_estimators=4)
                clf.fit(self.X_train, self.y_train)
                clf.predict(self.X_test)

                # Verify time increased by the estimated duration
                expected_duration = estimate_duration(
                    self.X_train.shape[0] + self.X_test.shape[0],
                    self.X_test.shape[1],
                    "classification",
                    self.config,
                )
                self.assertAlmostEqual(
                    get_mock_time() - initial_time, expected_duration
                )

                # Verify cost was tracked
                expected_cost = (
                    (self.X_train.shape[0] + self.X_test.shape[0])
                    * self.X_test.shape[1]
                    * self.config["n_estimators"]
                )
                self.assertEqual(get_mock_cost() - initial_cost, expected_cost)

        # Verify mock mode is disabled after context
        self.assertFalse(is_mock_mode())

    def test_mock_predict_output_consistency(self):
        """Test that mock predictions maintain consistent shapes and ranges"""
        # Patch get_access_token and init to prevent authentication prompts
        with (
            patch("tabpfn_client.config.get_access_token") as mock_token,
            patch("tabpfn_client.get_access_token") as mock_token_direct,
            patch("tabpfn_client.config.init") as mock_init,
            patch("tabpfn_client.init") as mock_init_direct,
            patch("tabpfn_client.config.Config.is_initialized", True),
        ):
            mock_token.return_value = "fake_token"
            mock_token_direct.return_value = "fake_token"
            mock_init.return_value = None
            mock_init_direct.return_value = None

            with mock_mode():
                # Test classification probabilities
                clf = TabPFNClassifier(n_estimators=4)
                clf.fit(self.X_train, self.y_train)
                result = clf.predict_proba(self.X_test)

                self.assertEqual(
                    result.shape, (self.X_test.shape[0], len(np.unique(self.y_train)))
                )
                self.assertTrue(np.all(result >= 0) and np.all(result <= 1))
                np.testing.assert_array_almost_equal(
                    result.sum(axis=1), np.ones(self.X_test.shape[0])
                )

                # Test regression with full output
                y_train_reg = np.random.rand(100)
                reg = TabPFNRegressor(n_estimators=8)
                reg.fit(self.X_train, y_train_reg)
                result = reg.predict(self.X_test, output_type="full")

                # Check all required fields are present with correct shapes
                self.assertTrue(isinstance(result, dict))
                self.assertEqual(result["logits"].shape, (self.X_test.shape[0], 5000))
                self.assertEqual(result["mean"].shape, (self.X_test.shape[0],))
                self.assertEqual(result["median"].shape, (self.X_test.shape[0],))
                self.assertEqual(result["mode"].shape, (self.X_test.shape[0],))
                self.assertEqual(result["quantiles"].shape, (3, self.X_test.shape[0]))
                self.assertEqual(result["borders"].shape, (5001,))
                self.assertEqual(result["ei"].shape, (self.X_test.shape[0],))
                self.assertEqual(result["pi"].shape, (self.X_test.shape[0],))

    def test_check_api_credits_decorator(self):
        """Test that the credit checker properly validates available credits"""

        @check_api_credits
        def dummy_function():
            # Use TabPFNClassifier to consume credits
            clf = TabPFNClassifier(n_estimators=4)
            clf.fit(self.X_train, self.y_train)
            clf.predict(self.X_test)
            return "success"

        # Mock both get_access_token, init, and get_api_usage
        with (
            patch("tabpfn_client.config.get_access_token") as mock_token,
            patch("tabpfn_client.get_access_token") as mock_token_direct,
            patch("tabpfn_client.config.init") as mock_init,
            patch("tabpfn_client.init") as mock_init_direct,
            patch("tabpfn_client.config.Config.is_initialized", True),
            patch(
                "tabpfn_client.mock_prediction.ServiceClient.get_api_usage"
            ) as mock_usage,
        ):
            mock_token.return_value = "fake_token"
            mock_token_direct.return_value = "fake_token"
            mock_init.return_value = None
            mock_init_direct.return_value = None

            # Test when we have enough credits
            mock_usage.return_value = {"usage_limit": 10000, "current_usage": 0}
            with mock_mode():
                result = dummy_function()
                self.assertEqual(result, "success")

            # Test when we don't have enough credits
            mock_usage.return_value = {"usage_limit": 1, "current_usage": 1}
            with self.assertRaises(RuntimeError):
                with mock_mode():
                    dummy_function()

            # Test with unlimited credits
            mock_usage.return_value = {"usage_limit": -1, "current_usage": 1000}
            with mock_mode():
                result = dummy_function()
                self.assertEqual(result, "success")


if __name__ == "__main__":
    unittest.main()
