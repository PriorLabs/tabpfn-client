"""
Integration tests for TabPFN v2/v2.5 estimators against the live server.

Run with:
    pytest tests/quick_test_v2.py -v
    pytest tests/quick_test_v2.py -v -k classifier
    pytest tests/quick_test_v2.py -v -k regressor
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split

from tabpfn_client.constants import ModelVersion
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_webbrowser():
    """Prevent browser login popup during tests."""
    with patch("webbrowser.open", return_value=False):
        yield


@pytest.fixture()
def clf_data():
    """Breast cancer classification dataset, small split."""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    return X_train[:99], X_test, y_train[:99], y_test


@pytest.fixture()
def reg_data():
    """Diabetes regression dataset, small split."""
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    return X_train[:99], X_test, y_train[:99], y_test


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fit_and_predict_classifier(clf, X_train, y_train, X_test):
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probas = clf.predict_proba(X_test)
    return preds, probas


def _fit_and_predict_regressor(reg, X_train, y_train, X_test, **predict_kwargs):
    reg.fit(X_train, y_train)
    return reg.predict(X_test, **predict_kwargs)


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------


class TestClassifierVersions:
    """Test classification with different model versions."""

    @pytest.mark.parametrize("version", [ModelVersion.V2, ModelVersion.V2_5])
    def test_predict_and_predict_proba(self, clf_data, version):
        X_train, X_test, y_train, _ = clf_data
        clf = TabPFNClassifier.create_default_for_version(version, n_estimators=3)
        preds, probas = _fit_and_predict_classifier(clf, X_train, y_train, X_test)

        assert preds.shape == (len(X_test),)
        assert probas.shape[0] == len(X_test)
        assert probas.shape[1] == 2  # binary classification
        # probabilities sum to 1
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)
        assert clf.last_meta, "last_meta should be populated after predict"


class TestClassifierConfig:
    """Test classification with different estimator configurations."""

    @pytest.mark.parametrize("n_estimators", [1, 4])
    def test_n_estimators(self, clf_data, n_estimators):
        X_train, X_test, y_train, _ = clf_data
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5, n_estimators=n_estimators
        )
        preds, probas = _fit_and_predict_classifier(clf, X_train, y_train, X_test)
        assert preds.shape == (len(X_test),)
        assert probas.shape[0] == len(X_test)

    @pytest.mark.parametrize("softmax_temperature", [0.5, 0.9, 1.0])
    def test_softmax_temperature(self, clf_data, softmax_temperature):
        X_train, X_test, y_train, _ = clf_data
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5,
            n_estimators=3,
            softmax_temperature=softmax_temperature,
        )
        preds, probas = _fit_and_predict_classifier(clf, X_train, y_train, X_test)
        assert preds.shape == (len(X_test),)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    def test_balance_probabilities(self, clf_data):
        X_train, X_test, y_train, _ = clf_data
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5,
            n_estimators=3,
            balance_probabilities=True,
        )
        preds, probas = _fit_and_predict_classifier(clf, X_train, y_train, X_test)
        assert preds.shape == (len(X_test),)
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-5)

    def test_average_before_softmax(self, clf_data):
        X_train, X_test, y_train, _ = clf_data
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5,
            n_estimators=3,
            average_before_softmax=True,
        )
        preds, probas = _fit_and_predict_classifier(clf, X_train, y_train, X_test)
        assert preds.shape == (len(X_test),)

    def test_predict_twice_without_refit(self, clf_data):
        """Calling predict multiple times should not require refitting."""
        X_train, X_test, y_train, _ = clf_data
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        clf.fit(X_train, y_train)
        # NOTE: multiple predicts do not require refitting, but one predict always
        # requires a fit call even if it was already fitted (no-op in that case).
        preds1 = clf.predict(X_test)
        preds2 = clf.predict(X_test)
        np.testing.assert_array_equal(preds1, preds2)


class TestClassifierInputFormats:
    """Test classification with different input data formats."""

    def test_pandas_input(self, clf_data):
        X_train, X_test, y_train, _ = clf_data
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        preds, probas = _fit_and_predict_classifier(clf, X_train_df, y_train, X_test_df)
        assert preds.shape == (len(X_test),)

    def test_small_train_set(self, clf_data):
        """Fit with very few training samples."""
        X_train, X_test, y_train, _ = clf_data
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        preds, probas = _fit_and_predict_classifier(
            clf, X_train[:10], y_train[:10], X_test
        )
        assert preds.shape == (len(X_test),)

    def test_single_test_sample(self, clf_data):
        """Predict on a single test sample."""
        X_train, X_test, y_train, _ = clf_data
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test[:1])
        probas = clf.predict_proba(X_test[:1])
        assert preds.shape == (1,)
        assert probas.shape == (1, 2)

    def test_few_features(self):
        """Classification with a small number of features."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 2)
        y_train = (X_train[:, 0] > 0).astype(int)
        X_test = rng.randn(10, 2)
        clf = TabPFNClassifier.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        preds, probas = _fit_and_predict_classifier(clf, X_train, y_train, X_test)
        assert preds.shape == (10,)


# ---------------------------------------------------------------------------
# Regressor tests
# ---------------------------------------------------------------------------


class TestRegressorVersions:
    """Test regression with different model versions."""

    @pytest.mark.parametrize("version", [ModelVersion.V2, ModelVersion.V2_5])
    def test_predict_mean(self, reg_data, version):
        X_train, X_test, y_train, _ = reg_data
        reg = TabPFNRegressor.create_default_for_version(version, n_estimators=3)
        preds = _fit_and_predict_regressor(reg, X_train, y_train, X_test)
        assert preds.shape == (len(X_test),)
        assert reg.last_meta, "last_meta should be populated after predict"


class TestRegressorOutputTypes:
    """Test regression with different output_type values."""

    @pytest.mark.parametrize("output_type", ["mean", "median", "mode"])
    def test_scalar_output_types(self, reg_data, output_type):
        X_train, X_test, y_train, _ = reg_data
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        preds = _fit_and_predict_regressor(
            reg, X_train, y_train, X_test, output_type=output_type
        )
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X_test),)

    def test_output_type_quantiles_default(self, reg_data):
        X_train, X_test, y_train, _ = reg_data
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        preds = _fit_and_predict_regressor(
            reg, X_train, y_train, X_test, output_type="quantiles"
        )
        # default quantiles: [0.1, 0.2, ..., 0.9] -> 9 quantiles
        assert isinstance(preds, (list, np.ndarray))

    def test_output_type_quantiles_custom(self, reg_data):
        X_train, X_test, y_train, _ = reg_data
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        quantiles = [0.25, 0.5, 0.75]
        preds = _fit_and_predict_regressor(
            reg,
            X_train,
            y_train,
            X_test,
            output_type="quantiles",
            quantiles=quantiles,
        )
        assert isinstance(preds, (list, np.ndarray))

    def test_output_type_main(self, reg_data):
        X_train, X_test, y_train, _ = reg_data
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        preds = _fit_and_predict_regressor(
            reg, X_train, y_train, X_test, output_type="main"
        )
        assert isinstance(preds, dict)

    def test_output_type_full(self, reg_data):
        """Full output returns a dict with distribution details."""
        X_train, X_test, y_train, _ = reg_data
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        # Use fewer test samples to stay under the full-output row limit
        preds = _fit_and_predict_regressor(
            reg, X_train, y_train, X_test[:5], output_type="full"
        )
        assert isinstance(preds, dict)


class TestRegressorConfig:
    """Test regression with different estimator configurations."""

    @pytest.mark.parametrize("n_estimators", [1, 4])
    def test_n_estimators(self, reg_data, n_estimators):
        X_train, X_test, y_train, _ = reg_data
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5, n_estimators=n_estimators
        )
        preds = _fit_and_predict_regressor(reg, X_train, y_train, X_test)
        assert preds.shape == (len(X_test),)

    @pytest.mark.parametrize("softmax_temperature", [0.5, 0.9, 1.0])
    def test_softmax_temperature(self, reg_data, softmax_temperature):
        X_train, X_test, y_train, _ = reg_data
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5,
            n_estimators=3,
            softmax_temperature=softmax_temperature,
        )
        preds = _fit_and_predict_regressor(reg, X_train, y_train, X_test)
        assert preds.shape == (len(X_test),)

    def test_average_before_softmax(self, reg_data):
        X_train, X_test, y_train, _ = reg_data
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5,
            n_estimators=3,
            average_before_softmax=True,
        )
        preds = _fit_and_predict_regressor(reg, X_train, y_train, X_test)
        assert preds.shape == (len(X_test),)

    def test_predict_twice_without_refit(self, reg_data):
        """Calling predict multiple times should not require refitting."""
        X_train, X_test, y_train, _ = reg_data
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        reg.fit(X_train, y_train)
        preds1 = reg.predict(X_test)
        preds2 = reg.predict(X_test)
        np.testing.assert_array_equal(preds1, preds2)


class TestRegressorInputFormats:
    """Test regression with different input data formats."""

    def test_pandas_input(self, reg_data):
        X_train, X_test, y_train, _ = reg_data
        X_train_df = pd.DataFrame(X_train)
        X_test_df = pd.DataFrame(X_test)
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        preds = _fit_and_predict_regressor(reg, X_train_df, y_train, X_test_df)
        assert preds.shape == (len(X_test),)

    def test_small_train_set(self, reg_data):
        """Fit with very few training samples."""
        X_train, X_test, y_train, _ = reg_data
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        preds = _fit_and_predict_regressor(reg, X_train[:10], y_train[:10], X_test)
        assert preds.shape == (len(X_test),)

    def test_single_test_sample(self, reg_data):
        """Predict on a single test sample."""
        X_train, X_test, y_train, _ = reg_data
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test[:1])
        assert preds.shape == (1,)

    def test_few_features(self):
        """Regression with a small number of features."""
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 2)
        y_train = X_train[:, 0] + 0.5 * rng.randn(50)
        X_test = rng.randn(10, 2)
        reg = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2_5, n_estimators=3
        )
        preds = _fit_and_predict_regressor(reg, X_train, y_train, X_test)
        assert preds.shape == (10,)
