"""Unit tests for the KV-cache client surface:

- `fit_mode` reaching the server-side `tabpfn_config` on the wire,
- `model_id_` (the fitted-train-set id, written by `fit()`/`load_model()`)
  as the single fitted-state slot that makes an estimator predictable,
- `save_model()` / `load_model()` round-trips,
- the refit-fallback guard for load-by-id estimators.

These stay hermetic by patching `InferenceClient` / `ServiceClient` rather
than talking to a mock server — the concern here is client-side wiring, not
the HTTP layer.
"""

import unittest
from unittest.mock import patch
from uuid import UUID

import numpy as np

from tabpfn_client.api_models import FitMode
from tabpfn_client.client import NeedsRefittingError, PredictionResult, ServiceClient
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.service_wrapper import InferenceClient

_MODEL_ID = "00000000-0000-0000-0000-000000000abc"


def _regressor_prediction() -> PredictionResult:
    return PredictionResult(y_pred={"mean": np.zeros(4)}, metadata={})


class TestFitMode(unittest.TestCase):
    def test_default_omits_fit_mode_on_wire(self):
        # Default is None, so nothing is sent and the server applies its own
        # default (fit_preprocessors) — matching the SDK's "None means unset"
        # convention for server-backed config fields.
        for Est in (TabPFNClassifier, TabPFNRegressor):
            cfg = Est()._get_tabpfn_config()
            dumped = cfg.model_dump(mode="json", exclude_none=True)
            self.assertNotIn("fit_mode", dumped, Est.__name__)

    def test_fit_with_cache_reaches_wire_config(self):
        for Est in (TabPFNClassifier, TabPFNRegressor):
            cfg = Est(fit_mode=FitMode.FIT_WITH_CACHE)._get_tabpfn_config()
            dumped = cfg.model_dump(mode="json", exclude_none=True)
            self.assertEqual(dumped.get("fit_mode"), "fit_with_cache", Est.__name__)

    def test_fit_mode_forwarded_on_predict_task_config(self):
        reg = TabPFNRegressor(fit_mode=FitMode.FIT_WITH_CACHE)
        reg.model_id_ = UUID(_MODEL_ID)
        with patch("tabpfn_client.estimator.init"):
            with patch.object(ServiceClient, "get_model_limits", return_value=None):
                with patch.object(
                    InferenceClient, "predict", return_value=_regressor_prediction()
                ) as mock_predict:
                    reg.predict(np.random.randn(4, 3))
        task_config = mock_predict.call_args[1]["task_config"]
        dumped = task_config.tabpfn_config.model_dump(mode="json", exclude_none=True)
        self.assertEqual(dumped.get("fit_mode"), "fit_with_cache")


class TestModelIdFittedState(unittest.TestCase):
    def test_model_id_marks_estimator_fitted(self):
        for Est in (TabPFNClassifier, TabPFNRegressor):
            est = Est()
            self.assertFalse(est.__sklearn_is_fitted__(), Est.__name__)
            est.model_id_ = UUID(_MODEL_ID)
            self.assertTrue(est.__sklearn_is_fitted__(), Est.__name__)

    def test_predict_uses_model_id_without_fitting(self):
        reg = TabPFNRegressor()
        reg.model_id_ = UUID(_MODEL_ID)
        with patch("tabpfn_client.estimator.init"):
            with patch.object(ServiceClient, "get_model_limits", return_value=None):
                with patch.object(InferenceClient, "fit") as mock_fit:
                    with patch.object(
                        InferenceClient, "predict", return_value=_regressor_prediction()
                    ) as mock_predict:
                        reg.predict(np.random.randn(4, 3))
        mock_fit.assert_not_called()
        self.assertEqual(
            mock_predict.call_args[1]["fitted_train_set_id"], UUID(_MODEL_ID)
        )


class TestSaveLoadModel(unittest.TestCase):
    def test_regressor_round_trip_dict(self):
        reg = TabPFNRegressor(n_estimators=4)
        reg.model_id_ = UUID(_MODEL_ID)
        handle = reg.save_model()
        self.assertEqual(handle["task"], "regression")
        self.assertEqual(handle["model_id"], _MODEL_ID)
        # `model_id_` is fitted state, not a constructor param, so it must not
        # leak into the params blob.
        self.assertNotIn("model_id", handle["params"])
        self.assertNotIn("client_options", handle["params"])

        loaded = TabPFNRegressor.load_model(handle)
        self.assertTrue(loaded.__sklearn_is_fitted__())
        self.assertEqual(loaded.model_id_, UUID(_MODEL_ID))
        self.assertEqual(loaded.get_params()["n_estimators"], 4)

    def test_classifier_round_trip_restores_classes(self):
        clf = TabPFNClassifier()
        clf.model_id_ = UUID(_MODEL_ID)
        clf.classes_ = np.array([0, 1, 2])
        loaded = TabPFNClassifier.load_model(clf.save_model())
        self.assertTrue(loaded.__sklearn_is_fitted__())
        self.assertEqual(loaded.model_id_, UUID(_MODEL_ID))
        np.testing.assert_array_equal(loaded.classes_, np.array([0, 1, 2]))

    def test_round_trip_via_file(self):
        import tempfile
        from pathlib import Path

        reg = TabPFNRegressor()
        reg.model_id_ = UUID(_MODEL_ID)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.json"
            returned = reg.save_model(path)
            self.assertEqual(Path(returned), path)
            self.assertTrue(path.exists())
            loaded = TabPFNRegressor.load_model(path)
        self.assertEqual(loaded.model_id_, UUID(_MODEL_ID))

    def test_load_model_task_mismatch_raises(self):
        clf = TabPFNClassifier()
        clf.model_id_ = UUID(_MODEL_ID)
        clf.classes_ = np.array([0, 1])
        with self.assertRaises(ValueError):
            TabPFNRegressor.load_model(clf.save_model())

    def test_save_model_unfitted_raises(self):
        from sklearn.exceptions import NotFittedError

        with self.assertRaises(NotFittedError):
            TabPFNRegressor().save_model()

    def test_unsupported_handle_version_raises(self):
        reg = TabPFNRegressor()
        reg.model_id_ = UUID(_MODEL_ID)
        handle = reg.save_model()
        handle["format_version"] = 999
        with self.assertRaises(ValueError):
            TabPFNRegressor.load_model(handle)


class TestRefitGuard(unittest.TestCase):
    def test_load_by_id_cannot_auto_refit(self):
        reg = TabPFNRegressor()
        reg.model_id_ = UUID(_MODEL_ID)  # no in-memory train data
        with patch("tabpfn_client.estimator.init"):
            with patch.object(ServiceClient, "get_model_limits", return_value=None):
                with patch.object(InferenceClient, "fit") as mock_fit:
                    with patch.object(
                        InferenceClient, "predict", side_effect=NeedsRefittingError()
                    ):
                        with self.assertRaises(RuntimeError) as ctx:
                            reg.predict(np.random.randn(4, 3))
        self.assertIn("no in-memory", str(ctx.exception).lower())
        mock_fit.assert_not_called()


class TestSklearnCompatibility(unittest.TestCase):
    def test_clone_drops_fitted_state(self):
        # `model_id_` is fitted state, not a constructor param, so `clone`
        # must NOT carry it over: clones start unfitted and CV folds refit
        # cleanly instead of inheriting a pointer to remote fitted state.
        from sklearn.base import clone

        from typing import Any, cast as _cast

        for Est in (TabPFNClassifier, TabPFNRegressor):
            original = Est(n_estimators=4)
            original.model_id_ = UUID(_MODEL_ID)
            # `clone`'s overloaded return over a `type[C] | type[R]` union
            # confuses pyright; runtime type is Est, but we cast to Any to
            # avoid re-typing at every call site below.
            cloned = _cast(Any, clone(original))
            self.assertFalse(cloned.__sklearn_is_fitted__(), Est.__name__)
            self.assertNotIn("model_id", cloned.get_params(), Est.__name__)
            self.assertEqual(cloned.get_params()["n_estimators"], 4, Est.__name__)

    def test_unfitted_estimator_raises_not_fitted(self):
        from sklearn.exceptions import NotFittedError
        from sklearn.utils.validation import check_is_fitted

        for Est in (TabPFNClassifier, TabPFNRegressor):
            est = Est()
            self.assertFalse(est.__sklearn_is_fitted__(), Est.__name__)
            with self.assertRaises(NotFittedError):
                check_is_fitted(est)

    def test_predict_triggers_init_from_cold_state(self):
        # A fresh process that only calls `load_model(...).predict(...)` never
        # touches `fit()`, so `_predict` itself must call `init()` to authorize
        # the HTTP client — otherwise the main cross-process reuse path 401s.
        reg = TabPFNRegressor()
        reg.model_id_ = UUID(_MODEL_ID)
        with patch("tabpfn_client.estimator.init") as mock_init:
            with patch.object(ServiceClient, "get_model_limits", return_value=None):
                with patch.object(
                    InferenceClient, "predict", return_value=_regressor_prediction()
                ):
                    reg.predict(np.random.randn(4, 3))
        mock_init.assert_called()


if __name__ == "__main__":
    unittest.main()
