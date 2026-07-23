"""Unit tests for the KV-cache client surface:

- `fit_mode` reaching the server-side `tabpfn_config` on the wire,
- the `model_id` constructor param (Option 1) making an estimator "fitted"
  and predictable without a local `fit()`,
- `save_model()` / `load_model()` round-trips (Option 3),
- the refit-fallback guard for load-by-id estimators.

These stay hermetic by patching `InferenceClient` / `ServiceClient` rather
than talking to a mock server — the concern here is client-side wiring, not
the HTTP layer.
"""

import unittest
from unittest.mock import patch
from uuid import UUID

import numpy as np

from tabpfn_client.client import NeedsRefittingError, PredictionResult, ServiceClient
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.service_wrapper import InferenceClient

_MODEL_ID = "00000000-0000-0000-0000-000000000abc"


def _regressor_prediction() -> PredictionResult:
    return PredictionResult(y_pred={"mean": np.zeros(4)}, metadata={})


def _classifier_prediction() -> PredictionResult:
    return PredictionResult(y_pred=np.zeros(4), metadata={})


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
            cfg = Est(fit_mode="fit_with_cache")._get_tabpfn_config()
            dumped = cfg.model_dump(mode="json", exclude_none=True)
            self.assertEqual(dumped.get("fit_mode"), "fit_with_cache", Est.__name__)

    def test_fit_mode_forwarded_on_predict_task_config(self):
        reg = TabPFNRegressor(fit_mode="fit_with_cache", model_id=_MODEL_ID)
        with patch.object(ServiceClient, "get_model_limits", return_value=None):
            with patch.object(
                InferenceClient, "predict", return_value=_regressor_prediction()
            ) as mock_predict:
                reg.predict(np.random.randn(4, 3))
        task_config = mock_predict.call_args[1]["task_config"]
        dumped = task_config.tabpfn_config.model_dump(mode="json", exclude_none=True)
        self.assertEqual(dumped.get("fit_mode"), "fit_with_cache")


class TestModelIdConstructor(unittest.TestCase):
    def test_model_id_marks_estimator_fitted(self):
        for Est in (TabPFNClassifier, TabPFNRegressor):
            est = Est(model_id=_MODEL_ID)
            self.assertTrue(est.__sklearn_is_fitted__(), Est.__name__)
            self.assertEqual(est._last_fitted_train_set_id, UUID(_MODEL_ID))
            self.assertFalse(Est().__sklearn_is_fitted__(), Est.__name__)

    def test_predict_uses_model_id_without_fitting(self):
        reg = TabPFNRegressor(model_id=_MODEL_ID)
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

    def test_model_id_accepts_uuid_and_str(self):
        self.assertEqual(
            TabPFNRegressor(model_id=UUID(_MODEL_ID))._last_fitted_train_set_id,
            UUID(_MODEL_ID),
        )


class TestSaveLoadModel(unittest.TestCase):
    def test_regressor_round_trip_dict(self):
        reg = TabPFNRegressor(model_id=_MODEL_ID, n_estimators=4)
        handle = reg.save_model()
        self.assertEqual(handle["task"], "regression")
        self.assertNotIn("client_options", handle["params"])
        self.assertEqual(handle["params"]["model_id"], _MODEL_ID)

        loaded = TabPFNRegressor.load_model(handle)
        self.assertTrue(loaded.__sklearn_is_fitted__())
        self.assertEqual(loaded._last_fitted_train_set_id, UUID(_MODEL_ID))
        self.assertEqual(loaded.get_params()["n_estimators"], 4)

    def test_classifier_round_trip_restores_classes(self):
        clf = TabPFNClassifier(model_id=_MODEL_ID)
        clf.classes_ = np.array([0, 1, 2])
        loaded = TabPFNClassifier.load_model(clf.save_model())
        self.assertTrue(loaded.__sklearn_is_fitted__())
        np.testing.assert_array_equal(loaded.classes_, np.array([0, 1, 2]))

    def test_round_trip_via_file(self):
        import tempfile
        from pathlib import Path

        reg = TabPFNRegressor(model_id=_MODEL_ID)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.json"
            returned = reg.save_model(path)
            self.assertEqual(Path(returned), path)
            self.assertTrue(path.exists())
            loaded = TabPFNRegressor.load_model(path)
        self.assertEqual(loaded._last_fitted_train_set_id, UUID(_MODEL_ID))

    def test_effective_id_beats_constructor_model_id(self):
        # A real fit supersedes the constructor's model_id; save_model must
        # persist the effective (post-fit) id, not the original.
        reg = TabPFNRegressor(model_id=_MODEL_ID)
        new_id = "00000000-0000-0000-0000-000000000fff"
        reg._last_fitted_train_set_id = UUID(new_id)
        self.assertEqual(reg.save_model()["params"]["model_id"], new_id)

    def test_load_model_task_mismatch_raises(self):
        clf = TabPFNClassifier(model_id=_MODEL_ID)
        clf.classes_ = np.array([0, 1])
        with self.assertRaises(ValueError):
            TabPFNRegressor.load_model(clf.save_model())

    def test_save_model_unfitted_raises(self):
        from sklearn.exceptions import NotFittedError

        with self.assertRaises(NotFittedError):
            TabPFNRegressor().save_model()

    def test_save_classifier_without_classes_raises(self):
        # A bare model_id classifier is "fitted" (has an id) but has no
        # classes_ to persist; save_model must fail clearly, not AttributeError.
        clf = TabPFNClassifier(model_id=_MODEL_ID)
        with self.assertRaises(ValueError) as ctx:
            clf.save_model()
        self.assertIn("classes_", str(ctx.exception))

    def test_unsupported_handle_version_raises(self):
        handle = TabPFNRegressor(model_id=_MODEL_ID).save_model()
        handle["format_version"] = 999
        with self.assertRaises(ValueError):
            TabPFNRegressor.load_model(handle)


class TestRefitGuard(unittest.TestCase):
    def test_load_by_id_cannot_auto_refit(self):
        reg = TabPFNRegressor(model_id=_MODEL_ID)  # no in-memory train data
        with patch.object(ServiceClient, "get_model_limits", return_value=None):
            with patch.object(InferenceClient, "fit") as mock_fit:
                with patch.object(
                    InferenceClient, "predict", side_effect=NeedsRefittingError()
                ):
                    with self.assertRaises(RuntimeError) as ctx:
                        reg.predict(np.random.randn(4, 3))
        self.assertIn("no in-memory", str(ctx.exception).lower())
        mock_fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
