"""Unit tests for the KV-cache client surface:

- `fit_mode` reaching the server-side `tabpfn_config` on the wire,
- `model_id_` (the fitted-train-set id, written by `fit()` or assigned
  directly to reuse a previous fit) as the single fitted-state slot that
  makes an estimator predictable.

These stay hermetic by patching `InferenceClient` / `ServiceClient` rather
than talking to a mock server — the concern here is client-side wiring, not
the HTTP layer.
"""

import unittest
from unittest.mock import patch
from uuid import UUID

import numpy as np

from tabpfn_client.api_models import FitMode
from tabpfn_client.client import PredictionResult, ServiceClient
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
            with patch.object(ServiceClient, "get_settings", return_value=None):
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
            with patch.object(ServiceClient, "get_settings", return_value=None):
                with patch.object(InferenceClient, "fit") as mock_fit:
                    with patch.object(
                        InferenceClient, "predict", return_value=_regressor_prediction()
                    ) as mock_predict:
                        reg.predict(np.random.randn(4, 3))
        mock_fit.assert_not_called()
        self.assertEqual(
            mock_predict.call_args[1]["fitted_train_set_id"], UUID(_MODEL_ID)
        )


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
        # A fresh process that only assigns a saved `model_id_` and predicts
        # never touches `fit()`, so `_predict` itself must call `init()` to
        # authorize the HTTP client — otherwise the main cross-process reuse
        # path 401s.
        reg = TabPFNRegressor()
        reg.model_id_ = UUID(_MODEL_ID)
        with patch("tabpfn_client.estimator.init") as mock_init:
            with patch.object(ServiceClient, "get_settings", return_value=None):
                with patch.object(
                    InferenceClient, "predict", return_value=_regressor_prediction()
                ):
                    reg.predict(np.random.randn(4, 3))
        mock_init.assert_called()


if __name__ == "__main__":
    unittest.main()
