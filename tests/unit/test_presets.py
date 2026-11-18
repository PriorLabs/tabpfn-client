"""`presets` reach the fit, and the classifier refuses the regression-only one."""

import time
from uuid import UUID
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tabpfn_client import config
from tabpfn_client.client import GetSettingsResponse, ServiceClient
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.api_models import FitResponse, FitStatus
from tabpfn_client.service_wrapper import InferenceClient
from tests.unit.test_tabpfn_classifier import _api_settings_payload

_FIT_RESULT = FitResponse(fitted_train_set_id=UUID(int=0), status=FitStatus.COMPLETED)


@pytest.fixture(autouse=True)
def skip_init():
    config.Config.is_initialized = True
    ServiceClient._api_settings = GetSettingsResponse(**_api_settings_payload())
    ServiceClient._api_settings_ts = time.monotonic()
    yield
    ServiceClient._api_settings = None
    ServiceClient._api_settings_ts = 0.0
    config.reset()


def _data() -> tuple[pd.DataFrame, np.ndarray]:
    X = pd.DataFrame({"f0": np.arange(20, dtype=float), "f1": np.arange(20) % 4})
    return X, np.arange(20) % 2


@pytest.mark.parametrize("presets", ["zero_inflated", ["grouped", "zero_inflated"]])
def test_classifier_rejects_the_zero_inflated_preset_before_upload(presets):
    X, y = _data()
    with patch.object(InferenceClient, "fit_with_result") as mock_fit:
        with pytest.raises(ValueError, match="regression only"):
            TabPFNClassifier(presets=presets).fit(X, y)
    mock_fit.assert_not_called()


def test_regressor_sends_the_zero_inflated_preset():
    X, y = _data()
    with patch.object(
        InferenceClient, "fit_with_result", return_value=_FIT_RESULT
    ) as mock_fit:
        TabPFNRegressor(presets=["zero_inflated", "grouped"]).fit(X, y.astype(float))

    presets = mock_fit.call_args.kwargs["task_config"].tabpfn_config.presets
    assert presets == ["zero_inflated", "grouped"]
