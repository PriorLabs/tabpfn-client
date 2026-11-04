"""`thinking_timeout_s` is checked against the server's limit before the upload."""

import time
from uuid import UUID
from unittest.mock import patch

import numpy as np
import pytest

from tabpfn_client import config
from tabpfn_client.client import GetSettingsResponse, ServiceClient
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.api_models import FitResponse, FitStatus
from tabpfn_client.service_wrapper import InferenceClient
from tests.unit.test_tabpfn_classifier import _api_settings_payload

_FIT_RESULT = FitResponse(fitted_train_set_id=UUID(int=0), status=FitStatus.COMPLETED)
_X = np.random.rand(20, 3)
_Y = np.arange(20) % 2


def _use_settings(**overrides) -> None:
    ServiceClient._api_settings = GetSettingsResponse(
        **{**_api_settings_payload(), **overrides}
    )
    ServiceClient._api_settings_ts = time.monotonic()


@pytest.fixture(autouse=True)
def skip_init():
    config.Config.is_initialized = True
    yield
    ServiceClient._api_settings = None
    ServiceClient._api_settings_ts = 0.0
    config.reset()


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test_fit_rejects_a_timeout_above_the_server_limit_before_upload(estimator_cls):
    _use_settings(thinking_max_timeout_secs=600.0)
    with patch.object(InferenceClient, "fit_with_result") as mock_fit:
        with pytest.raises(ValueError, match="exceeds the maximum of 600 seconds"):
            estimator_cls(thinking_mode=True, thinking_timeout_s=601).fit(_X, _Y)
    mock_fit.assert_not_called()


def test_fit_sends_a_timeout_at_the_server_limit():
    _use_settings(thinking_max_timeout_secs=600.0)
    with patch.object(
        InferenceClient, "fit_with_result", return_value=_FIT_RESULT
    ) as mock_fit:
        TabPFNClassifier(thinking_mode=True, thinking_timeout_s=600).fit(_X, _Y)
    assert mock_fit.call_args.kwargs["thinking_config"].timeout_secs == 600


def test_fit_leaves_the_check_to_a_server_that_reports_no_limit():
    _use_settings()
    with patch.object(
        InferenceClient, "fit_with_result", return_value=_FIT_RESULT
    ) as mock_fit:
        TabPFNClassifier(thinking_mode=True, thinking_timeout_s=10_000).fit(_X, _Y)
    assert mock_fit.call_args.kwargs["thinking_config"].timeout_secs == 10_000
