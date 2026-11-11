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


def _thinking_limit(max_timeout_secs: dict[str, float]) -> dict:
    limit = _api_settings_payload()["max_model_limit"]
    return {
        **limit,
        "efforts": list(max_timeout_secs),
        "max_timeout_secs": max_timeout_secs,
    }


def _use_level_caps() -> None:
    _use_settings(
        thinking_max_timeout_secs=5_000.0,
        thinking_default_model_version="v3.5",
        thinking_model_limits={
            "v3.5": _thinking_limit({"micro": 100.0, "max": 5_000.0}),
            "v3": _thinking_limit({"medium": 900.0, "high": 900.0}),
        },
    )


@pytest.mark.parametrize(
    ("model_path", "effort", "cap"),
    [
        (None, "micro", 100),
        ("auto", "micro", 100),
        ("tabpfn-v3-classifier-v3_default.ckpt", "high", 900),
    ],
)
def test_fit_rejects_a_timeout_above_the_level_s_cap_before_upload(
    model_path, effort, cap
):
    _use_level_caps()
    kwargs = {} if model_path is None else {"model_path": model_path}
    with patch.object(InferenceClient, "fit_with_result") as mock_fit:
        with pytest.raises(
            ValueError,
            match=f"exceeds the maximum of {cap} seconds for thinking_effort='{effort}'",
        ):
            TabPFNClassifier(
                thinking_effort=effort, thinking_timeout_s=cap + 1, **kwargs
            ).fit(_X, _Y)
    mock_fit.assert_not_called()


def test_fit_sends_a_timeout_at_the_level_s_cap():
    _use_level_caps()
    with patch.object(
        InferenceClient, "fit_with_result", return_value=_FIT_RESULT
    ) as mock_fit:
        TabPFNClassifier(thinking_effort="micro", thinking_timeout_s=100).fit(_X, _Y)
    assert mock_fit.call_args.kwargs["thinking_config"].timeout_secs == 100


def test_fit_without_an_effort_checks_the_longest_timeout_any_level_takes():
    _use_level_caps()
    with patch.object(InferenceClient, "fit_with_result") as mock_fit:
        with pytest.raises(
            ValueError, match="exceeds the maximum of 5000 seconds\\.$"
        ):
            TabPFNClassifier(thinking_mode=True, thinking_timeout_s=5_001).fit(_X, _Y)
    mock_fit.assert_not_called()
