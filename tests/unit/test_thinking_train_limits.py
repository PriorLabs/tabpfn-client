"""A thinking fit's train set is checked against the server's thinking limits."""

import time
from uuid import UUID
from unittest.mock import patch

import numpy as np
import pytest

from tabpfn_client import config
from tabpfn_client.api_models import FitResponse, FitStatus
from tabpfn_client.client import GetSettingsResponse, ServiceClient
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.service_wrapper import InferenceClient
from tests.unit.test_tabpfn_classifier import _api_settings_payload

_FIT_RESULT = FitResponse(fitted_train_set_id=UUID(int=0), status=FitStatus.COMPLETED)
_X = np.random.rand(20, 3)
_Y = np.arange(20) % 2


def _use_settings(*, thinking_upload_cells: int | None) -> None:
    payload = _api_settings_payload()
    for limit in (payload["max_model_limit"], payload["model_limits"]["v2.5"]):
        limit["train_set_max_upload_cells"] = _X.size - 1
    if thinking_upload_cells is not None:
        thinking_limit = {
            **_api_settings_payload()["max_model_limit"],
            "train_set_max_upload_cells": thinking_upload_cells,
            "efforts": ["medium"],
        }
        payload["thinking_default_model_version"] = "v2.5"
        payload["thinking_model_limits"] = {"v2.5": thinking_limit}
    ServiceClient._api_settings = GetSettingsResponse(**payload)
    ServiceClient._api_settings_ts = time.monotonic()


@pytest.fixture(autouse=True)
def skip_init():
    config.Config.is_initialized = True
    yield
    ServiceClient._api_settings = None
    ServiceClient._api_settings_ts = 0.0
    config.reset()


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test_thinking_fit_uses_the_thinking_upload_limit(estimator_cls):
    _use_settings(thinking_upload_cells=_X.size)
    with patch.object(
        InferenceClient, "fit_with_result", return_value=_FIT_RESULT
    ) as mock_fit:
        estimator_cls(thinking_mode=True).fit(_X, _Y)
    mock_fit.assert_called_once()


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
def test_plain_fit_keeps_the_model_upload_limit(estimator_cls):
    _use_settings(thinking_upload_cells=_X.size)
    with patch.object(InferenceClient, "fit_with_result") as mock_fit:
        with pytest.raises(ValueError, match="training upload has 60 cells"):
            estimator_cls().fit(_X, _Y)
    mock_fit.assert_not_called()


def test_thinking_fit_falls_back_to_the_model_limit_without_thinking_limits():
    _use_settings(thinking_upload_cells=None)
    with patch.object(InferenceClient, "fit_with_result") as mock_fit:
        with pytest.raises(ValueError, match="training upload has 60 cells"):
            TabPFNClassifier(thinking_mode=True).fit(_X, _Y)
    mock_fit.assert_not_called()
