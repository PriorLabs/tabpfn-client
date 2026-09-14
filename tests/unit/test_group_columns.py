"""`group_col`, `time_col` and `group_time_col` are checked client side and sent with the fit."""

import time
from uuid import UUID
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tabpfn_client import config
from tabpfn_client.client import GetSettingsResponse, ServiceClient
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.service_wrapper import InferenceClient
from tests.unit.test_tabpfn_classifier import _api_settings_payload


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
    X = pd.DataFrame(
        {
            "f0": np.arange(20, dtype=float),
            "patient": [f"p{i % 5}" for i in range(20)],
            "visit": np.arange(20) % 4,
        }
    )
    return X, np.arange(20) % 2


def test_fit_sends_group_columns_in_the_thinking_config():
    X, y = _data()
    with patch.object(InferenceClient, "fit", return_value="dummy_uid") as mock_fit:
        TabPFNClassifier(
            thinking_mode=True, group_col="patient", group_time_col="visit"
        ).fit(X, y)

    thinking_config = mock_fit.call_args.kwargs["thinking_config"]
    assert thinking_config.group_col == "patient"
    assert thinking_config.time_col is None
    assert thinking_config.group_time_col == "visit"


def test_regressor_fit_sends_time_col_in_the_thinking_config():
    X, y = _data()
    with patch.object(InferenceClient, "fit", return_value="dummy_uid") as mock_fit:
        TabPFNRegressor(thinking_mode=True, time_col="visit").fit(X, y.astype(float))

    assert mock_fit.call_args.kwargs["thinking_config"].time_col == "visit"


@pytest.mark.parametrize(
    ("kwargs", "X", "match"),
    [
        ({"group_col": "patient"}, _data()[0], "only supported in thinking mode"),
        (
            {"thinking_mode": True, "group_col": "patient"},
            _data()[0].to_numpy(),
            "must be a pandas DataFrame",
        ),
        (
            {"thinking_mode": True, "group_col": "session"},
            _data()[0],
            "missing the columns",
        ),
        (
            {"thinking_mode": True, "group_col": "patient", "time_col": "visit"},
            _data()[0],
            "cannot be combined",
        ),
        (
            {"thinking_mode": True, "group_time_col": "visit"},
            _data()[0],
            "requires `group_col`",
        ),
    ],
)
def test_fit_rejects_invalid_group_columns_before_upload(kwargs, X, match):
    _, y = _data()
    with patch.object(InferenceClient, "fit") as mock_fit:
        with pytest.raises(ValueError, match=match):
            TabPFNClassifier(**kwargs).fit(X, y)
    mock_fit.assert_not_called()


def test_predict_rejects_a_frame_without_the_group_col():
    X, _ = _data()
    classifier = TabPFNClassifier(thinking_mode=True, group_col="patient")
    classifier.model_id_ = UUID("00000000-0000-0000-0000-000000000000")
    classifier.classes_ = np.array([0, 1])
    classifier._n_train_rows = len(X)

    with patch.object(InferenceClient, "predict") as mock_predict:
        with pytest.raises(ValueError, match="missing the columns"):
            classifier.predict_proba(X.drop(columns=["patient"]))
    mock_predict.assert_not_called()
