"""`sample_weight` and `label_reliability` reach the fit, and bad ones fail before upload."""

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
    X = pd.DataFrame({"f0": np.arange(4, dtype=float), "f1": np.arange(4) % 2})
    return X, np.array([0, 1, 0, 1])


@pytest.mark.parametrize("estimator", [TabPFNClassifier, TabPFNRegressor])
@pytest.mark.parametrize("method", ["fit", "submit_fit"])
def test_weights_are_sent_with_the_train_set(estimator, method):
    X, y = _data()
    with patch.object(
        InferenceClient, "fit_with_result", return_value=_FIT_RESULT
    ) as mock_fit:
        getattr(estimator(), method)(X, y, [1, 2, 0, 1], [1.0, 0.5, 1.0, 0.0])

    pd.testing.assert_frame_equal(
        mock_fit.call_args.kwargs["row_weights"],
        pd.DataFrame(
            {
                "sample_weight": [1.0, 2.0, 0.0, 1.0],
                "label_reliability": [1.0, 0.5, 1.0, 0.0],
            }
        ),
    )


def test_a_fit_without_weights_sends_none():
    X, y = _data()
    with patch.object(
        InferenceClient, "fit_with_result", return_value=_FIT_RESULT
    ) as mock_fit:
        TabPFNClassifier().fit(X, y, description="toy data")

    assert mock_fit.call_args.kwargs["row_weights"] is None
    assert mock_fit.call_args.kwargs["description"] == "toy data"


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"sample_weight": [1.0, 1.0]}, "one value per row"),
        ({"sample_weight": "toy data"}, "numbers"),
        ({"sample_weight": [1.0, np.nan, 1.0, 1.0]}, "finite"),
        ({"sample_weight": [1.0, -1.0, 1.0, 1.0]}, "negative"),
        ({"sample_weight": [0.0, 0.0, 0.0, 0.0]}, "all zero"),
        ({"label_reliability": [1.0, 1.5, 1.0, 1.0]}, r"\[0, 1\]"),
    ],
)
def test_bad_weights_fail_before_upload(weights, message):
    X, y = _data()
    with patch.object(InferenceClient, "fit_with_result") as mock_fit:
        with pytest.raises(ValueError, match=message):
            TabPFNClassifier().fit(X, y, **weights)
    mock_fit.assert_not_called()
