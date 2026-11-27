from unittest.mock import patch
from uuid import UUID

import numpy as np
import pytest

from tabpfn_client import config
from tabpfn_client.api_models import FitResponse, FitStatus, ModelVersion
from tabpfn_client.client import ServiceClient
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.service_wrapper import InferenceClient

_DUMMY_FIT = FitResponse(fitted_train_set_id=UUID(int=0), status=FitStatus.COMPLETED)
ESTIMATORS = [TabPFNClassifier, TabPFNRegressor]


@pytest.fixture(autouse=True)
def _server_mode():  # pyright: ignore[reportUnusedFunction]
    config.Config.is_initialized = True
    config.Config.use_server = True
    with patch.object(ServiceClient, "get_settings", return_value=None):
        yield
    config.reset()


def _data(estimator_cls):
    X = np.random.rand(10, 3)
    if estimator_cls is TabPFNClassifier:
        return X, np.array([0, 1] * 5)
    return X, np.random.rand(10)


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"thinking_mode": True}, "thinking mode"),
        ({"thinking_effort": "low"}, "thinking mode"),
        ({"fit_mode": "fit_with_cache"}, "fit_with_cache"),
    ],
)
def test_v4_rejects_unsupported_options_before_upload(estimator_cls, options, match):
    model = estimator_cls.create_default_for_version(ModelVersion.V4, **options)
    with patch.object(InferenceClient, "fit_with_result") as mock_fit:
        with pytest.raises(ValueError, match=match):
            model.fit(*_data(estimator_cls))
    mock_fit.assert_not_called()


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
def test_v4_fits_with_the_v4_alias(estimator_cls):
    model = estimator_cls.create_default_for_version(ModelVersion.V4)
    with patch.object(
        InferenceClient, "fit_with_result", return_value=_DUMMY_FIT
    ) as mock_fit:
        model.fit(*_data(estimator_cls))
    task_config = mock_fit.call_args.kwargs["task_config"]
    assert task_config.tabpfn_config.model_path == "v4_default"


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
def test_fit_with_cache_stays_allowed_for_other_versions(estimator_cls):
    model = estimator_cls.create_default_for_version(
        ModelVersion.V3_5, fit_mode="fit_with_cache"
    )
    with patch.object(InferenceClient, "fit_with_result", return_value=_DUMMY_FIT):
        model.fit(*_data(estimator_cls))
