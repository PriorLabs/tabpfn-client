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
_INVALID_V4 = (
    "Fail to call fit: [HTTP 422] [{'type': 'value_error', 'msg': 'Value error, "
    "Invalid model path: v4_default. Valid model names: v3.5_default, v3_default'}]."
)
ESTIMATORS = [TabPFNClassifier, TabPFNRegressor]


@pytest.fixture(autouse=True)
def _server_mode():
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


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
def test_a_rejected_v4_model_path_explains_early_access(estimator_cls):
    model = estimator_cls.create_default_for_version(ModelVersion.V4)
    with patch.object(
        InferenceClient, "fit_with_result", side_effect=RuntimeError(_INVALID_V4)
    ):
        with pytest.raises(RuntimeError, match="early access") as exc_info:
            model.fit(*_data(estimator_cls))
    assert "v4_default" in str(exc_info.value)
    assert str(exc_info.value.__cause__) == _INVALID_V4


@pytest.mark.parametrize("estimator_cls", ESTIMATORS)
@pytest.mark.parametrize(
    ("model_path", "error"),
    [
        ("v4_default", "Fail to call fit: [HTTP 500] Internal Server Error."),
        ("v3_unknown", "Fail to call fit: [HTTP 422] Invalid model path: v3_unknown."),
    ],
)
def test_other_errors_pass_through_unchanged(estimator_cls, model_path, error):
    model = estimator_cls(model_path=model_path)
    with patch.object(
        InferenceClient, "fit_with_result", side_effect=RuntimeError(error)
    ):
        with pytest.raises(RuntimeError) as exc_info:
            model.fit(*_data(estimator_cls))
    assert str(exc_info.value) == error
