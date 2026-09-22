from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from tabpfn_client.api_models import (
    ClassifierTabPFNConfig,
    GetSettingsResponse,
    RegressorTabPFNConfig,
)
from tabpfn_client.client import ServiceClient
from tabpfn_client.estimator import validate_test_set, validate_train_set


DEFAULT_BUDGET = 250_000 * 1_000_000


def _limits(
    test_set_max_rows: int = 1_000_000,
    predict_row_pairs_budget: int = DEFAULT_BUDGET,
) -> GetSettingsResponse:
    model_limit: dict[str, Any] = {
        "train_set_max_rows": 1_000_000,
        "train_set_max_cells": 100_000_000,
        "test_set_max_rows": test_set_max_rows,
        "test_set_max_cells": 100_000_000,
        "test_set_max_rows_w_full_regression_output": 400,
        "max_cols": 2_000,
        "max_classes": 10,
    }
    model_limit["predict_row_pairs_budget"] = predict_row_pairs_budget
    return GetSettingsResponse.model_validate(
        {
            "default_model_version": "v3",
            "max_model_limit": model_limit,
            "model_limits": {"v3": model_limit},
            "dataset_max_size_bytes": 100_000_000,
            "async_settings": {
                "use_above_trainset_size_bytes": 50 * 1024 * 1024,
                "poll_timeout_secs": 7200.0,
            },
        }
    )


def _X(n_rows: int) -> np.ndarray:
    return np.zeros((n_rows, 1))


def _validate(
    n_test_rows: int,
    train_rows: int | None = None,
    limits: GetSettingsResponse | None = None,
):
    with patch.object(ServiceClient, "get_settings", return_value=limits or _limits()):
        validate_test_set(_X(n_test_rows), None, train_rows=train_rows)


def test_static_limit_applies_without_train_rows():
    _validate(1_000_000)
    with pytest.raises(ValueError, match="exceeds the maximum of 1000000"):
        _validate(1_000_001)


def test_adaptive_limit_shrinks_with_train_rows():
    # 1M train rows -> 250k test rows per call
    _validate(250_000, train_rows=1_000_000)
    with pytest.raises(ValueError, match="exceeds the maximum of 250000"):
        _validate(250_001, train_rows=1_000_000)


def test_adaptive_limit_never_exceeds_static_limit():
    # Small train sets leave the static cap binding
    small_train = DEFAULT_BUDGET // 1_000_000 // 10
    _validate(1_000_000, train_rows=small_train)
    with pytest.raises(ValueError, match="exceeds the maximum of 1000000"):
        _validate(1_000_001, train_rows=small_train)


def test_smaller_budget_shrinks_adaptive_limit():
    # Half the default budget -> 125k test rows for 1M train rows
    limits = _limits(predict_row_pairs_budget=125_000 * 1_000_000)
    _validate(125_000, train_rows=1_000_000, limits=limits)
    with pytest.raises(ValueError, match="exceeds the maximum of 125000"):
        _validate(125_001, train_rows=1_000_000, limits=limits)


def test_no_limits_available_skips_validation():
    with patch.object(ServiceClient, "get_settings", return_value=None):
        validate_test_set(_X(2_000_000), None, train_rows=1_000_000)


@pytest.mark.parametrize("config_type", [ClassifierTabPFNConfig, RegressorTabPFNConfig])
def test_subsampled_union_has_separate_upload_and_context_limits(
    config_type: type[ClassifierTabPFNConfig] | type[RegressorTabPFNConfig],
) -> None:
    settings = _limits(predict_row_pairs_budget=100)
    for limit in (settings.max_model_limit, *settings.model_limits.values()):
        limit.train_set_max_upload_cells = 300
        limit.train_set_max_cells = 200
    config = config_type(
        n_estimators=2,
        inference_config={"SUBSAMPLE_SAMPLES": [list(range(10)), list(range(15))]},
    )
    with patch.object(ServiceClient, "get_settings", return_value=settings):
        validate_train_set(np.zeros((30, 10)), tabpfn_config=config)
        validate_test_set(np.zeros((6, 10)), None, train_rows=30, tabpfn_config=config)
        with pytest.raises(ValueError, match="upload"):
            validate_train_set(np.zeros((31, 10)), tabpfn_config=config)
        with pytest.raises(ValueError, match="estimator"):
            validate_train_set(np.zeros((30, 10)), tabpfn_config=config_type())
        with pytest.raises(ValueError, match="maximum of 6"):
            validate_test_set(
                np.zeros((7, 10)), None, train_rows=30, tabpfn_config=config
            )
