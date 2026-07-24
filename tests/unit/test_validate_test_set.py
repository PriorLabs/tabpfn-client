from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from tabpfn_client.api_models import GetModelLimitsResponse
from tabpfn_client.client import ServiceClient
from tabpfn_client.estimator import PREDICT_ROW_PAIRS_BUDGET, validate_test_set


def _limits(test_set_max_rows: int = 1_000_000) -> GetModelLimitsResponse:
    model_limit: dict[str, Any] = {
        "train_set_max_rows": 1_000_000,
        "train_set_max_cells": 100_000_000,
        "test_set_max_rows": test_set_max_rows,
        "test_set_max_cells": 100_000_000,
        "test_set_max_rows_w_full_regression_output": 400,
        "max_cols": 2_000,
        "max_classes": 10,
    }
    return GetModelLimitsResponse.model_validate(
        {
            "default_model_version": "v3",
            "max_model_limit": model_limit,
            "model_limits": {"v3": model_limit},
            "dataset_max_size_bytes": 100_000_000,
        }
    )


def _X(n_rows: int) -> np.ndarray:
    return np.zeros((n_rows, 1))


def _validate(n_test_rows: int, train_rows: int | None = None):
    with patch.object(ServiceClient, "get_model_limits", return_value=_limits()):
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
    small_train = PREDICT_ROW_PAIRS_BUDGET // 1_000_000 // 10
    _validate(1_000_000, train_rows=small_train)
    with pytest.raises(ValueError, match="exceeds the maximum of 1000000"):
        _validate(1_000_001, train_rows=small_train)


def test_no_limits_available_skips_validation():
    with patch.object(ServiceClient, "get_model_limits", return_value=None):
        validate_test_set(_X(2_000_000), None, train_rows=1_000_000)
