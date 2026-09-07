"""Estimate quota cost on the server using only dataset dimensions."""

from typing import Literal

import numpy as np
import pandas as pd

from tabpfn_client.api_models import EstimateCostRequest, EstimateCostResponse
from tabpfn_client.client import ServiceClient
from tabpfn_client.config import get_access_token


def _shape(X: np.ndarray | pd.DataFrame, name: str) -> tuple[int, int]:
    shape = getattr(X, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError(f"{name} must be a two-dimensional array or DataFrame")
    rows, columns = int(shape[0]), int(shape[1])
    if rows < 0 or columns <= 0 or (name == "X_train" and rows == 0):
        raise ValueError(f"{name} must have valid row and feature counts")
    return rows, columns


def estimate_cost(
    X_train: np.ndarray | pd.DataFrame,
    X_test: np.ndarray | pd.DataFrame | None = None,
    *,
    model_version: str | None = None,
    operation: Literal[
        "predict", "thinking_fit", "thinking_predict", "cache_predict"
    ] = "predict",
    n_estimators: int | None = None,
    thinking_effort: Literal["medium", "high"] | None = None,
) -> EstimateCostResponse:
    """Estimate one operation without uploading data or consuming quota.

    Only raw row/column counts and the supplied configuration are sent. The
    server resolves omitted model/ensemble defaults and returns them in
    ``result.inputs`` alongside ``estimated_cost`` and ``pricing_version``.
    Quota v3 must be enabled on the server. Costs are tokens for ``quota_v3``
    and legacy cell-prediction credits for ``legacy_v2``.

    ``thinking_fit`` takes no X_test; its default effort is medium. For
    ``thinking_predict``, supply the fitted model's version and actual
    per-base-estimator count. ``cache_predict`` assumes a cache hit; fallback
    or different fitted estimator counts can change the final charge.
    A quote does not guarantee model access or dataset eligibility.
    """
    train_rows, raw_columns = _shape(X_train, "X_train")
    test_rows = 0
    if X_test is not None:
        if operation == "thinking_fit":
            raise ValueError("thinking_fit does not use X_test")
        test_rows, test_columns = _shape(X_test, "X_test")
        if test_columns != raw_columns:
            raise ValueError("X_train and X_test must have the same number of features")
    if n_estimators is not None and (
        type(n_estimators) is not int or n_estimators <= 0
    ):
        raise ValueError("n_estimators must be a positive integer")
    req = EstimateCostRequest.model_validate(
        {
            "train_rows": train_rows,
            "test_rows": test_rows,
            "raw_columns": raw_columns,
            "model_version": model_version,
            "operation": operation,
            "n_estimators": n_estimators,
            "thinking_effort": thinking_effort,
        }
    )
    return ServiceClient.estimate_cost(req, access_token=get_access_token())
