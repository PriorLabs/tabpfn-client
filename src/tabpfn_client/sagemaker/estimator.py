#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
"""scikit-learn estimators that invoke a TabPFN SageMaker BYOC endpoint.

The endpoint is the container defined in `dists/marketplaces/aws` in the
`tabpfn-server` repo. It accepts a single inline JSON body at POST
/invocations matching `prior.predictor.requests.PredictRequest`, and returns
`{"prediction": ..., "metadata": ..., "model_id": ...}`. The estimators here
build that body from the sklearn-style call surface, dispatch via
`boto3.client("sagemaker-runtime").invoke_endpoint`, and return the
prediction as a numpy array.

`fit()` is local-only: TabPFN is in-context, so we just keep X/y around and
ship them with each `predict*` call. The optional `use_kv_cache=True` path
opts into the server's V3 FIT_WITH_CACHE mode — the first round-trip uploads
training data and captures a `model_id`; subsequent predicts skip the upload
and reference that id.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

try:
    import boto3  # type: ignore[import-untyped]

    _BOTO3_AVAILABLE = True
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]
    _BOTO3_AVAILABLE = False


ThinkingEffort = Literal["medium", "high"]


def _require_boto3() -> None:
    if not _BOTO3_AVAILABLE:
        raise ImportError(
            "boto3 is required for tabpfn_client.sagemaker. "
            "Install with: pip install 'tabpfn-client[sagemaker]'"
        )


def _to_jsonable(X: Any) -> list:
    """Coerce numpy / pandas inputs to plain Python lists for JSON."""
    if isinstance(X, pd.DataFrame):
        return X.values.tolist()
    if isinstance(X, pd.Series):
        return X.tolist()
    return np.asarray(X).tolist()


class _SagemakerBase(BaseEstimator):
    """Shared invoke_endpoint plumbing for SageMaker TabPFN estimators.

    Subclasses set `_TASK`. Constructor kwargs mirror the public
    `tabpfn_client.TabPFNClassifier` so user code is portable; everything but
    the SageMaker-specific bits is forwarded into `task_config.tabpfn_config`
    on the wire. `model_path` is currently dropped server-side (the active
    checkpoint is whatever was baked into the model artifact); we keep it on
    the constructor for API parity.
    """

    _TASK: str = ""  # overridden by subclasses

    def __init__(
        self,
        endpoint_name: str,
        region_name: Optional[str] = None,
        boto_session: Optional[Any] = None,
        model_path: str = "auto",
        n_estimators: int = 8,
        softmax_temperature: float = 0.9,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        ignore_pretraining_limits: bool = True,
        inference_precision: Literal["autocast", "auto"] = "auto",
        random_state: Optional[int] = 0,
        inference_config: Optional[Dict[str, Any]] = None,
        paper_version: bool = False,
        thinking_mode: bool = False,
        thinking_effort: Optional[ThinkingEffort] = None,
        thinking_timeout_s: Optional[float] = None,
        thinking_metric: Optional[str] = None,
        use_kv_cache: bool = False,
    ):
        self.endpoint_name = endpoint_name
        self.region_name = region_name
        self.boto_session = boto_session
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision = inference_precision
        self.random_state = random_state
        self.inference_config = inference_config
        self.paper_version = paper_version
        self.thinking_mode = thinking_mode
        self.thinking_effort = thinking_effort
        self.thinking_timeout_s = thinking_timeout_s
        self.thinking_metric = thinking_metric
        self.use_kv_cache = use_kv_cache

    def _build_tabpfn_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "softmax_temperature": self.softmax_temperature,
            "average_before_softmax": self.average_before_softmax,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
            "inference_precision": self.inference_precision,
            "random_state": self.random_state,
            "inference_config": self.inference_config,
            "fit_mode": "fit_with_cache" if self.use_kv_cache else "fit_preprocessors",
        }
        # paper_version is OSS TabPFN-only — server's tagged-union forbids
        # extras. balance_probabilities lives only on ClassifierTabPFNConfig.
        if self._TASK == "classification":
            cfg["balance_probabilities"] = self.balance_probabilities
        if self.thinking_mode or self.thinking_effort is not None:
            cfg["thinking_mode"] = True
            if self.thinking_effort is not None:
                cfg["thinking_effort"] = self.thinking_effort
            if self.thinking_timeout_s is not None:
                cfg["thinking_timeout_s"] = self.thinking_timeout_s
            if self.thinking_metric is not None:
                cfg["thinking_metric"] = self.thinking_metric
        if self.model_path not in ("auto", "default"):
            cfg["model_path"] = self.model_path
        return cfg

    def _runtime_client(self):
        # Cached on the instance: boto3 service-model load + credential
        # resolution is non-trivial and we don't want it on every predict.
        client = getattr(self, "_cached_client", None)
        if client is not None:
            return client
        _require_boto3()
        if self.boto_session is not None:
            client = self.boto_session.client("sagemaker-runtime")
        elif self.region_name is not None:
            client = boto3.client("sagemaker-runtime", region_name=self.region_name)
        else:
            client = boto3.client("sagemaker-runtime")
        self._cached_client = client
        return client

    def __getstate__(self) -> Dict[str, Any]:
        # boto3 clients aren't pickleable. Strip the cache so the estimator
        # stays compatible with sklearn's pickle-based parallel/grid utilities.
        state = self.__dict__.copy()
        state.pop("_cached_client", None)
        return state

    def fit(self, X: Any, y: Any) -> "_SagemakerBase":
        # X must be 2D; only DataFrame/array. y can be 1D (Series/array) or 2D.
        X_arr = X if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples; "
                f"got X={X_arr.shape}, y={y_arr.shape}"
            )
        self.X_train_ = X_arr
        self.y_train_ = y_arr
        self._cached_model_id: Optional[str] = None
        if self._TASK == "classification":
            self.classes_ = np.unique(y_arr)
        return self

    def _invoke(
        self,
        X_test: Any,
        output_type: str,
        predict_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        check_is_fitted(self, ["X_train_", "y_train_"])
        params: Dict[str, Any] = {"output_type": output_type}
        if predict_params:
            params.update(predict_params)
        body: Dict[str, Any] = {
            "task_config": {
                "task": self._TASK,
                "tabpfn_config": self._build_tabpfn_config(),
                "predict_params": params,
            },
            "X_test": _to_jsonable(X_test),
        }
        if self.use_kv_cache and self._cached_model_id is not None:
            body["context"] = {"model_id": self._cached_model_id}
        else:
            body["X_train"] = _to_jsonable(self.X_train_)
            # `y_train` on the wire is 2D (n_samples, 1) per PredictRequest.
            # np.asarray handles pd.Series too, so the single-path form covers
            # ndarray / list / DataFrame / Series uniformly.
            y_arr = np.asarray(self.y_train_)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            body["y_train"] = y_arr.tolist()
        resp = self._runtime_client().invoke_endpoint(
            EndpointName=self.endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps(body).encode("utf-8"),
        )
        # StreamingBody is file-like; json.load avoids buffering the full
        # response in memory, which matters for output_type="full".
        payload = json.load(resp["Body"])
        if self.use_kv_cache:
            self._cached_model_id = payload.get("model_id") or self._cached_model_id
        return payload


class TabPFNClassifier(_SagemakerBase, ClassifierMixin):
    """TabPFN classifier backed by a SageMaker real-time endpoint.

    Example:
        from tabpfn_client.sagemaker import TabPFNClassifier
        clf = TabPFNClassifier(
            endpoint_name="tabpfn-sm-alpha-v3-thinking-001",
            region_name="us-east-1",
        )
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        clf.predict_proba(X_test)
    """

    _TASK = "classification"

    def predict(self, X: Any) -> np.ndarray:
        result = self._invoke(X, output_type="preds")
        return np.asarray(result["prediction"])

    def predict_proba(self, X: Any) -> np.ndarray:
        result = self._invoke(X, output_type="probas")
        return np.asarray(result["prediction"])


class TabPFNRegressor(_SagemakerBase, RegressorMixin):
    """TabPFN regressor backed by a SageMaker real-time endpoint.

    `output_type` defaults to "mean"; pass "median", "mode", "full", or
    "quantiles" for the alternative distributional outputs the server
    exposes. When `output_type="quantiles"`, `quantiles` selects the cut
    points (each in [0, 1]).
    """

    _TASK = "regression"

    def predict(
        self,
        X: Any,
        output_type: str = "mean",
        quantiles: Optional[list] = None,
    ) -> np.ndarray:
        predict_params: Dict[str, Any] = {}
        if quantiles is not None:
            predict_params["quantiles"] = quantiles
        result = self._invoke(X, output_type=output_type, predict_params=predict_params)
        return np.asarray(result["prediction"])
