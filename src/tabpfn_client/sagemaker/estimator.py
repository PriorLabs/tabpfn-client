#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""scikit-learn estimators for the TabPFN SageMaker BYOC endpoint.

Mirrors the `tabpfn_client.TabPFNClassifier` / `TabPFNRegressor` surface;
each `predict*` call dispatches via `boto3.client("sagemaker-runtime")`
against your SageMaker endpoint. `fit()` does not call the endpoint —
it just stores `X` / `y` on the estimator. The training data is shipped
to the endpoint on the next `predict*` call, where the actual fit runs.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

try:
    import boto3  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

_BOTO3_AVAILABLE = boto3 is not None


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
    """Shared invoke_endpoint plumbing for the SageMaker TabPFN estimators."""

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
        use_async: bool = False,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "async-io",
        async_poll_interval_s: float = 2.0,
        async_timeout_s: float = 60 * 60,
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
        # Async Inference path: needed for payloads > 6 MB or compute > 60 s.
        self.use_async = use_async
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.async_poll_interval_s = async_poll_interval_s
        self.async_timeout_s = async_timeout_s

    @property
    def _thinking_active(self) -> bool:
        return self.thinking_mode or self.thinking_effort is not None

    @property
    def _effective_use_kv_cache(self) -> bool:
        # Thinking implies caching: without it every predict redoes the fit.
        return self.use_kv_cache or self._thinking_active

    def _build_tabpfn_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "softmax_temperature": self.softmax_temperature,
            "average_before_softmax": self.average_before_softmax,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
            "inference_precision": self.inference_precision,
            "random_state": self.random_state,
            "inference_config": self.inference_config,
            "fit_mode": "fit_with_cache"
            if self._effective_use_kv_cache
            else "fit_preprocessors",
        }
        if self._TASK == "classification":
            cfg["balance_probabilities"] = self.balance_probabilities
        if self.model_path not in ("auto", "default"):
            cfg["model_path"] = self.model_path
        return cfg

    def _build_thinking_block(self) -> Dict[str, Any]:
        """Top-level wire fields for thinking-mode. Empty when inactive."""
        if not self._thinking_active:
            return {}
        block: Dict[str, Any] = {
            "thinking_effort": self.thinking_effort
            if self.thinking_effort is not None
            else "medium",
        }
        if self.thinking_timeout_s is not None:
            block["thinking_timeout_s"] = self.thinking_timeout_s
        if self.thinking_metric is not None:
            block["thinking_metric"] = self.thinking_metric
        return block

    def _runtime_client(self):
        # Cache the client: boto3 client construction is expensive per call.
        client = getattr(self, "_cached_client", None)
        if client is not None:
            return client
        _require_boto3()
        assert boto3 is not None
        if self.boto_session is not None:
            client = self.boto_session.client("sagemaker-runtime")
        elif self.region_name is not None:
            client = boto3.client("sagemaker-runtime", region_name=self.region_name)
        else:
            client = boto3.client("sagemaker-runtime")
        self._cached_client = client
        return client

    def _s3_client(self):
        client = getattr(self, "_cached_s3", None)
        if client is not None:
            return client
        _require_boto3()
        assert boto3 is not None
        if self.boto_session is not None:
            client = self.boto_session.client("s3")
        elif self.region_name is not None:
            client = boto3.client("s3", region_name=self.region_name)
        else:
            client = boto3.client("s3")
        self._cached_s3 = client
        return client

    def __getstate__(self) -> Dict[str, Any]:
        # boto3 clients aren't pickleable; strip the cache for sklearn pickling.
        state = self.__dict__.copy()
        state.pop("_cached_client", None)
        state.pop("_cached_s3", None)
        return state

    def fit(self, X: Any, y: Any) -> "_SagemakerBase":
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
            **self._build_thinking_block(),
        }
        if self._effective_use_kv_cache and self._cached_model_id is not None:
            body["context"] = {"model_id": self._cached_model_id}
        else:
            body["X_train"] = _to_jsonable(self.X_train_)
            # y_train on the wire is 2D (n_samples, 1).
            y_arr = np.asarray(self.y_train_)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            body["y_train"] = y_arr.tolist()
        body_bytes = json.dumps(body).encode("utf-8")
        if self.use_async:
            payload = self._invoke_async(body_bytes)
        else:
            resp = self._runtime_client().invoke_endpoint(
                EndpointName=self.endpoint_name,
                ContentType="application/json",
                Accept="application/json",
                Body=body_bytes,
            )
            payload = json.load(resp["Body"])
        if self._effective_use_kv_cache:
            self._cached_model_id = payload.get("model_id") or self._cached_model_id
        return payload

    def _invoke_async(self, body_bytes: bytes) -> Dict[str, Any]:
        """Async Inference path: stage input through S3, invoke_endpoint_async,
        poll the OutputLocation S3 object. Required for payloads > 6 MB or
        compute > 60 s; the endpoint must be created with AsyncInferenceConfig.
        """
        if not self.s3_bucket:
            raise RuntimeError(
                "use_async=True requires `s3_bucket` (writable bucket the "
                "endpoint's execution role can read from).",
            )
        s3 = self._s3_client()
        inference_id = uuid.uuid4().hex
        input_key = f"{self.s3_prefix}/inputs/{inference_id}.json"
        s3.put_object(
            Bucket=self.s3_bucket,
            Key=input_key,
            Body=body_bytes,
            ContentType="application/json",
        )
        resp = self._runtime_client().invoke_endpoint_async(
            EndpointName=self.endpoint_name,
            InputLocation=f"s3://{self.s3_bucket}/{input_key}",
            ContentType="application/json",
            Accept="application/json",
            InferenceId=inference_id,
        )
        out_bucket, out_key = resp["OutputLocation"].removeprefix("s3://").split("/", 1)
        fail_location = resp.get("FailureLocation")
        fail_bucket, fail_key = (None, None)
        if fail_location:
            fail_bucket, fail_key = fail_location.removeprefix("s3://").split("/", 1)

        deadline = time.time() + self.async_timeout_s
        not_found = s3.exceptions.NoSuchKey
        while time.time() < deadline:
            try:
                obj = s3.get_object(Bucket=out_bucket, Key=out_key)
                return json.load(obj["Body"])
            except not_found:
                pass
            if fail_bucket is not None:
                try:
                    f_obj = s3.get_object(Bucket=fail_bucket, Key=fail_key)
                    detail = f_obj["Body"].read().decode("utf-8", errors="replace")
                    raise RuntimeError(
                        f"SageMaker async inference failed for id={inference_id}: {detail}",
                    )
                except not_found:
                    pass
            time.sleep(self.async_poll_interval_s)
        raise TimeoutError(
            f"SageMaker async inference timed out after {self.async_timeout_s}s "
            f"for id={inference_id}; output never appeared at {resp['OutputLocation']}",
        )


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
    """TabPFN regressor backed by a SageMaker endpoint."""

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
