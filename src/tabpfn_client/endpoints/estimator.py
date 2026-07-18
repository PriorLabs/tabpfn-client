#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""scikit-learn estimators for a TabPFN inference container reached over HTTP.

Mirrors the `tabpfn_client.TabPFNClassifier` / `TabPFNRegressor` surface;
each `predict*` call POSTs to the user-supplied `endpoint_url` (the full
scoring URL, including the `/predict` path). `fit()` does not call the
endpoint — it just stores `X` / `y` on the estimator. The training data
is shipped to the endpoint on the next `predict*` call, where the actual
fit runs.

Auth is optional: when `api_key` is set it is sent as `Bearer <api_key>`,
otherwise no `Authorization` header is added.

Cached predict is a per-call option: pass `model_id` to any `predict*`
call to send `context.model_id` and omit the training data, letting the
endpoint reuse an already-fit model. `fit()` is not required in that
case. When the endpoint returns a `model_id` in its response, it is
exposed as `self.model_id_` so callers can pass it forward. `model_path`
is similarly optional and only sent when set; some deployments reject
overrides.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import httpx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


def _to_jsonable(X: Any) -> list:
    """Coerce numpy / pandas inputs to plain Python lists for JSON."""
    if isinstance(X, pd.DataFrame):
        return X.values.tolist()
    if isinstance(X, pd.Series):
        return X.tolist()
    return np.asarray(X).tolist()


class _EndpointBase(BaseEstimator):
    """Shared HTTP plumbing for the endpoint-backed TabPFN estimators."""

    _TASK: str = ""  # overridden by subclasses

    def __init__(
        self,
        endpoint_url: str,
        api_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        model_path: Optional[str] = None,
        n_estimators: int = 8,
        softmax_temperature: float = 0.9,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        inference_precision: Literal["autocast", "auto"] = "auto",
        random_state: Optional[int] = 0,
        inference_config: Optional[Dict[str, Any]] = None,
        n_preprocessing_jobs: int = 4,
        memory_saving_mode: Optional[bool | Literal["auto"]] = None,
        categorical_features_indices: Optional[List[int]] = None,
        timeout_s: float = 300.0,
    ):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.extra_headers = extra_headers
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.inference_precision = inference_precision
        self.random_state = random_state
        self.inference_config = inference_config
        self.n_preprocessing_jobs = n_preprocessing_jobs
        self.memory_saving_mode = memory_saving_mode
        self.categorical_features_indices = categorical_features_indices
        self.timeout_s = timeout_s

    def _build_tabpfn_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "softmax_temperature": self.softmax_temperature,
            "average_before_softmax": self.average_before_softmax,
            "inference_precision": self.inference_precision,
            "random_state": self.random_state,
            "inference_config": self.inference_config,
            "n_preprocessing_jobs": self.n_preprocessing_jobs,
        }
        if self.memory_saving_mode is not None:
            cfg["memory_saving_mode"] = self.memory_saving_mode
        if self.categorical_features_indices is not None:
            cfg["categorical_features_indices"] = self.categorical_features_indices
        if self.model_path is not None:
            cfg["model_path"] = self.model_path
        if self._TASK == "classification":
            cfg["balance_probabilities"] = self.balance_probabilities
        return cfg

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        # The API key is optional because some deployments do not require it
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.extra_headers:
            headers.update(self.extra_headers)
        return headers

    def _http_client(self) -> httpx.Client:
        # Cache the httpx.Client so repeated predict* calls reuse the TCP /
        # TLS connection (keep-alive) instead of redoing the handshake on
        # every request.
        client = getattr(self, "_cached_client", None)
        if client is not None:
            return client
        client = httpx.Client(timeout=self.timeout_s)
        self._cached_client = client
        return client

    def __getstate__(self) -> Dict[str, Any]:
        # httpx.Client isn't pickleable; strip the cache for sklearn pickling.
        state = self.__dict__.copy()
        state.pop("_cached_client", None)
        return state

    def fit(self, X: Any, y: Any) -> "_EndpointBase":
        X_arr = X if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples; "
                f"got X={X_arr.shape}, y={y_arr.shape}"
            )
        self.X_train_ = X_arr
        self.y_train_ = y_arr
        if self._TASK == "classification":
            self.classes_ = np.unique(y_arr)
        return self

    def _invoke(
        self,
        X_test: Any,
        output_type: str,
        predict_params: Optional[Dict[str, Any]] = None,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
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

        # Allow passing model_id to reference a previously fitted
        # model and run predict by skipping the ICL step
        if model_id is not None:
            body["context"] = {"model_id": model_id}
        else:
            check_is_fitted(self, ["X_train_", "y_train_"])
            # y_train on the wire is 2D (n_samples, 1).
            y_arr = np.asarray(self.y_train_)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            body["X_train"] = _to_jsonable(self.X_train_)
            body["y_train"] = y_arr.tolist()

        resp = self._http_client().post(
            self.endpoint_url,
            json=body,
            headers=self._headers(),
        )
        resp.raise_for_status()
        payload = resp.json()

        returned_id = payload.get("model_id")
        if returned_id is not None:
            self.model_id_ = returned_id

        return payload


class TabPFNClassifier(_EndpointBase, ClassifierMixin):
    """TabPFN classifier backed by a self-hosted inference endpoint.

    Example:
        from tabpfn_client.endpoints import TabPFNClassifier
        clf = TabPFNClassifier(
            endpoint_url="https://<your-endpoint>/predict",
            api_key="<optional-bearer-token>",
        )
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        clf.predict_proba(X_test)
    """

    _TASK = "classification"

    def predict(self, X: Any, model_id: Optional[str] = None) -> np.ndarray:
        result = self._invoke(X, output_type="preds", model_id=model_id)
        return np.asarray(result["prediction"])

    def predict_proba(self, X: Any, model_id: Optional[str] = None) -> np.ndarray:
        result = self._invoke(X, output_type="probas", model_id=model_id)
        return np.asarray(result["prediction"])


class TabPFNRegressor(_EndpointBase, RegressorMixin):
    """TabPFN regressor backed by a self-hosted inference endpoint.

    Example:
        from tabpfn_client.endpoints import TabPFNRegressor
        reg = TabPFNRegressor(
            endpoint_url="https://<your-endpoint>/predict",
            api_key="<optional-bearer-token>",
        )
        reg.fit(X_train, y_train)
        reg.predict(X_test)
        reg.predict(X_test, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
    """

    _TASK = "regression"

    def predict(
        self,
        X: Any,
        output_type: str = "mean",
        quantiles: Optional[list] = None,
        model_id: Optional[str] = None,
    ) -> np.ndarray:
        predict_params: Dict[str, Any] = {}
        if quantiles is not None:
            predict_params["quantiles"] = quantiles
        result = self._invoke(
            X,
            output_type=output_type,
            predict_params=predict_params,
            model_id=model_id,
        )
        return np.asarray(result["prediction"])
