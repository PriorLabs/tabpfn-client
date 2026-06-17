#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""scikit-learn estimators for the TabPFN Azure AI Foundry endpoint.

Mirrors the `tabpfn_client.TabPFNClassifier` / `TabPFNRegressor` surface;
each `predict*` call POSTs to the user-supplied `endpoint_url` (the full
scoring URL, including the `/predict` path) with a Bearer token. `fit()`
does not call the endpoint — it just stores `X` / `y` on the estimator.
The training data is shipped to the endpoint on the next `predict*`
call, where the actual fit runs.

This client sends requests as `application/json` only (Foundry also
accepts `multipart/form-data`, but we don't use it here) and does not
currently support thinking mode.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

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


def _build_request_body(
    task: str,
    tabpfn_config: Dict[str, Any],
    predict_params: Dict[str, Any],
    X_test: Any,
    X_train: Optional[Any] = None,
    y_train: Optional[Any] = None,
    cached_model_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble a Foundry `/predict` JSON body.

    When `cached_model_id` is provided, the body targets the V3 cache-hit
    path: `X_train` / `y_train` are omitted and a `context.model_id` is
    sent instead. Otherwise training data is shipped inline.
    """
    body: Dict[str, Any] = {
        "task_config": {
            "task": task,
            "tabpfn_config": tabpfn_config,
            "predict_params": predict_params,
        },
        "x_test": _to_jsonable(X_test),
    }

    # Build up the KV-cache context if we have a model_id,
    # otherwise ship the training data
    if cached_model_id is not None:
        body["context"] = {"model_id": cached_model_id}
    else:
        body["x_train"] = _to_jsonable(X_train)

        # y_train on the wire is 2D (n_samples, 1)
        y_arr = np.asarray(y_train)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        body["y_train"] = y_arr.tolist()

    return body


class _FoundryBase(BaseEstimator):
    """Shared HTTP plumbing for the Foundry TabPFN estimators."""

    def __init__(
        self,
        endpoint_url: str,
        api_key: str,
        task: str = "classification",
        n_estimators: int = 8,
        softmax_temperature: float = 0.9,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        ignore_pretraining_limits: bool = True,
        inference_precision: Literal["autocast", "auto"] = "auto",
        random_state: Optional[int] = 0,
        inference_config: Optional[Dict[str, Any]] = None,
        paper_version: bool = False,
        use_kv_cache: bool = False,
        timeout_s: float = 300.0,
    ):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self._task = task
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision = inference_precision
        self.random_state = random_state
        self.inference_config = inference_config
        self.paper_version = paper_version
        self.use_kv_cache = use_kv_cache
        self.timeout_s = timeout_s

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

        if self._task == "classification":
            cfg["balance_probabilities"] = self.balance_probabilities

        return cfg

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

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

    def fit(self, X: Any, y: Any) -> "_FoundryBase":
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
        if self._task == "classification":
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
        body = _build_request_body(
            task=self._task,
            tabpfn_config=self._build_tabpfn_config(),
            predict_params=params,
            X_test=X_test,
            X_train=self.X_train_,
            y_train=self.y_train_,
            cached_model_id=self._cached_model_id if self.use_kv_cache else None,
        )
        resp = self._http_client().post(
            self.endpoint_url,
            json=body,
            headers=self._headers(),
        )
        resp.raise_for_status()
        payload = resp.json()
        if self.use_kv_cache:
            self._cached_model_id = payload.get("model_id") or self._cached_model_id
        return payload


class TabPFNClassifier(_FoundryBase, ClassifierMixin):
    """TabPFN classifier backed by an Azure AI Foundry endpoint.

    Example:
        from tabpfn_client.foundry import TabPFNClassifier
        clf = TabPFNClassifier(
            endpoint_url="https://<your-endpoint>.<region>.inference.ml.azure.com/predict",
            api_key="<your-foundry-bearer-token>",
        )
        clf.fit(X_train, y_train)
        clf.predict(X_test)
        clf.predict_proba(X_test)
    """

    def __init__(self, *args: Any, task: str = "classification", **kwargs: Any):
        super().__init__(*args, task=task, **kwargs)

    def predict(self, X: Any) -> np.ndarray:
        result = self._invoke(X, output_type="preds")
        return np.asarray(result["prediction"])

    def predict_proba(self, X: Any) -> np.ndarray:
        result = self._invoke(X, output_type="probas")
        return np.asarray(result["prediction"])


class TabPFNRegressor(_FoundryBase, RegressorMixin):
    """TabPFN regressor backed by an Azure AI Foundry endpoint.

    Example:
        from tabpfn_client.foundry import TabPFNRegressor
        reg = TabPFNRegressor(
            endpoint_url="https://<your-endpoint>.<region>.inference.ml.azure.com/predict",
            api_key="<your-foundry-bearer-token>",
        )
        reg.fit(X_train, y_train)
        reg.predict(X_test)
        reg.predict(X_test, output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
    """

    def __init__(self, *args: Any, task: str = "regression", **kwargs: Any):
        super().__init__(*args, task=task, **kwargs)

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
