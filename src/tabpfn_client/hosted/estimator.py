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

Cached predict is opt-in via the `model_id` constructor argument.
When set (and `fit()` has not been called), `predict*` sends
`context.model_id` and omits the training data, letting the endpoint
reuse an already-fit model. Calling `fit()` on the estimator supersedes
the constructor `model_id` — the freshly-stored training data is
shipped on the next `predict*`, matching sklearn's re-fit semantics.
To make the endpoint build a reusable cache in the first
place, set `fit_mode="fit_with_cache"`; the endpoint then returns a
`model_id` in its response, captured on the fitted attribute
`self.model_id_` so callers can construct a follow-up estimator with
`model_id=prior.model_id_`. `model_path` is similarly optional and
only sent when set; some deployments reject overrides.

`use_kv_cache=True` automates that round trip within one estimator: the
first `predict*` after `fit()` ships the training data as usual, and
every later one reuses the `model_id` the endpoint returned. Workloads
that predict many times against a single fit — SHAP, permutation
importance, partial dependence — otherwise pay a full re-fit per call.
The endpoint's cache is bounded, so a `model_id` can be evicted between
predicts; that surfaces as an error and the estimator has to be re-fit.

`model_id` lives on the constructor (not on `predict`) so the sklearn
estimator contract stays intact: `predict(X)` / `predict_proba(X)` work
with `Pipeline`, `GridSearchCV`, `cross_validate`, etc.

`payload_format="parquet"` sends the datasets as multipart Parquet files
instead of inline JSON. Encoding dominates request time on large tables,
and Parquet carries missing and non-finite values exactly. The JSON path
encodes them as the `NaN` / `Infinity` literals Python's `json` accepts
but the standard does not, so a strict endpoint needs Parquet for those.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Literal, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


# Multipart form field name -> JSON body key for the same dataset.
_JSON_DATASET_KEYS = {"x_train": "X_train", "y_train": "y_train", "x_test": "X_test"}

_PARQUET_MIME = "application/vnd.apache.parquet"


def _to_jsonable(X: Any) -> list:
    """Coerce numpy / pandas inputs to plain Python lists for JSON."""
    if isinstance(X, pd.DataFrame):
        return X.values.tolist()
    if isinstance(X, pd.Series):
        return X.tolist()
    if isinstance(X, list):
        return X
    return np.asarray(X).tolist()


def _to_parquet_part(X: Any, name: str) -> Tuple[str, bytes, str]:
    """Serialize one dataset to a multipart Parquet file part."""
    frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False, compression="zstd")
    return (f"{name}.parquet", buf.getvalue(), _PARQUET_MIME)


class _HostedBase(BaseEstimator):
    """Shared HTTP plumbing for the endpoint-backed TabPFN estimators."""

    _TASK: str = ""  # overridden by subclasses

    def __init__(
        self,
        endpoint_url: str,
        api_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        fit_mode: Optional[
            Literal["fit_preprocessors", "low_memory", "fit_with_cache", "batched"]
        ] = None,
        n_estimators: int | None = None,
        softmax_temperature: float | None = None,
        balance_probabilities: bool = False,
        average_before_softmax: bool | None = None,
        inference_precision: Literal["autocast", "auto"] | None = None,
        random_state: int | None = 0,
        inference_config: Dict[str, Any] | None = None,
        n_preprocessing_jobs: int = 4,
        memory_saving_mode: bool | Literal["auto"] | None = None,
        categorical_features_indices: List[int] | None = None,
        timeout_s: float = 300.0,
        payload_format: Literal["json", "parquet"] = "json",
        use_kv_cache: bool = False,
    ):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.extra_headers = extra_headers
        self.model_id = model_id
        self.model_path = model_path
        self.fit_mode = fit_mode
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
        self.payload_format = payload_format
        self.use_kv_cache = use_kv_cache

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
        # The endpoint hands back a reusable `model_id` only when asked to keep
        # the cache, so `use_kv_cache` implies that fit_mode; an explicit one
        # still wins. Resolved here to leave the constructor argument untouched.
        fit_mode = self.fit_mode
        if fit_mode is None and self.use_kv_cache:
            fit_mode = "fit_with_cache"
        if fit_mode is not None:
            cfg["fit_mode"] = fit_mode
        if self._TASK == "classification":
            cfg["balance_probabilities"] = self.balance_probabilities
        return cfg

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if self.payload_format == "json":
            headers["Content-Type"] = "application/json"
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

    def fit(self, X: Any, y: Any) -> "_HostedBase":
        X_arr = X if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y if isinstance(y, (pd.DataFrame, pd.Series)) else np.asarray(y)
        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples; "
                f"got X={X_arr.shape}, y={y_arr.shape}"
            )
        self.X_train_ = X_arr
        self.y_train_ = y_arr
        # New training data invalidates any cached model_id captured from a
        # prior predict; otherwise `predict(X, model_id=self.model_id_)` would
        # keep hitting the old server-side model.
        self.model_id_ = None
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
        params: Dict[str, Any] = {"output_type": output_type}
        if predict_params:
            params.update(predict_params)

        request: Dict[str, Any] = {
            "task_config": {
                "task": self._TASK,
                "tabpfn_config": self._build_tabpfn_config(),
                "predict_params": params,
            }
        }
        datasets: Dict[str, Any] = {"x_test": X_test}

        # Precedence: a completed fit() wins over the constructor's model_id.
        # Rationale — if the caller ran fit(new_X, new_y) on an estimator that
        # was constructed with model_id set, they have re-trained; the cached
        # id is now stale relative to the new data. Falls back to the cached
        # path only when fit() was never called.
        has_training_data = hasattr(self, "X_train_") and hasattr(self, "y_train_")
        # An earlier predict against this fit already left a model on the
        # endpoint; reuse it rather than re-sending data it still holds.
        cached_id = (
            getattr(self, "_cached_model_id", None) if self.use_kv_cache else None
        )
        if cached_id is not None:
            request["context"] = {"model_id": cached_id}
        elif has_training_data:
            # y_train on the wire is 2D (n_samples, 1).
            y_arr = np.asarray(self.y_train_)
            if y_arr.ndim == 1:
                y_arr = y_arr.reshape(-1, 1)
            datasets["x_train"] = self.X_train_
            datasets["y_train"] = y_arr
        elif self.model_id is not None:
            request["context"] = {"model_id": self.model_id}
        else:
            # Neither path available — raise the standard sklearn error.
            check_is_fitted(self, ["X_train_", "y_train_"])

        resp = self._post(request, datasets)
        resp.raise_for_status()
        payload = resp.json()

        returned_id = payload.get("model_id")
        if returned_id is not None:
            self.model_id_ = returned_id
            if self.use_kv_cache:
                self._cached_model_id = returned_id

        return payload

    def _post(
        self, request: Dict[str, Any], datasets: Dict[str, Any]
    ) -> httpx.Response:
        """POST the request, with the datasets inline or as Parquet files."""
        if self.payload_format == "parquet":
            return self._http_client().post(
                self.endpoint_url,
                data={"request": json.dumps(request)},
                files={
                    name: _to_parquet_part(value, name)
                    for name, value in datasets.items()
                },
                headers=self._headers(),
            )
        body = dict(request)
        for name, value in datasets.items():
            body[_JSON_DATASET_KEYS[name]] = _to_jsonable(value)
        # Not httpx's `json=`, which rejects non-finite floats outright;
        # `json.dumps` writes the literals Python's own decoder reads back.
        return self._http_client().post(
            self.endpoint_url,
            content=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
        )


class TabPFNClassifier(ClassifierMixin, _HostedBase):
    """TabPFN classifier backed by a self-hosted inference endpoint.

    Example:
        from tabpfn_client.hosted import TabPFNClassifier
        clf = TabPFNClassifier(
            endpoint_url="https://<your-endpoint>/predict",
            api_key="<optional-bearer-token>",
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


class TabPFNRegressor(RegressorMixin, _HostedBase):
    """TabPFN regressor backed by a self-hosted inference endpoint.

    Example:
        from tabpfn_client.hosted import TabPFNRegressor
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
    ) -> np.ndarray:
        predict_params: Dict[str, Any] = {}
        if quantiles is not None:
            predict_params["quantiles"] = quantiles
        result = self._invoke(
            X,
            output_type=output_type,
            predict_params=predict_params,
        )
        return np.asarray(result["prediction"])
