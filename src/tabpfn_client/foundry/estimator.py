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
accepts `multipart/form-data`, but we don't use it here).
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional, cast

import httpx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from tabpfn_client.models import FitModeLiteral


ThinkingEffort = Literal["medium", "high"]

# The endpoint validates these server-side and answers an out-of-range value
# with an opaque HTTP 424, so we mirror the accepted sets here to fail fast
# with an actionable message instead.
_VALID_THINKING_EFFORTS: tuple = ("medium", "high")
_VALID_THINKING_METRICS: Dict[str, tuple] = {
    "classification": ("accuracy", "log_loss", "balanced_accuracy"),
    "regression": ("rmse", "mse", "mae", "r2"),
}


class FoundryEndpointError(httpx.HTTPStatusError):
    """A non-2xx answer from the Foundry endpoint, with its error payload.

    Subclasses `httpx.HTTPStatusError`, so callers that already catch that
    keep working. The endpoint reports failures as a JSON body carrying
    `error_code` and `trace_id`; both are surfaced on the exception (and in
    its message) because the `trace_id` is what the endpoint provider needs
    in order to look the failure up server-side.
    """

    def __init__(
        self,
        message: str,
        *,
        request: Any,
        response: Any,
        error_code: Optional[str] = None,
        trace_id: Optional[str] = None,
    ):
        super().__init__(message, request=request, response=response)
        self.error_code = error_code
        self.trace_id = trace_id


def _raise_for_status(resp: httpx.Response) -> None:
    """Raise `FoundryEndpointError` carrying the endpoint's error payload."""
    if not resp.is_error:
        return

    detail, error_code, trace_id = "", None, None
    try:
        payload = resp.json()
    except Exception:
        detail = resp.text[:500]
    else:
        if isinstance(payload, dict):
            error_code = payload.get("error_code")
            trace_id = payload.get("trace_id")
            detail = payload.get("message") or ""
        else:
            detail = str(payload)[:500]

    message = f"Foundry endpoint returned HTTP {resp.status_code}"
    if error_code:
        message += f" ({error_code})"
    if detail:
        message += f": {detail}"
    if trace_id:
        message += f" [trace_id={trace_id}]"

    raise FoundryEndpointError(
        message,
        request=resp.request,
        response=resp,
        error_code=error_code,
        trace_id=trace_id,
    )


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
    thinking_block: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble a Foundry `/predict` JSON body.

    When `cached_model_id` is provided, the body targets the V3 cache-hit
    path: `X_train` / `y_train` are omitted and a `context.model_id` is
    sent instead. Otherwise training data is shipped inline.

    When `thinking_block` is provided (non-empty), its fields are merged
    at the top level of the request body (e.g. `thinking_effort`,
    `thinking_timeout_s`, `thinking_metric`).
    """
    body: Dict[str, Any] = {
        "task_config": {
            "task": task,
            "tabpfn_config": tabpfn_config,
            "predict_params": predict_params,
        },
        "x_test": _to_jsonable(X_test),
    }
    if thinking_block:
        body.update(thinking_block)

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
        thinking_mode: bool = False,
        thinking_effort: Optional[ThinkingEffort] = None,
        thinking_timeout_s: Optional[float] = None,
        thinking_metric: Optional[str] = None,
        use_kv_cache: bool = False,
        fit_mode: Optional[FitModeLiteral] = None,
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
        self.thinking_mode = thinking_mode
        self.thinking_effort = thinking_effort
        self.thinking_timeout_s = thinking_timeout_s
        self.thinking_metric = thinking_metric
        self.use_kv_cache = use_kv_cache
        self.fit_mode = fit_mode
        self.timeout_s = timeout_s
        self._validate_args()

    def _validate_args(self) -> None:
        """Reject settings that would silently disable caching for a
        configuration that plainly asked for it."""
        if self.fit_mode == "fit_preprocessors":
            if self.use_kv_cache:
                raise ValueError(
                    "Conflicting settings: fit_mode='fit_preprocessors' "
                    "cannot be combined with use_kv_cache=True. Either drop "
                    "use_kv_cache or set fit_mode='fit_with_cache'."
                )
            if self.thinking_mode or self.thinking_effort is not None:
                raise ValueError(
                    "Conflicting settings: fit_mode='fit_preprocessors' "
                    "cannot be combined with thinking mode (thinking_mode=True "
                    "or thinking_effort set). Thinking mode requires caching; "
                    "either drop the thinking-mode params or set "
                    "fit_mode='fit_with_cache' (or leave fit_mode unset)."
                )

        if (
            self.thinking_effort is not None
            and self.thinking_effort not in _VALID_THINKING_EFFORTS
        ):
            raise ValueError(
                f"thinking_effort must be one of "
                f"{list(_VALID_THINKING_EFFORTS)}; got {self.thinking_effort!r}."
            )

        if self.thinking_metric is not None:
            allowed = _VALID_THINKING_METRICS.get(self._task, ())
            if self.thinking_metric not in allowed:
                raise ValueError(
                    f"thinking_metric={self.thinking_metric!r} is not supported "
                    f"for task={self._task!r}; expected one of {list(allowed)}."
                )

        if self.thinking_timeout_s is not None and self.thinking_timeout_s < 0:
            raise ValueError(
                f"thinking_timeout_s must be >= 0 (0 means no client-requested "
                f"limit); got {self.thinking_timeout_s!r}."
            )

    @property
    def _thinking_active(self) -> bool:
        return self.thinking_mode or self.thinking_effort is not None

    @property
    def _effective_fit_mode(self) -> FitModeLiteral:
        # Explicit `fit_mode` wins; `_validate_args` guarantees it doesn't
        # conflict with `use_kv_cache` / thinking mode.
        if self.fit_mode is not None:
            return cast(FitModeLiteral, self.fit_mode)
        if self.use_kv_cache or self._thinking_active:
            return "fit_with_cache"
        return "fit_preprocessors"

    @property
    def _cache_active(self) -> bool:
        return self._effective_fit_mode == "fit_with_cache"

    def _build_tabpfn_config(self) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "softmax_temperature": self.softmax_temperature,
            "average_before_softmax": self.average_before_softmax,
            "ignore_pretraining_limits": self.ignore_pretraining_limits,
            "inference_precision": self.inference_precision,
            "random_state": self.random_state,
            "inference_config": self.inference_config,
            "fit_mode": self._effective_fit_mode,
        }

        if self._task == "classification":
            cfg["balance_probabilities"] = self.balance_probabilities

        return cfg

    def _build_thinking_block(self) -> Dict[str, Any]:
        """Top-level wire fields for thinking-mode. Empty when inactive.

        The endpoint only accepts these keys at the top level of the request
        body — nesting them under `task_config` / `tabpfn_config` /
        `predict_params` is rejected the same way an unknown field is.

        Note that `thinking_timeout_s` is a budget the server must be able to
        meet: a value too small for the dataset makes the request fail rather
        than return a cheaper answer, so leave it unset (or 0) unless you
        specifically need to bound the call.
        """
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
            cached_model_id=self._cached_model_id if self._cache_active else None,
            thinking_block=self._build_thinking_block(),
        )
        resp = self._http_client().post(
            self.endpoint_url,
            json=body,
            headers=self._headers(),
        )
        _raise_for_status(resp)
        payload = resp.json()
        if self._cache_active:
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
