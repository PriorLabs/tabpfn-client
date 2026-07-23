#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Literal, cast, overload
from typing_extensions import Self
from uuid import UUID

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from tabpfn_client.client import (
    ServiceClient,
    ClientOptions,
    PredictionResult,
    NeedsRefittingError,
    ThinkingEffort,
)
from tabpfn_client.config import Config, init
from tabpfn_client.constants import URL_TABPFN_EXTENSIONS_GITHUB_MANY_CLASS_CODE
from tabpfn_client.api_models import ModelVersion
from tabpfn_client.utils import model_limit_from_version, model_version_from_path
from tabpfn_client.service_wrapper import InferenceClient
from tabpfn_client.api_models import (
    FitMode,
    RegressorTabPFNConfig,
    ClassifierTabPFNConfig,
    RegressorPredictParams,
    ClassifierPredictParams,
    ClassifierConfig,
    RegressorConfig,
)
from tabpfn_client.options import get_opts

try:
    from torch import Tensor  # type: ignore
except ImportError:
    Tensor = None

TORCH_AVAILABLE = Tensor is not None

logger = logging.getLogger(__name__)

# Special strings used to identify model families in model paths.
V_2_5_IDENTIFIER = "v2.5"
V_2_6_IDENTIFIER = "v2.6"
V_3_IDENTIFIER = "v3"

DEFAULT_V2_MODEL_PATH = "v2_default"
DEFAULT_V2_5_MODEL_PATH = "v2.5_default"
DEFAULT_V2_6_MODEL_PATH = "v2.6_default"
DEFAULT_V3_MODEL_PATH = "v3_default"

# Sentinel values for `model_path` that defer model selection to the server.
# `None` means the caller didn't pick a model; "auto" is the canonical name
# (matches the OSS tabpfn package); "default" is a backward-compatible alias.
_AUTO_MODEL_PATH_ALIASES: frozenset[str | None] = frozenset({None, "auto", "default"})

THINKING_TIMEOUT_MAX_S = 40 * 60

_VALID_THINKING_EFFORT_LEVELS = frozenset({"medium", "high"})

# The server's `FitMode` enum also carries internal `low_memory`/`batched`
# values, but gapi only wires two end-to-end: `fit_preprocessors` (the stateless
# default) and `fit_with_cache` (persist a server-side KV cache keyed by the
# returned fitted-train-set id, so later predicts against that id are served
# from the cache instead of re-fitting). The value is validated server-side by
# `FitRequest._validate_fit_mode`; the client just forwards it.

# Bumped when the shape of the dict produced by `save_model()` changes.
_MODEL_HANDLE_VERSION = 1


def _cast_fitted_id(model_id: str | UUID | None) -> UUID | None:
    """Normalize a user-supplied model id into a UUID (or None)."""
    if model_id is None:
        return None
    if isinstance(model_id, UUID):
        return model_id
    return UUID(str(model_id))


def _resolve_train_set_id(estimator: Any) -> UUID | None:
    """Resolve the server-side fitted-train-set id an estimator should predict against.

    A real `fit()` sets `fitted_train_set_id_` and always wins; otherwise fall
    back to the constructor's `model_id` (the load-by-id path). Keeping this as
    a resolver instead of syncing state at both mutation sites means `fit()`
    never has to mutate `self.model_id`, so `get_params()` / `clone()` stay
    sklearn-correct.

    Returns None if `model_id` is set but not a valid UUID — sklearn's
    convention is that `__init__` does no validation, so bad ids must not
    crash `__sklearn_is_fitted__`. The estimator degrades to "not fitted"
    and `predict` surfaces the standard `NotFittedError`.
    """
    fitted = getattr(estimator, "fitted_train_set_id_", None)
    if fitted is not None:
        return fitted
    try:
        return _cast_fitted_id(estimator.model_id)
    except (ValueError, TypeError):
        return None


def _read_model_handle(handle: dict[str, Any] | str | Path) -> dict[str, Any]:
    """Accept either an in-memory handle dict or a path to a JSON file
    produced by `save_model()`."""
    if isinstance(handle, dict):
        return handle
    return json.loads(Path(handle).read_text())


def _build_model_handle(
    estimator: Any,
    task: Literal["classification", "regression"],
    path: str | Path | None,
) -> dict[str, Any] | Path:
    """Shared implementation of `save_model()` for both estimators."""
    check_is_fitted(estimator)
    params = estimator.get_params(deep=False)
    # `client_options` is runtime-only (timeouts, auth/trace headers) and not
    # JSON-serializable; drop it so the handle stays portable.
    params.pop("client_options", None)
    # Persist the *effective* id: a real `fit()` supersedes the constructor's
    # `model_id`. Coerce to str for JSON.
    params["model_id"] = str(_resolve_train_set_id(estimator))
    handle: dict[str, Any] = {
        "format_version": _MODEL_HANDLE_VERSION,
        "task": task,
        "params": params,
    }
    if task == "classification":
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            # A bare `model_id` classifier (never fit / never load_model'd) has
            # no class labels to persist; fail clearly instead of AttributeError.
            raise ValueError(
                "Cannot save a classifier without `classes_`. Fit it first, or "
                "reconstruct it via `load_model()`."
            )
        handle["classes"] = classes.tolist()
    if path is None:
        return handle
    out = Path(path)
    out.write_text(json.dumps(handle, indent=2))
    return out


def _load_model_handle_for(
    cls: type,
    handle: dict[str, Any] | str | Path,
    task: Literal["classification", "regression"],
) -> dict[str, Any]:
    """Read + validate a `save_model()` handle for the target estimator class."""
    data = _read_model_handle(handle)
    version = data.get("format_version")
    if version != _MODEL_HANDLE_VERSION:
        raise ValueError(
            f"Unsupported model handle version {version!r}; "
            f"expected {_MODEL_HANDLE_VERSION}."
        )
    if data.get("task") != task:
        raise ValueError(
            f"Cannot load a {data.get('task')!r} model into {cls.__name__}."
        )
    return data


class TabPFNModelSelection:
    """Base class for TabPFN model selection and path handling."""

    _AVAILABLE_MODELS: list[str] = []
    _VALID_TASKS = {"classification", "regression"}

    @classmethod
    def list_available_models(cls) -> list[str]:
        return cls._AVAILABLE_MODELS

    @classmethod
    def _validate_model_name(cls, model_name: str | None) -> None:
        # `None` (defer to the server) is one of the auto aliases, so it counts
        # as valid rather than an unknown model name.
        if (
            model_name not in _AUTO_MODEL_PATH_ALIASES
            and model_name not in cls._AVAILABLE_MODELS
        ):
            raise ValueError(
                f"Invalid model name: {model_name}. "
                f"Available models are: {cls.list_available_models()}"
            )

    @classmethod
    def _model_name_to_path(
        cls, task: Literal["classification", "regression"], model_name: str | None
    ) -> str | None:
        cls._validate_model_name(model_name)
        model_name_task = "classifier" if task == "classification" else "regressor"
        # Let the server pick the default model when the caller defers to us.
        if model_name in _AUTO_MODEL_PATH_ALIASES:
            return None
        # `None` is one of the auto aliases handled above, so the remainder is a
        # concrete model name; assert it to narrow `str | None` -> `str`.
        assert model_name is not None
        if V_3_IDENTIFIER in model_name:
            return f"tabpfn-{V_3_IDENTIFIER}-{model_name_task}-{model_name}.ckpt"
        if V_2_6_IDENTIFIER in model_name:
            return f"tabpfn-{V_2_6_IDENTIFIER}-{model_name_task}-{model_name}.ckpt"
        if V_2_5_IDENTIFIER in model_name:
            return f"tabpfn-{V_2_5_IDENTIFIER}-{model_name_task}-{model_name}.ckpt"
        return f"tabpfn-v2-{model_name_task}-{model_name}.ckpt"

    @classmethod
    def create_default_for_version(cls, version: ModelVersion, **overrides) -> Self:
        """Construct an estimator that uses the given version of the model.

        In addition to selecting the model, this also configures the estimator with
        certain default settings associated with this model version.

        Any kwargs will override the default settings.
        """
        options: dict[str, Any] = {
            "n_estimators": 8,
            "softmax_temperature": 0.9,
        }
        if version == ModelVersion.V2:
            options["model_path"] = DEFAULT_V2_MODEL_PATH
        elif version == ModelVersion.V2_5:
            options["model_path"] = DEFAULT_V2_5_MODEL_PATH
        elif version == ModelVersion.V2_6:
            options["model_path"] = DEFAULT_V2_6_MODEL_PATH
        elif version == ModelVersion.V3:
            options["model_path"] = DEFAULT_V3_MODEL_PATH
        else:
            raise ValueError(f"Unknown version: {version}")

        options.update(overrides)

        return cls(**options)


class TabPFNClassifier(ClassifierMixin, BaseEstimator, TabPFNModelSelection):
    _AVAILABLE_MODELS = [
        DEFAULT_V3_MODEL_PATH,
        DEFAULT_V2_6_MODEL_PATH,
        "v2.5_default-2",
        DEFAULT_V2_5_MODEL_PATH,
        "v2.5_large-features-L",
        "v2.5_large-features-XL",
        "v2.5_large-samples",
        "v2.5_real-large-features",
        "v2.5_real-large-samples-and-features",
        "v2.5_real",
        "v2.5_variant",
        DEFAULT_V2_MODEL_PATH,
        "auto",
        # Deprecated alias for "auto"; kept for backward compat with users and
        # downstream packages (e.g. tabpfn-time-series) that read this list.
        "default",
        "gn2p4bpt",
        "llderlii",
        "od3j1g5m",
        "vutqq28w",
        "znskzxi4",
    ]

    def __init__(
        self,
        # start: tabpfn_config
        model_path: str | None = None,
        n_estimators: int | None = None,
        softmax_temperature: float | None = None,
        balance_probabilities: bool = False,
        average_before_softmax: bool | None = None,
        ignore_pretraining_limits: bool = False,
        inference_precision: Literal["autocast", "auto"] | None = None,
        random_state: int | None = 0,
        inference_config: dict[str, Any] | None = None,
        categorical_features_indices: list[int] | None = None,
        fit_mode: FitMode | None = None,
        # end: tabpfn_config
        paper_version: bool = False,
        thinking_mode: bool = False,
        thinking_effort: ThinkingEffort | None = None,
        thinking_timeout_s: float | None = None,
        thinking_metric: str | None = None,
        client_options: ClientOptions | None = None,
        model_id: str | UUID | None = None,
    ):
        """Construct a TabPFN classifier.

        This constructs a classifier using the latest model and settings. If you would
        like to use a previous model version, use `create_default_for_version()`
        instead. You can also use `model_path` to specify a particular model

        Parameters
        ----------
        model_path: str, default="auto"
            The name of the model to use. "auto" lets the server pick the
            latest default model; "default" is accepted as a backward-compatible
            alias. Use `create_default_for_version()` to pin to a specific
            major version.
        n_estimators: int, default=8
            The number of estimators in the TabPFN ensemble. We aggregate the
             predictions of `n_estimators`-many forward passes of TabPFN. Each forward
             pass has (slightly) different input data. Think of this as an ensemble of
             `n_estimators`-many "prompts" of the input data.
        softmax_temperature: float, default=0.9
            The temperature for the softmax function. This is used to control the
            confidence of the model's predictions. Lower values make the model's
            predictions more confident. This is only applied when predicting during a
            post-processing step. Set `softmax_temperature=1.0` for no effect.
        balance_probabilities: bool, default=False
            Whether to balance the probabilities based on the class distribution
            in the training data. This can help to improve predictive performance
            when the classes are highly imbalanced. This is only applied when predicting
            during a post-processing step.
        average_before_softmax: bool, default=False
             Only used if `n_estimators > 1`. Whether to average the predictions of the
             estimators before applying the softmax function. This can help to improve
             predictive performance when there are many classes or when calibrating the
             model's confidence. This is only applied when predicting during a
             post-processing.
        ignore_pretraining_limits: bool, default=True
            Whether to ignore the pre-training limits of the model. The TabPFN models
            have been pre-trained on a specific range of input data. If the input data
            is outside of this range, the model may not perform well. You may ignore
            our limits to use the model on data outside the pre-training range.
            Defaults to True (vs False in the OSS package): the server enforces its
            own capacity limits, so the OSS check is redundant and stricter.
        inference_precision: "autocast" or "auto", default="auto"
            The precision to use for inference. This can dramatically affect the
            speed and reproducibility of the inference.
        random_state: int or RandomState or RandomGenerator or None, default=0
            Controls the randomness of the model. Pass an int for reproducible
            results; pass `None` to use a fresh random seed each run.
        inference_config: dict or None, default=None
            Additional advanced arguments for model interface. See the doc of InferenceConfig
            in the tabpfn package for more details. For the client, the inference_config and the
            preprocess transforms need to be dictionaries.
        categorical_features_indices: list[int] or None, default=None
            The indices of the columns that should be treated as categorical.
            If None, the model infers which columns are categorical.
        paper_version: bool, default=False
            If True, will use the model described in the paper, instead of the newest
            version available on the API, which e.g handles text features better.
        thinking_mode: bool, default=False
            If True, spend extra fit-time compute for higher precision.
            Equivalent to passing `thinking_effort="medium"` — setting any
            `thinking_effort` value also enables thinking, so this flag is
            optional when you've set the level explicitly.
        thinking_effort: {"medium", "high"} or None, default=None
            Effort level for thinking mode. When set, thinking is enabled
            (you don't also need `thinking_mode=True`). When None and
            `thinking_mode=True`, defaults to "medium".
        thinking_timeout_s: float or None, default=None
            Budget for the fit, in seconds. Only consulted when thinking is
            enabled. Capped at 2400.
        thinking_metric: str or None, default=None
            Optimization metric for the fit. Only consulted when thinking
            is enabled.

            Binary classification:
                "accuracy", "balanced_accuracy", "mcc", "log_loss",
                "pac", "quadratic_kappa", "roc_auc", "average_precision",
                "precision", "precision_macro", "precision_micro",
                "precision_weighted", "recall", "recall_macro",
                "recall_micro", "recall_weighted", "f1", "f1_macro",
                "f1_micro", "f1_weighted".
            Multiclass classification:
                "accuracy", "balanced_accuracy", "mcc", "log_loss",
                "pac", "quadratic_kappa", "precision_macro",
                "precision_micro", "precision_weighted", "recall_macro",
                "recall_micro", "recall_weighted", "f1_macro",
                "f1_micro", "f1_weighted", "roc_auc_ovo",
                "roc_auc_ovo_macro", "roc_auc_ovr", "roc_auc_ovr_macro",
                "roc_auc_ovr_micro", "roc_auc_ovr_weighted".

            Aliases "acc", "nll", "pac_score" are also accepted.
        fit_mode: {"fit_preprocessors", "fit_with_cache"} or None, default=None
            Controls what the server persists at fit time. None defers to the
            server default, which is "fit_preprocessors".
            "fit_preprocessors" fits only the preprocessing state, so every
            predict re-runs the forward pass from the uploaded train set.
            "fit_with_cache" additionally builds and persists a server-side KV
            cache keyed by the resulting fitted-train-set id; later predicts
            against that id (see `model_id` / `save_model` / `load_model`) are
            served from the cache instead of re-fitting.
        client_options : ClientOptions, default=None
            Client specific options (e.g. timeout, headers).
        model_id : str, UUID or None, default=None
            Reference to an already-fitted train set on the server (the id
            returned by a previous `fit()`, exposed via `save_model()`). When
            set, the estimator is considered fitted and `predict`/`predict_proba`
            run against that id without calling `fit()` again — this is what
            lets a `fit_with_cache` model be reused across processes. A fresh
            `fit()` call always supersedes it. Note that a classifier
            constructed this way has no `classes_` unless it is restored via
            `load_model()`.
        """
        self.model_path = model_path
        self.categorical_features_indices = categorical_features_indices
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.balance_probabilities = balance_probabilities
        self.average_before_softmax = average_before_softmax
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision = inference_precision
        self.random_state = random_state
        self.inference_config = inference_config
        self.fit_mode: FitMode | None = fit_mode
        self.paper_version = paper_version
        self.thinking_mode = thinking_mode
        self.thinking_effort: ThinkingEffort | None = thinking_effort
        self.thinking_timeout_s = thinking_timeout_s
        self.thinking_metric = thinking_metric
        self.client_options = client_options or ClientOptions()
        self.model_id = model_id

        self._last_trace_id = None
        self._last_train_X = None
        self._last_train_y = None
        self._last_meta = {}
        self._last_train_set_description = None
        self._fit_count = 0

    def __sklearn_is_fitted__(self) -> bool:
        # `fitted_` is a legacy fittedness flag some tests set directly; it is
        # kept as a fallback so those tests keep passing.
        return _resolve_train_set_id(self) is not None or getattr(
            self, "fitted_", False
        )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        description: str | None = None,
    ):
        # assert init() is called
        init()

        tabpfn_config = self._get_tabpfn_config()

        task_config = ClassifierConfig(tabpfn_config=tabpfn_config)

        validate_train_set(X, y)
        validate_thinking_mode(
            self.thinking_mode,
            self.thinking_effort,
            self.thinking_timeout_s,
            self.thinking_metric,
            self.model_path,
        )
        X_clean = _clean_text_features(X)
        self._validate_targets_and_classes(y)

        if Config.use_server:
            # Create a new sentry trace at every fit, provided that:
            # - The user has not explicitly set a sentry-trace header.
            # - In any case if we're going to refit.
            # - In any case if we have already called .fit() on this instance.
            if self._fit_count > 0 or "sentry-trace" not in self.client_options.headers:
                self.client_options.headers["sentry-trace"] = uuid4().hex

            self._last_trace_id = self.client_options.headers["sentry-trace"]
            self._last_train_set_description = description

            def fit_task() -> UUID:
                return InferenceClient.fit(
                    X_clean,
                    y,
                    task_config=task_config,
                    paper_version=self.paper_version,
                    thinking_mode=self.thinking_mode,
                    thinking_effort=self.thinking_effort,
                    thinking_timeout_s=self.thinking_timeout_s,
                    thinking_metric=self.thinking_metric,
                    client_options=self.client_options,
                    description=description,
                )

            self.fitted_train_set_id_ = cast(UUID, run_task(fit_task, "Fitting"))
            self._last_train_X = X_clean
            self._last_train_y = y
            self.fitted_ = True
            self._fit_count += 1
        else:
            raise NotImplementedError(
                "Only server mode is supported at the moment for init(use_server=False)"
            )
        return self

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: The input samples.

        Returns:
            The predicted class labels.
        """
        return self._predict(X, output_type="preds")

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Args:
            X: The input samples.

        Returns:
            The class probabilities of the input samples.
        """
        return self._predict(X, output_type="probas")

    def _predict(
        self,
        X,
        output_type: Literal["probas", "preds"],
    ) -> np.ndarray:
        # IMPORTANT: self._get_predict_params() should be called first to make sure
        # we capture the original user-provided values.
        predict_params = self._get_predict_params(locals())

        # A load-by-id estimator (via `model_id` / `load_model`) can reach
        # `predict` without ever calling `fit()`, so `init()` (which authorizes
        # the HTTP client) must run here too. It short-circuits after the first
        # successful call.
        init()
        check_is_fitted(self)

        tabpfn_config = self._get_tabpfn_config()
        task_config = ClassifierConfig(
            tabpfn_config=tabpfn_config,
            predict_params=predict_params,
        )

        validate_test_set(X, output_type, tabpfn_config.model_path)
        X_clean = _clean_text_features(X)

        if (
            "sentry-trace" not in self.client_options.headers
            and self._last_trace_id is not None
        ):
            self.client_options.headers["sentry-trace"] = self._last_trace_id

        def predict_task() -> PredictionResult:
            last_exc = None
            refit_attempts = 0
            while True:
                if refit_attempts > 1:
                    raise RuntimeError(
                        "Failed to predict after refitting"
                    ) from last_exc
                try:
                    return InferenceClient.predict(
                        X_clean,
                        fitted_train_set_id=cast(UUID, _resolve_train_set_id(self)),
                        task_config=task_config,
                        client_options=self.client_options,
                    )
                except NeedsRefittingError as exc:
                    last_exc = exc
                    refit_attempts += 1
                    if self._last_train_X is None or self._last_train_y is None:
                        # A load-by-id estimator (constructed via `model_id` /
                        # `load_model`) has no training data in memory to rebuild
                        # from, so we can't transparently recover here.
                        raise RuntimeError(
                            "The referenced fitted model is no longer available "
                            "on the server and this estimator has no in-memory "
                            "training data to refit (it was created via "
                            "`model_id` / `load_model`). Re-create it by calling "
                            "`fit(X, y)`."
                        ) from exc
                    self.fitted_train_set_id_ = InferenceClient.fit(
                        self._last_train_X,
                        self._last_train_y,
                        task_config=task_config,
                        paper_version=self.paper_version,
                        thinking_mode=self.thinking_mode,
                        thinking_effort=self.thinking_effort,
                        thinking_timeout_s=self.thinking_timeout_s,
                        thinking_metric=self.thinking_metric,
                        client_options=self.client_options,
                        description=self._last_train_set_description,
                    )

        result = run_task(predict_task, "Predicting")
        # Unpack and store metadata
        self._last_meta = result.metadata

        return result.y_pred

    def _get_tabpfn_config(self) -> ClassifierTabPFNConfig:
        init_params = self.get_params()
        cfg = {
            k: v
            for k, v in init_params.items()
            # Nones are treated as unset
            if k in ClassifierTabPFNConfig.model_fields and v is not None
        }
        cfg["model_path"] = self._model_name_to_path("classification", self.model_path)
        return ClassifierTabPFNConfig.model_validate(cfg)

    def _get_predict_params(self, kwargs: dict[str, Any]) -> ClassifierPredictParams:
        params = {
            k: v for k, v in kwargs.items() if k in ClassifierPredictParams.model_fields
        }
        return ClassifierPredictParams.model_validate(params)

    def _validate_targets_and_classes(self, y) -> None:
        y_ = column_or_1d(y, warn=True)
        if sum(pd.isnull(y_)) > 0:
            raise ValueError("Input y contains NaN.")
        check_classification_targets(y)
        # Get classes and encode before type conversion to guarantee correct class labels.
        # TODO: should pass this from the server
        self.classes_ = np.unique(y_)

        # TODO: these things should ideally be shared with the local package
        limits = ServiceClient.get_model_limits()
        if limits is None:
            return

        # We use the most permissive limit across all models as at fit time we
        # don't yet know yet which model will be used.
        limit = limits.max_model_limit

        if len(self.classes_) > limit.max_classes:
            raise ValueError(
                f"Number of classes {len(self.classes_)} exceeds the maximal number of "
                f"{limit.max_classes} classes supported by TabPFN. Consider using "
                "the many_class extension to reduce the number of classes. For code see "
                f"{URL_TABPFN_EXTENSIONS_GITHUB_MANY_CLASS_CODE}"
            )

    @overload
    def save_model(self, path: None = None) -> dict[str, Any]: ...
    @overload
    def save_model(self, path: str | Path) -> Path: ...
    def save_model(self, path: str | Path | None = None) -> dict[str, Any] | Path:
        """Serialize a portable reference to the fitted (server-side) model.

        The handle captures the server-side fitted-train-set id, the estimator
        hyperparameters, and the class labels — everything `load_model()` needs
        to reconstruct an equivalent classifier. No training data leaves the
        client; predictions run against the model referenced by the id, so this
        is most useful with `fit_mode="fit_with_cache"`.

        Parameters
        ----------
        path : str, Path or None, default=None
            If given, the handle is written to this path as JSON and the path
            is returned. If None, the handle dict is returned directly.
        """
        return _build_model_handle(self, task="classification", path=path)

    @classmethod
    def load_model(cls, handle: dict[str, Any] | str | Path) -> "TabPFNClassifier":
        """Reconstruct a classifier from a `save_model()` handle (dict or path).

        The resulting estimator is already fitted (it references the server-side
        model by id) and restores `classes_`, so it can `predict` without
        calling `fit()` again.
        """
        data = _load_model_handle_for(cls, handle, task="classification")
        est = cls(**data["params"])
        est.classes_ = np.asarray(data["classes"])
        return est


class TabPFNRegressor(RegressorMixin, BaseEstimator, TabPFNModelSelection):
    _AVAILABLE_MODELS = [
        DEFAULT_V3_MODEL_PATH,
        DEFAULT_V2_6_MODEL_PATH,
        DEFAULT_V2_5_MODEL_PATH,
        "v2.5_low-skew",
        "v2.5_quantiles",
        "v2.5_real-variant",
        "v2.5_real",
        "v2.5_small-samples",
        "v2.5_variant",
        DEFAULT_V2_MODEL_PATH,
        "auto",
        # Deprecated alias for "auto"; kept for backward compat with users and
        # downstream packages (e.g. tabpfn-time-series) that read this list.
        "default",
        "2noar4o2",
        "5wof9ojf",
        "09gpqh39",
        "wyl4o83o",
    ]

    def __init__(
        self,
        # start: tabpfn_config
        model_path: str | None = None,
        n_estimators: int | None = None,
        softmax_temperature: float | None = None,
        average_before_softmax: bool | None = None,
        ignore_pretraining_limits: bool = False,
        inference_precision: Literal["autocast", "auto"] | None = None,
        random_state: int | None = 0,
        inference_config: dict[str, Any] | None = None,
        categorical_features_indices: list[int] | None = None,
        fit_mode: FitMode | None = None,
        # end: tabpfn_config
        paper_version: bool = False,
        thinking_mode: bool = False,
        thinking_effort: ThinkingEffort | None = None,
        thinking_timeout_s: float | None = None,
        thinking_metric: str | None = None,
        client_options: ClientOptions | None = None,
        model_id: str | UUID | None = None,
    ):
        """Construct a TabPFN regressor.

        This constructs a regressor using the latest model and settings. If you would
        like to use a previous model version, use `create_default_for_version()`
        instead. You can also use `model_path` to specify a particular model.

        Parameters
        ----------
        model_path: str, default="auto"
            The name of the model to use. "auto" lets the server pick the
            latest default model; "default" is accepted as a backward-compatible
            alias. Use `create_default_for_version()` to pin to a specific
            major version.
        n_estimators: int, default=8
            The number of estimators in the TabPFN ensemble. We aggregate the
             predictions of `n_estimators`-many forward passes of TabPFN. Each forward
             pass has (slightly) different input data. Think of this as an ensemble of
             `n_estimators`-many "prompts" of the input data.
        softmax_temperature: float, default=0.9
            The temperature for the softmax function. This is used to control the
            confidence of the model's predictions. Lower values make the model's
            predictions more confident. This is only applied when predicting during a
            post-processing step. Set `softmax_temperature=1.0` for no effect.
        average_before_softmax: bool, default=False
            Only used if `n_estimators > 1`. Whether to average the predictions of the
            estimators before applying the softmax function. This can help to improve
            predictive performance when calibrating the model's confidence. This is only
            applied when predicting during a post-processing step.
        ignore_pretraining_limits: bool, default=False
            Whether to ignore the pre-training limits of the model. The TabPFN models
            have been pre-trained on a specific range of input data. If the input data
            is outside of this range, the model may not perform well. You may ignore
            our limits to use the model on data outside the pre-training range.
        inference_precision: "autocast" or "auto", default="auto"
            The precision to use for inference. This can dramatically affect the
            speed and reproducibility of the inference.
        random_state: int or RandomState or RandomGenerator or None, default=0
            Controls the randomness of the model. Pass an int for reproducible
            results; pass `None` to use a fresh random seed each run.
        inference_config: dict or None, default=None
            Additional advanced arguments for model interface. See the doc of InferenceConfig
            in the tabpfn package for more details. For the client, the inference_config and the
            preprocess transforms need to be dictionaries.
        categorical_features_indices: list[int] or None, default=None
            The indices of the columns that should be treated as categorical.
            If None, the model infers which columns are categorical.
        paper_version: bool, default=False
            If True, will use the model described in the paper, instead of the newest
            version available on the API, which e.g handles text features better.
        thinking_mode: bool, default=False
            If True, spend extra fit-time compute for higher precision.
            Equivalent to passing `thinking_effort="medium"` — setting any
            `thinking_effort` value also enables thinking, so this flag is
            optional when you've set the level explicitly.
        thinking_effort: {"medium", "high"} or None, default=None
            Effort level for thinking mode. When set, thinking is enabled
            (you don't also need `thinking_mode=True`). When None and
            `thinking_mode=True`, defaults to "medium".
        thinking_timeout_s: float or None, default=None
            Budget for the fit, in seconds. Only consulted when thinking is
            enabled. Capped at 2400.
        thinking_metric: str or None, default=None
            Optimization metric for the fit. Only consulted when thinking
            is enabled.

            Regression:
                "r2", "mean_squared_error", "root_mean_squared_error",
                "mean_absolute_error", "median_absolute_error",
                "mean_absolute_percentage_error",
                "symmetric_mean_absolute_percentage_error", "spearmanr",
                "pearsonr".

            Aliases "mse", "rmse", "mae", "mape", "smape" are also
            accepted.
        fit_mode: {"fit_preprocessors", "fit_with_cache"} or None, default=None
            Controls what the server persists at fit time. None defers to the
            server default, which is "fit_preprocessors".
            "fit_preprocessors" fits only the preprocessing state, so every
            predict re-runs the forward pass from the uploaded train set.
            "fit_with_cache" additionally builds and persists a server-side KV
            cache keyed by the resulting fitted-train-set id; later predicts
            against that id (see `model_id` / `save_model` / `load_model`) are
            served from the cache instead of re-fitting.
        client_options : ClientOptions, default=None
            Client specific options (e.g. timeout, headers).
        model_id : str, UUID or None, default=None
            Reference to an already-fitted train set on the server (the id
            returned by a previous `fit()`, exposed via `save_model()`). When
            set, the estimator is considered fitted and `predict` runs against
            that id without calling `fit()` again — this is what lets a
            `fit_with_cache` model be reused across processes. A fresh `fit()`
            call always supersedes it.
        """
        self.model_path = model_path
        self.categorical_features_indices = categorical_features_indices
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision = inference_precision
        self.random_state = random_state
        self.inference_config = inference_config
        self.fit_mode: FitMode | None = fit_mode
        self.paper_version = paper_version
        self.thinking_mode = thinking_mode
        self.thinking_effort: ThinkingEffort | None = thinking_effort
        self.thinking_timeout_s = thinking_timeout_s
        self.thinking_metric = thinking_metric
        self.client_options = client_options or ClientOptions()
        self.model_id = model_id

        self._last_trace_id = None
        self._last_train_X = None
        self._last_train_y = None
        self._last_meta = {}
        self._last_train_set_description = None
        self._fit_count = 0

    def __sklearn_is_fitted__(self) -> bool:
        # `fitted_` is a legacy fittedness flag some tests set directly; it is
        # kept as a fallback so those tests keep passing.
        return _resolve_train_set_id(self) is not None or getattr(
            self, "fitted_", False
        )

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        description: str | None = None,
    ):
        # assert init() is called
        init()

        tabpfn_config = self._get_tabpfn_config()

        task_config = RegressorConfig(tabpfn_config=tabpfn_config)

        validate_train_set(X, y)
        validate_thinking_mode(
            self.thinking_mode,
            self.thinking_effort,
            self.thinking_timeout_s,
            self.thinking_metric,
            self.model_path,
        )
        self._validate_targets(y)
        X_clean = _clean_text_features(X)

        if Config.use_server:
            # Create a new sentry trace at every fit, provided that:
            # - The user has not explicitly set a sentry-trace header.
            # - In any case if we're going to refit.
            # - In any case if we have already called .fit() on this instance.
            if self._fit_count > 0 or "sentry-trace" not in self.client_options.headers:
                self.client_options.headers["sentry-trace"] = uuid4().hex

            self._last_trace_id = self.client_options.headers["sentry-trace"]

            self._last_train_set_description = description

            def fit_task() -> UUID:
                return InferenceClient.fit(
                    X_clean,
                    y,
                    task_config=task_config,
                    paper_version=self.paper_version,
                    thinking_mode=self.thinking_mode,
                    thinking_effort=self.thinking_effort,
                    thinking_timeout_s=self.thinking_timeout_s,
                    thinking_metric=self.thinking_metric,
                    client_options=self.client_options,
                    description=description,
                )

            self.fitted_train_set_id_ = cast(UUID, run_task(fit_task, "Fitting"))
            self._last_train_X = X_clean
            self._last_train_y = y
            self.fitted_ = True
            self._fit_count += 1
        else:
            raise NotImplementedError(
                "Only server mode is supported at the moment for init(use_server=False)"
            )

        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        output_type: Literal[
            "mean", "median", "mode", "quantiles", "full", "main"
        ] = "mean",
        quantiles: list[float] | None = None,
    ) -> np.ndarray | list[np.ndarray] | dict[str, np.ndarray]:
        """Predict regression target for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        output_type : str, default="mean"
            The type of prediction to return:
            - "mean": Return mean prediction
            - "median": Return median prediction
            - "mode": Return mode prediction
            - "quantiles": Return predictions for specified quantiles
            - "full": Return full prediction details
            - "main": Return main prediction metrics
        quantiles : list[float] or None, default=None
            Quantiles to compute when output_type="quantiles".
            Default is [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        Returns
        -------
        array-like or dict
            The predicted values.
        """
        # IMPORTANT: self._get_predict_params() should be called first to make sure
        # we capture the original user-provided values.
        predict_params = self._get_predict_params(locals())

        # A load-by-id estimator (via `model_id` / `load_model`) can reach
        # `predict` without ever calling `fit()`, so `init()` (which authorizes
        # the HTTP client) must run here too. It short-circuits after the first
        # successful call.
        init()
        check_is_fitted(self)

        tabpfn_config = self._get_tabpfn_config()
        task_config = RegressorConfig(
            tabpfn_config=tabpfn_config,
            predict_params=predict_params,
        )

        validate_test_set(X, output_type, tabpfn_config.model_path)
        X_clean = _clean_text_features(X)

        if (
            "sentry-trace" not in self.client_options.headers
            and self._last_trace_id is not None
        ):
            self.client_options.headers["sentry-trace"] = self._last_trace_id

        def predict_task() -> PredictionResult:
            last_exc = None
            refit_attempts = 0
            while True:
                if refit_attempts > 1:
                    raise RuntimeError(
                        "Failed to predict after refitting"
                    ) from last_exc
                try:
                    return InferenceClient.predict(
                        X_clean,
                        fitted_train_set_id=cast(UUID, _resolve_train_set_id(self)),
                        task_config=task_config,
                        client_options=self.client_options,
                    )
                except NeedsRefittingError as exc:
                    last_exc = exc
                    refit_attempts += 1
                    if self._last_train_X is None or self._last_train_y is None:
                        # A load-by-id estimator (constructed via `model_id` /
                        # `load_model`) has no training data in memory to rebuild
                        # from, so we can't transparently recover here.
                        raise RuntimeError(
                            "The referenced fitted model is no longer available "
                            "on the server and this estimator has no in-memory "
                            "training data to refit (it was created via "
                            "`model_id` / `load_model`). Re-create it by calling "
                            "`fit(X, y)`."
                        ) from exc
                    self.fitted_train_set_id_ = InferenceClient.fit(
                        self._last_train_X,
                        self._last_train_y,
                        task_config=task_config,
                        paper_version=self.paper_version,
                        thinking_mode=self.thinking_mode,
                        thinking_effort=self.thinking_effort,
                        thinking_timeout_s=self.thinking_timeout_s,
                        thinking_metric=self.thinking_metric,
                        client_options=self.client_options,
                        description=self._last_train_set_description,
                    )

        result = run_task(predict_task, "Predicting")
        # Unpack and store metadata
        self._last_meta = result.metadata

        output = result.y_pred
        if output_type == "quantiles" and isinstance(output, np.ndarray):
            return list(output) if output.ndim == 2 else [output]
        if output_type == "full":
            try:
                from tabpfn.regressor import FullSupportBarDistribution  # type: ignore
                import torch  # type: ignore

                output["criterion"] = FullSupportBarDistribution(
                    borders=torch.tensor(output["borders"])
                )
            except ImportError:
                logger.warning(
                    "Optional dependencies 'tabpfn' and 'torch' are required to "
                    "construct the criterion when output_type='full'. Skipping criterion."
                )

        return output

    def _get_tabpfn_config(self) -> RegressorTabPFNConfig:
        init_params = self.get_params()
        cfg = {
            k: v
            for k, v in init_params.items()
            # Nones are treated as unset
            if k in RegressorTabPFNConfig.model_fields and v is not None
        }
        cfg["model_path"] = self._model_name_to_path("regression", self.model_path)
        return RegressorTabPFNConfig.model_validate(cfg)

    def _get_predict_params(self, kwargs: dict[str, Any]) -> RegressorPredictParams:
        params = {
            k: v
            for k, v in kwargs.items()
            # Nones are treated as unset
            if k in RegressorPredictParams.model_fields and v is not None
        }
        return RegressorPredictParams.model_validate(params)

    def _validate_targets(self, y) -> None:
        y_ = column_or_1d(y, warn=True)
        if sum(pd.isnull(y_)) > 0:
            raise ValueError("Input y contains NaN.")

    @overload
    def save_model(self, path: None = None) -> dict[str, Any]: ...
    @overload
    def save_model(self, path: str | Path) -> Path: ...
    def save_model(self, path: str | Path | None = None) -> dict[str, Any] | Path:
        """Serialize a portable reference to the fitted (server-side) model.

        The handle captures the server-side fitted-train-set id and the
        estimator hyperparameters — everything `load_model()` needs to
        reconstruct an equivalent regressor. No training data leaves the
        client; predictions run against the model referenced by the id, so this
        is most useful with `fit_mode="fit_with_cache"`.

        Parameters
        ----------
        path : str, Path or None, default=None
            If given, the handle is written to this path as JSON and the path
            is returned. If None, the handle dict is returned directly.
        """
        return _build_model_handle(self, task="regression", path=path)

    @classmethod
    def load_model(cls, handle: dict[str, Any] | str | Path) -> "TabPFNRegressor":
        """Reconstruct a regressor from a `save_model()` handle (dict or path).

        The resulting estimator is already fitted (it references the server-side
        model by id), so it can `predict` without calling `fit()` again.
        """
        data = _load_model_handle_for(cls, handle, task="regression")
        return cls(**data["params"])


def _is_thinking_supported_model_path(model_path: str | None) -> bool:
    """Thinking is server-side supported only for v3 models (or the auto sentinel,
    which lets the server pick — currently a v3 model)."""
    if model_path in _AUTO_MODEL_PATH_ALIASES:
        return True
    # `None` is an auto alias handled above, so the remainder is a concrete path.
    assert model_path is not None
    return V_3_IDENTIFIER in model_path


def validate_thinking_mode(
    thinking_mode: bool,
    thinking_effort: ThinkingEffort | None,
    thinking_timeout_s: float | None,
    thinking_metric: str | None,
    model_path: str | None,
) -> None:
    if (
        thinking_effort is not None
        and thinking_effort not in _VALID_THINKING_EFFORT_LEVELS
    ):
        raise ValueError(
            f"thinking_effort must be one of "
            f"{sorted(_VALID_THINKING_EFFORT_LEVELS)}, got {thinking_effort!r}."
        )
    # Setting `thinking_effort` is itself a way to enable thinking, so the
    # effective state is "either flag set". Knobs that only make sense when
    # thinking is on are rejected only when neither is set.
    thinking_enabled = thinking_mode or thinking_effort is not None
    if not thinking_enabled and (
        thinking_timeout_s is not None or thinking_metric is not None
    ):
        raise ValueError(
            "thinking_timeout_s and thinking_metric are only "
            "consulted when thinking is enabled; pass `thinking_mode=True` "
            "or `thinking_effort=...` to use them."
        )
    if thinking_timeout_s is not None and thinking_timeout_s > THINKING_TIMEOUT_MAX_S:
        raise ValueError(
            f"thinking_timeout_s ({thinking_timeout_s}) exceeds the "
            f"maximum allowed of {THINKING_TIMEOUT_MAX_S} seconds "
            f"({THINKING_TIMEOUT_MAX_S // 60} minutes)."
        )
    # Server silently no-ops thinking on non-v3 models, so users can pay for a
    # request expecting a thinking fit and get a plain one back. Reject upfront.
    if thinking_enabled and not _is_thinking_supported_model_path(model_path):
        raise ValueError(
            f"Thinking mode is only supported on v3 models, got "
            f"model_path={model_path!r}. Either leave model_path at its "
            f"default ('auto') or set it to a v3 model (e.g. 'v3_default')."
        )


def validate_train_set(
    X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None
):
    """Check the integrity of the training data."""

    # check if the number of samples is consistent (ValueError)
    if y is not None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

    limits = ServiceClient.get_model_limits()
    if limits is None:
        return

    # We don't yet know which model will be used, so we use the most permissive limit
    # across all models.
    limit = limits.max_model_limit

    if X.shape[0] > limit.train_set_max_rows:
        raise ValueError(
            f"The number of train rows ({X.shape[0]}) exceeds the maximum of {limit.train_set_max_rows}."
        )
    if X.shape[1] > limit.max_cols:
        raise ValueError(
            f"The number of train columns ({X.shape[1]}) exceeds the maximum of {limit.max_cols}."
        )
    n_cells = X.shape[0] * X.shape[1]
    if n_cells > limit.train_set_max_cells:
        raise ValueError(
            f"The number of train cells ({n_cells}) exceeds the maximum of {limit.train_set_max_cells}."
        )


def validate_test_set(
    X: pd.DataFrame | np.ndarray,
    output_type: str | None,
    model_path: str | None = None,
):
    """Check the integrity of the test data."""

    limits = ServiceClient.get_model_limits()
    if limits is None:
        return

    if not model_path:
        limit = limits.model_limits[limits.default_model_version]
    else:
        model_version = model_version_from_path(model_path)
        limit = model_limit_from_version(model_version, limits.model_limits)

    if X.shape[0] > limit.test_set_max_rows:
        raise ValueError(
            f"The number of test rows ({X.shape[0]}) exceeds the maximum of {limit.test_set_max_rows}. "
            "Split the test set across multiple calls to reduce the number of rows."
        )
    if X.shape[1] > limit.max_cols:
        raise ValueError(
            f"The number of test columns ({X.shape[1]}) exceeds the maximum of {limit.max_cols}."
        )
    n_cells = X.shape[0] * X.shape[1]
    if n_cells > limit.test_set_max_cells:
        raise ValueError(
            f"The number of test cells ({n_cells}) exceeds the maximum of {limit.test_set_max_cells}. "
            "Split the test set across multiple calls to reduce the number of cells."
        )
    if output_type == "full":
        if X.shape[0] > limit.test_set_max_rows_w_full_regression_output:
            raise ValueError(
                f"The number of test rows ({X.shape[0]}) exceeds the maximum of {limit.test_set_max_rows_w_full_regression_output} "
                "for full regression output."
            )


@overload
def _clean_text_features(X: pd.DataFrame) -> pd.DataFrame: ...
@overload
def _clean_text_features(X: np.ndarray) -> np.ndarray: ...
def _clean_text_features(X):
    """
    Clean text features in the input data. This is used to avoid
    serialization errors, which happens when the input data contains
    commas or weird spaces, and to limit the length of the text features.
    """
    # Convert numpy array to pandas DataFrame if necessary
    # not necessary if numpy array of numbers
    data = X
    if Tensor is not None and isinstance(data, Tensor):
        if data.requires_grad:
            data = data.detach()
        if data.is_cuda:
            data = data.cpu()

        data = data.numpy()

    if isinstance(data, np.ndarray):
        if np.issubdtype(data.dtype, np.number):
            return data
        else:
            df = pd.DataFrame(data.copy())
    else:
        df = data.copy()

    # limit to 2500 chars and remove commas for text features
    for col in df.columns:
        # check if we can't convert to float
        try:
            pd.to_numeric(df[col])
        except ValueError:
            if df[col].dtype == object:  # only process string/object columns
                df[col] = (
                    df[col]
                    .str.replace(",", "")
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                    .str.slice(0, 2500)
                )

    # Convert back to numpy if input was numpy (or tensor that was converted to numpy)
    if isinstance(data, np.ndarray):
        return df.to_numpy()
    return df


def run_task(task: Callable, message: str, with_spinner: bool = True) -> Any:
    if not with_spinner or get_opts().TABPFN_CLIENT_CI_MODE:
        result = task()
    else:
        start = time.time()
        spinner = ["-", "\\", "|", "/"]
        i = 0
        minutes = 0
        seconds = 0
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(task)
            while not future.done():
                elapsed = int(time.time() - start)
                minutes = elapsed // 60
                seconds = elapsed % 60
                sys.stdout.write(
                    f"\r{minutes:02d}:{seconds:02d} {message}... {spinner[i % len(spinner)]}"
                )
                sys.stdout.flush()
                time.sleep(0.2)
                i += 1
            result = future.result()
        # Remove spinner, but keep elapsed time
        sys.stdout.write(f"\r{minutes:02d}:{seconds:02d} {message}... Done!\n")
        sys.stdout.flush()
    return result
