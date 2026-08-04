#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
import sys
import time
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
    ThinkingConfig,
    FitTaskConfig,
    ClassifierFitTaskConfig,
    RegressorFitTaskConfig,
    ThinkingEffort,
    TabPFNSystem,
)
from tabpfn_client.models import ApiMode, TabPFNConfig
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

# Prediction compute scales with n_train_rows * n_test_rows, so the API caps
# their product. The effective per-call test row limit therefore shrinks as
# the fitted training set grows (e.g. 1M training rows -> 250k test rows).
# The server sends its budget via get_settings (`predict_row_pairs_budget`);
# this is only the fallback for servers that predate that field.
FALLBACK_PREDICT_ROW_PAIRS_BUDGET = 250_000 * 1_000_000

_VALID_THINKING_EFFORT_LEVELS = frozenset({"medium", "high"})


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
            # In case we get UnknownEnum
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

    # The server-side fitted-train-set id predictions run against. Written by
    # `fit()` (the id the server returns) or assigned directly to reuse a
    # previous fit; absent on unfitted instances, which is what makes
    # `__sklearn_is_fitted__` work.
    model_id_: UUID  # annotation only, no class attribute

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
        api_mode: ApiMode = ApiMode.AUTO,
        client_options: ClientOptions | None = None,
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
            against that id (stored on the estimator as `model_id_`) are
            served from the cache instead of re-fitting.
        api_mode: ApiMode, default=ApiMode.AUTO
            Controls how the client calls the server.
            SYNC: the client waits for the server to complete the request before returning.
            ASYNC: the client returns immediately and the server completes the request in the background.
            AUTO: the client automatically determines the best mode to use based on the request.
        client_options : ClientOptions, default=None
            Client specific options (e.g. timeout, headers).
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
        self.fit_mode = fit_mode
        self.paper_version = paper_version
        self.thinking_mode = thinking_mode
        self.thinking_effort = thinking_effort
        self.thinking_timeout_s = thinking_timeout_s
        self.thinking_metric = thinking_metric
        self.api_mode = api_mode
        self.client_options = client_options or ClientOptions()

        self._last_trace_id = None
        self._last_train_X = None
        self._last_meta = {}
        self._fit_count = 0

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "model_id_", None) is not None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        description: str | None = None,
    ):
        # assert init() is called
        init()
        tabpfn_config = self._get_tabpfn_config()

        validate_train_set(X, y)
        X_clean = _clean_text_features(X)
        classes = self._validate_targets_and_classes(y)  # XXX

        self.thinking_mode = _resolve_thinking_mode(
            self.thinking_mode, self.thinking_effort
        )
        task_config = _build_fit_task_config(tabpfn_config)
        tabpfn_systems = _build_tabpfn_systems(self.paper_version, self.thinking_mode)
        thinking_config = _build_thinking_config(
            enabled=self.thinking_mode,
            effort=self.thinking_effort,
            timeout_secs=self.thinking_timeout_s,
            metric=self.thinking_metric,
        )

        if Config.use_server:
            # NOTE(@trace_id)
            # Create a new sentry trace at every fit, provided that:
            # - The user has not explicitly set a sentry-trace header.
            # - In any case if we're going to refit.
            # - In any case if we have already called .fit() on this instance.
            if self._fit_count > 0 or "sentry-trace" not in self.client_options.headers:
                self.client_options.headers["sentry-trace"] = uuid4().hex

            self._last_trace_id = self.client_options.headers["sentry-trace"]

            def fit_task() -> UUID:
                return InferenceClient.fit(
                    X_clean,
                    y,
                    task_config=task_config,
                    tabpfn_systems=tabpfn_systems,
                    thinking_config=thinking_config,
                    api_mode=self.api_mode,
                    client_options=self.client_options,
                    description=description,
                )

            self.model_id_ = cast(UUID, run_task(fit_task, "Fitting"))
            # Fitted state is committed together, only after the server fit
            # succeeded, so a failed re-fit keeps the previous consistent state.
            self.classes_ = classes  # XXX
            self._last_train_X = X_clean
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

        # An estimator whose `model_id_` was assigned directly (reusing a
        # previous fit) can reach `predict` without ever calling `fit()`, so
        # `init()` (which authorizes the HTTP client) must run here too. It
        # short-circuits after the first successful call.
        init()
        check_is_fitted(self)

        tabpfn_config = self._get_tabpfn_config()
        task_config = ClassifierConfig(
            tabpfn_config=tabpfn_config,
            predict_params=predict_params,
        )

        validate_test_set(
            X,
            output_type,
            tabpfn_config.model_path,
            train_rows=self._last_train_X.shape[0]
            if self._last_train_X is not None
            else None,
        )
        X_clean = _clean_text_features(X)

        if (
            "sentry-trace" not in self.client_options.headers
            and self._last_trace_id is not None
        ):
            self.client_options.headers["sentry-trace"] = self._last_trace_id

        def predict_task() -> PredictionResult:
            return InferenceClient.predict(
                X_clean,
                fitted_train_set_id=self.model_id_,
                task_config=task_config,
                client_options=self.client_options,
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

    def _validate_targets_and_classes(self, y) -> np.ndarray:
        """Validate the targets and return their classes without committing
        them to ``classes_`` — `fit()` assigns fitted state only once the
        server fit succeeded, so a failed re-fit can't leave the old
        ``model_id_`` paired with the new targets' classes."""
        y_ = column_or_1d(y, warn=True)
        if sum(pd.isnull(y_)) > 0:
            raise ValueError("Input y contains NaN.")
        check_classification_targets(y)
        # Get classes and encode before type conversion to guarantee correct class labels.
        # TODO: should pass this from the server
        classes = np.unique(y_)

        # TODO: these things should ideally be shared with the local package
        api_settings = ServiceClient.get_settings()
        if api_settings is None:
            return classes

        # We use the most permissive limit across all models as at fit time we
        # don't yet know yet which model will be used.
        limit = api_settings.max_model_limit

        if len(classes) > limit.max_classes:
            raise ValueError(
                f"Number of classes {len(classes)} exceeds the maximal number of "
                f"{limit.max_classes} classes supported by TabPFN. Consider using "
                "the many_class extension to reduce the number of classes. For code see "
                f"{URL_TABPFN_EXTENSIONS_GITHUB_MANY_CLASS_CODE}"
            )
        return classes


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

    # The server-side fitted-train-set id predictions run against. Written by
    # `fit()` (the id the server returns) or assigned directly to reuse a
    # previous fit; absent on unfitted instances, which is what makes
    # `__sklearn_is_fitted__` work.
    model_id_: UUID

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
        api_mode: ApiMode = ApiMode.AUTO,
        client_options: ClientOptions | None = None,
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
        fit_mode: {"fit_preprocessors", "fit_with_cache"} or None, default=None
            Controls what the server persists at fit time. None defers to the
            server default, which is "fit_preprocessors".
            "fit_preprocessors" fits only the preprocessing state, so every
            predict re-runs the forward pass from the uploaded train set.
            "fit_with_cache" additionally builds and persists a server-side KV
            cache keyed by the resulting fitted-train-set id; later predicts
            against that id (stored on the estimator as `model_id_`) are
            served from the cache instead of re-fitting.
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
        api_mode: ApiMode, default=ApiMode.AUTO
            Controls how the client calls the server.
            SYNC: the client waits for the server to complete the request before returning.
            ASYNC: the client returns immediately and the server completes the request in the background.
            AUTO: the client automatically determines the best mode to use based on the request.
        client_options : ClientOptions, default=None
            Client specific options (e.g. timeout, headers).
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
        self.fit_mode = fit_mode
        self.paper_version = paper_version
        self.thinking_mode = thinking_mode
        self.thinking_effort = thinking_effort
        self.thinking_timeout_s = thinking_timeout_s
        self.thinking_metric = thinking_metric
        self.api_mode = api_mode
        self.client_options = client_options or ClientOptions()

        self._last_trace_id = None
        self._last_train_X = None
        self._last_meta = {}
        self._fit_count = 0

    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "model_id_", None) is not None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        description: str | None = None,
    ):
        # assert init() is called
        init()
        tabpfn_config = self._get_tabpfn_config()

        validate_train_set(X, y)
        self._validate_targets(y)
        X_clean = _clean_text_features(X)

        self.thinking_mode = _resolve_thinking_mode(
            self.thinking_mode, self.thinking_effort
        )
        task_config = _build_fit_task_config(tabpfn_config)
        tabpfn_systems = _build_tabpfn_systems(self.paper_version, self.thinking_mode)
        thinking_config = _build_thinking_config(
            enabled=self.thinking_mode,
            effort=self.thinking_effort,
            timeout_secs=self.thinking_timeout_s,
            metric=self.thinking_metric,
        )

        if Config.use_server:
            # NOTE(@trace_id)
            # Create a new sentry trace at every fit, provided that:
            # - The user has not explicitly set a sentry-trace header.
            # - In any case if we're going to refit.
            # - In any case if we have already called .fit() on this instance.
            if self._fit_count > 0 or "sentry-trace" not in self.client_options.headers:
                self.client_options.headers["sentry-trace"] = uuid4().hex

            self._last_trace_id = self.client_options.headers["sentry-trace"]

            def fit_task() -> UUID:
                return InferenceClient.fit(
                    X_clean,
                    y,
                    task_config=task_config,
                    tabpfn_systems=tabpfn_systems,
                    thinking_config=thinking_config,
                    api_mode=self.api_mode,
                    client_options=self.client_options,
                    description=description,
                )

            self.model_id_ = cast(UUID, run_task(fit_task, "Fitting"))
            self._last_train_X = X_clean
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
        quantiles: list[float] | None = None,  # NOTE: captured in _get_predict_params()
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

        # An estimator whose `model_id_` was assigned directly (reusing a
        # previous fit) can reach `predict` without ever calling `fit()`, so
        # `init()` (which authorizes the HTTP client) must run here too. It
        # short-circuits after the first successful call.
        init()
        check_is_fitted(self)

        tabpfn_config = self._get_tabpfn_config()
        task_config = RegressorConfig(
            tabpfn_config=tabpfn_config,
            predict_params=predict_params,
        )

        validate_test_set(
            X,
            output_type,
            tabpfn_config.model_path,
            train_rows=self._last_train_X.shape[0]
            if self._last_train_X is not None
            else None,
        )
        X_clean = _clean_text_features(X)

        # NOTE(@trace_id)
        # If this instance reuses a previous fit via a directly-assigned
        # `model_id_` we assume this is a fit-once-predict-many scenario, so we
        # won't try to link all operations under the same trace. In this case we
        # will let the server create a new trace for every prediction or use the
        # user-supplied one.
        if (
            "sentry-trace" not in self.client_options.headers
            and self._last_trace_id is not None
        ):
            self.client_options.headers["sentry-trace"] = self._last_trace_id

        def predict_task() -> PredictionResult:
            return InferenceClient.predict(
                X_clean,
                fitted_train_set_id=self.model_id_,
                task_config=task_config,
                client_options=self.client_options,
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


def validate_train_set(
    X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray | None = None
):
    """Check the integrity of the training data."""

    # check if the number of samples is consistent (ValueError)
    if y is not None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

    api_settings = ServiceClient.get_settings()
    if api_settings is None:
        return

    # We don't yet know which model will be used, so we use the most permissive limit
    # across all models.
    limit = api_settings.max_model_limit

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
    train_rows: int | None = None,
):
    """Check the integrity of the test data."""

    api_settings = ServiceClient.get_settings()
    if api_settings is None:
        return

    if not model_path:
        limit = api_settings.model_limits[api_settings.default_model_version]
    else:
        model_version = model_version_from_path(model_path)
        limit = model_limit_from_version(model_version, api_settings.model_limits)

    max_rows = limit.test_set_max_rows
    if train_rows:
        budget = limit.predict_row_pairs_budget
        max_rows = min(max_rows, budget // train_rows)

    if X.shape[0] > max_rows:
        raise ValueError(
            f"The number of test rows ({X.shape[0]}) exceeds the maximum of {max_rows}. "
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


def _build_fit_task_config(tabpfn_config: TabPFNConfig) -> FitTaskConfig:
    match tabpfn_config:
        case ClassifierTabPFNConfig():
            return ClassifierFitTaskConfig(
                tabpfn_config=tabpfn_config,
            )
        case RegressorTabPFNConfig():
            return RegressorFitTaskConfig(
                tabpfn_config=tabpfn_config,
            )


def _build_tabpfn_systems(
    paper_version: bool, thinking_mode: bool
) -> list[TabPFNSystem]:
    if paper_version and thinking_mode:
        raise ValueError(
            "Paper version and thinking mode cannot be enabled at the same time"
        )
    if paper_version:
        return []
    if thinking_mode:
        return ["preprocessing", "text", "thinking"]
    return ["preprocessing", "text"]


def _resolve_thinking_mode(enabled: bool, effort: str | None = None) -> bool:
    # To honour previous contract, setting effort alone is enough to enable thinking.
    if enabled:
        return True
    if effort:
        return True
    return False


def _build_thinking_config(
    *,
    enabled: bool,
    effort: str | None = None,
    timeout_secs: float | None = None,
    metric: str | None = None,
) -> ThinkingConfig | None:
    if not enabled:
        return None
    return ThinkingConfig(
        effort=effort,
        timeout_secs=timeout_secs,
        metric=metric,
    )
