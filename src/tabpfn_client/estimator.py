#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Hosted TabPFN estimators with a scikit-learn-compatible interface."""

from __future__ import annotations

import logging
import warnings
from uuid import UUID, uuid4
from collections.abc import Callable
from typing import Any, Literal, TypeVar, cast, get_args, overload
from typing_extensions import Self

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils import column_or_1d
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from tabpfn_client.client import (
    ServiceClient,
    ClientOptions,
    PredictionResult,
    _resume_hint,
)
from tabpfn_client.config import Config, init
from tabpfn_client.constants import URL_TABPFN_EXTENSIONS_GITHUB_MANY_CLASS_CODE
from tabpfn_client.errors import (
    FitFailedError,
    FittedModelNotFoundError,
    PredictFailedError,
    PredictionGoneError,
)
from tabpfn_client.api_models import ModelVersion
from tabpfn_client.utils import (
    model_limit_from_version,
    model_version_from_path,
    validate_text_handling,
)
from tabpfn_client.service_wrapper import InferenceClient
from tabpfn_client.api_models import (
    ModelLimit,
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
    TextHandling,
    FitResponse,
    FitStatus,
    FitTimings,
)
from tabpfn_client.models import ApiMode, TabPFNConfig, FitModeLiteral
from tabpfn_client.persistence import ModelPersistenceMixin, _PendingPredict

try:
    from torch import Tensor  # type: ignore
except ImportError:
    Tensor = None

TORCH_AVAILABLE = Tensor is not None

logger = logging.getLogger(__name__)

# Sentinel values for `model_path` that defer model selection to the server.
# `None` means the caller didn't pick a model; "auto" is the canonical name
# (matches the OSS tabpfn package); "default" is a backward-compatible alias.
_AUTO_MODEL_PATH_ALIASES = (None, "auto", "default")

# One `<version>_default` alias per model version the API schema declares,
# newest first (the order users see in `list_available_models()`). The server
# resolves each alias to its current default checkpoint for that version.
_DEFAULT_MODEL_NAMES = tuple(f"{v.value}_default" for v in reversed(ModelVersion))

# Prediction compute scales with n_train_rows * n_test_rows, so the API caps
# their product. The effective per-call test row limit therefore shrinks as
# the fitted training set grows (e.g. 1M training rows -> 250k test rows).
# The server sends its budget via get_settings (`predict_row_pairs_budget`);
# this is only the fallback for servers that predate that field.
FALLBACK_PREDICT_ROW_PAIRS_BUDGET = 250_000 * 1_000_000

# The outputs each estimator predicts. The server accepts more, and would bill a
# background prediction whose output the estimator cannot return.
ClassifierOutputLiteral = Literal["preds", "probas"]
RegressorOutputLiteral = Literal["mean", "median", "mode", "quantiles", "full", "main"]


class TabPFNModelSelection:
    """Base class for TabPFN model selection and path handling."""

    _AVAILABLE_MODELS: list[str] = []

    @classmethod
    def list_available_models(cls) -> list[str]:
        """Return the model names accepted as `model_path`."""
        return cls._AVAILABLE_MODELS

    @classmethod
    def create_default_for_version(
        cls, version: ModelVersion | str, **overrides
    ) -> Self:
        """Construct an estimator that uses the given version of the model.

        Any kwargs will override the default settings, except for `model_path`.

        Args:
            version: The model version to use.
            **overrides: Constructor arguments other than `model_path`.

        Returns:
            The constructed estimator.

        Raises:
            ValueError: If `version` is not a known model version.
        """
        try:
            version = ModelVersion(version)
        except ValueError:
            raise ValueError(
                f"Invalid model version: {version}. "
                f"Available versions are: {', '.join(list(ModelVersion))}."
            )
        options = overrides.copy()
        options["model_path"] = f"{version.value}_default"
        return cls(**options)


class _ServerTimingsMixin:
    """Server-reported timings of an estimator's last `fit()` and `predict()`."""

    # Class-level default so estimators pickled before this attribute existed load.
    _last_timings: dict[str, Any] | None = None

    @property
    def last_predict_timings(self) -> dict[str, Any] | None:
        """Seconds per stage of the most recent prediction, or None."""
        return self._last_timings

    def get_timings(self) -> dict[str, dict[str, Any] | None]:
        """Seconds the server spent on the last fit and the last prediction.

        Returns ``{"fit": ..., "predict": ...}``:

        - ``fit``: ``elapsed_s``, ``queue_wait_s``, ``train_set_transform_s``,
          ``fit_s``.
        - ``predict``: ``test_set_transform_queue_wait_s``,
          ``test_set_transform_s``, ``predict_queue_wait_s``, ``predict_s``.
          A prediction sent in several requests reports their sum.

        Queue wait is time spent waiting for the server to start the work. An
        entry is None before the first call, while a fit is pending, on a model
        restored with `load_model()` from a completed fit, and when the server
        does not report timings.
        """
        return {
            "fit": getattr(self, "fit_timings_", None),
            "predict": self.last_predict_timings,
        }


def _combine_timings(
    timings: list[dict[str, Any] | None],
) -> dict[str, Any] | None:
    """Timings of one prediction sent in several requests: seconds add up per stage."""
    reported = [t for t in timings if t is not None]
    if not reported:
        return None
    combined: dict[str, Any] = {}
    for key in dict.fromkeys(k for t in reported for k in t):
        values = [t[key] for t in reported if t.get(key) is not None]
        combined[key] = round(sum(values), 3) if values else None
    return combined


def _timings_dict(timings: FitTimings | None) -> dict[str, Any] | None:
    """Return server-reported fit timings as the plain dict `fit_timings_` holds."""
    return timings.model_dump(mode="json") if timings is not None else None


# What a predict request returns: one result, or one per test set.
_PredictOutputT = TypeVar("_PredictOutputT")


class _AsyncMixin(ModelPersistenceMixin):
    """Starts work on the TabPFN server without waiting for it, and waits for it.

    `submit_fit()` returns as soon as the server has accepted the fit, leaving
    the model pending until `wait_for_fit()` sees the fit complete. In between,
    `save_model()` saves the pending fit, for a later run to load and wait for.
    Each estimator's `submit_predict()` does the same for a prediction, which
    the model holds until its `wait_for_predict()` returns it. `predict()`
    holds its own prediction while it waits for it, as `fit()` holds its fit,
    so a wait that stops can be resumed with `wait_for_predict()`.
    """

    # Set by the estimator's `__init__`.
    client_options: ClientOptions

    def submit_fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        description: str | None = None,
    ) -> Self:
        """Start fitting the model on the server, without waiting for the fit.

        Checks and uploads the training data as `fit()` does, but returns as
        soon as the server has accepted the fit, which it then runs as a
        background job, whatever `api_mode` is. Until `wait_for_fit()` sees
        the fit complete, the model is pending: `predict()` raises
        `NotFittedError`, and `save_model()` saves it as a pending fit, which
        a later run can `load_model()` and wait for. The id of the fit is in
        `model_id_` as soon as `submit_fit()` returns, for logs or support
        requests.

        Args:
            X: The input data.
            y: The target variable.
            description: Description of the dataset and task for the server.

        Returns:
            self

        Raises:
            ValueError: If the data or the estimator's parameters are invalid,
                or the data exceeds the model limits.
        """
        return self._fit(X, y, description, submit=True)

    def wait_for_fit(self) -> Self:
        """Wait for the server to complete the fit `submit_fit()` started.

        Polls the server until the fit has completed, after which the model
        can predict. A model `load_model()` restored from a record of a pending
        fit is waited for the same way. Returns at once, without a request, if
        the fit has already completed.

        Returns:
            self

        Raises:
            NotFittedError: If the model has no fit to wait for.
            FitFailedError: If the server reports that the fit failed. The
                model is then unfitted.
            FittedModelNotFoundError: If the server does not have the fit, for
                instance because it was deleted. The model is then unfitted.
            TimeoutError: If the fit has not completed within
                `TABPFN_CLIENT_ASYNC_POLL_TIMEOUT` seconds. The model is still
                pending, so `wait_for_fit()` can be called again.
            RuntimeError: If polling for the fit fails 4 times in a row with
                an unexpected error. The model is still pending.
        """
        if not self._fit_pending:
            check_is_fitted(
                self,
                msg="This %(name)s has no fit to wait for. Call submit_fit() first.",
            )
            return self
        # A model restored by `load_model()` can get here without `fit()`, which
        # is what otherwise authorizes the HTTP client.
        init()
        logger.debug(f"Waiting for fit {self.model_id_} to complete.")
        try:
            response = InferenceClient.wait_for_fit(
                self.model_id_, client_options=self.client_options
            )
        except (FitFailedError, FittedModelNotFoundError):
            self._forget_fit()
            raise
        self.fit_timings_ = _timings_dict(response.timings)
        self._fit_pending = False
        return self

    def _fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        description: str | None,
        *,
        submit: bool,
    ) -> Self:
        """Fit as `fit()` does or, with `submit`, as `submit_fit()` does."""
        raise NotImplementedError  # Each estimator fits its own task.

    def _check_fitted(self) -> None:
        """Raise `NotFittedError` unless the model has a completed fit.

        Asks the server nothing: whether a pending fit has completed is for
        `wait_for_fit()` to find out.
        """
        if self._fit_pending:
            raise NotFittedError(
                f"Fit {self.model_id_} may not have completed yet. Call "
                "wait_for_fit() before predicting."
            )
        check_is_fitted(self)

    def _set_fit(
        self,
        fit_result: FitResponse,
        *,
        n_train_rows: int,
    ) -> None:
        """Assign the fitted state both estimators keep, for one fit.

        A fit the server is still running leaves the model pending. A pending
        prediction belongs to the previous fit, so it is dropped.
        """
        self._drop_pending_predict("a new fit replaced the one it was made with")
        self.model_id_ = fit_result.fitted_train_set_id
        self.fit_timings_ = _timings_dict(fit_result.timings)
        self._n_train_rows = n_train_rows
        self._fit_pending = fit_result.status != FitStatus.COMPLETED

    def _forget_fit(self) -> None:
        """Drop the fit `model_id_` refers to, leaving the model unfitted."""
        for name in ("model_id_", "classes_", "fit_timings_"):
            vars(self).pop(name, None)
        self._n_train_rows = None
        self._fit_pending = False
        self._drop_pending_predict("the model no longer has its fit")

    def _set_pending_predict(self, pending: _PendingPredict) -> None:
        """Hold a prediction the server accepted, in place of any earlier one."""
        self._drop_pending_predict("submit_predict() started a new one")
        self._pending_predict = pending

    def _predict_holding(
        self,
        output_type: str,
        request: Callable[[Callable[[UUID | None], None]], _PredictOutputT],
        *,
        hold: bool = True,
    ) -> _PredictOutputT:
        """Run `predict()`'s request, holding its prediction as `fit()` holds a fit.

        Once the server has accepted the new prediction, it replaces the
        pending one, as with `submit_predict()`. While the client waits for a
        prediction the server runs as a background job, the model holds it, so
        that a wait that stops (a timeout, an interrupt) leaves it pending for
        `wait_for_predict()` and `save_model()`. A request that fails before
        the server accepts the prediction leaves the earlier one pending.

        Args:
            output_type: The output the prediction is asked for.
            request: Sends the prediction and waits for it, calling the
                callback it is given once the server has accepted it.
            hold: Whether to hold a background job. False for a request of
                several jobs, which one pending prediction cannot stand for.

        Returns:
            What `request` returns.
        """
        held: _PendingPredict | None = None

        def accepted(fitted_test_set_id: UUID | None) -> None:
            nonlocal held
            self._drop_pending_predict("predict() started a new one")
            if hold and fitted_test_set_id is not None:
                held = _PendingPredict(fitted_test_set_id, output_type)
                self._pending_predict = held

        try:
            output = request(accepted)
        except (PredictFailedError, PredictionGoneError):
            # Nothing is left to wait for.
            if held is not None:
                self._pending_predict = None
            raise
        if held is not None:
            self._pending_predict = None
        return output

    def _drop_pending_predict(self, reason: str) -> None:
        """Forget the pending prediction, if there is one, logging its id."""
        pending = self._pending_predict
        if pending is None:
            return
        # The id is the only way back to a prediction the server runs and bills
        # regardless; Python shows only WARNING and above by default.
        logger.warning(
            f"Prediction {pending.fitted_test_set_id} is no longer pending on "
            f"this {type(self).__name__}: {reason}. The server bills it whether "
            "or not anyone waits for it; "
            f"{_resume_hint(pending.fitted_test_set_id)}"
        )
        self._pending_predict = None

    def _await_pending_predict(self) -> tuple[_PendingPredict, PredictionResult]:
        """Wait for the pending prediction.

        The prediction stays pending, so that a wait that stops can be resumed,
        unless the server reports that it failed or that it does not have it;
        the caller drops it once it has its output.

        Raises:
            RuntimeError: If no prediction is pending.
        """
        pending = self._pending_predict
        if pending is None:
            raise RuntimeError(
                f"This {type(self).__name__} has no prediction to wait for. "
                "Call submit_predict() first."
            )
        # A pickled model carries its pending prediction into a process that
        # may not have authorized the HTTP client yet.
        init()
        logger.debug(
            f"Waiting for prediction {pending.fitted_test_set_id} to complete."
        )
        try:
            result = InferenceClient.wait_for_predict(
                pending.fitted_test_set_id, client_options=self.client_options
            )
        except (PredictFailedError, PredictionGoneError):
            # Nothing is left to wait for: no drop warning, whose resume hint
            # would fail.
            self._pending_predict = None
            raise
        return pending, result


class TabPFNClassifier(
    ClassifierMixin, _AsyncMixin, TabPFNModelSelection, _ServerTimingsMixin
):
    """Hosted TabPFN classifier with a scikit-learn-compatible interface.

    Usage guidance:
        - TabPFN-3 and later versions support up to 1,000,000 rows, subject to
          feature count and model/API limits.
        - For large datasets, use per-estimator subsampling,
          e.g. ``inference_config={"SUBSAMPLE_SAMPLES": 100_000}``.
        - Pass raw pandas DataFrames to ``fit`` and ``predict``. Categorical
          strings/categories, raw text, and missing feature values are handled
          automatically; no manual integer/one-hot encoding, imputation, scaling,
          or outlier removal is needed.
    """

    _AVAILABLE_MODELS = [
        # Downstream packages (e.g. tabpfn-time-series) read this list in order
        # to parse model names by substring, so "v2.5_default-2" must precede "v2.5_default".
        "v2.5_default-2",
        *_DEFAULT_MODEL_NAMES,
        "v2.5_large-features-L",
        "v2.5_large-features-XL",
        "v2.5_large-samples",
        "v2.5_real-large-features",
        "v2.5_real-large-samples-and-features",
        "v2.5_real",
        "v2.5_variant",
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
        fit_mode: FitModeLiteral | None = None,
        # end: tabpfn_config
        paper_version: bool = False,
        thinking_mode: bool = False,
        thinking_effort: ThinkingEffort | None = None,
        thinking_timeout_s: float | None = None,
        thinking_metric: str | None = None,
        group_col: str | list[str] | None = None,
        time_col: str | None = None,
        group_time_col: str | None = None,
        api_mode: ApiMode = ApiMode.AUTO,
        client_options: ClientOptions | None = None,
        text_handling: TextHandling = "advanced",
    ):
        """Construct a TabPFN classifier.

        This constructs a classifier using the latest model and settings. If you would
        like to use a previous model version, use `create_default_for_version()`
        instead. You can also use `model_path` to specify a particular model.

        Args:
            model_path:
                The name of the model to use. "auto" (or None) lets the server pick
                the latest default model; "default" is accepted as a backward-compatible
                alias. Use `create_default_for_version()` to pin to a specific major
                version, and `list_available_models()` for the accepted names.

            n_estimators:
                The number of estimators in the TabPFN ensemble. We aggregate the
                predictions of `n_estimators`-many forward passes of TabPFN. Each
                forward pass has (slightly) different input data. Think of this as an
                ensemble of `n_estimators`-many "prompts" of the input data. The
                server accepts at most 8. If None, it applies the model's default:

                - v3.5 (`"v3.5_default"`, the default model): 8.
                - v3.5-fast (`"v3.5-fast_default"`): 4.
                - v3 (`"v3_default"`): 8, raised on wide datasets until every
                  feature is seen by at least one estimator, up to 32.

                These are server defaults and may change with server releases.

            softmax_temperature:
                The temperature for the softmax function. This is used to control the
                confidence of the model's predictions. Lower values make the model's
                predictions more confident. This is only applied when predicting during
                a post-processing step. Set `softmax_temperature=1.0` for no effect. If
                None, the server default is used.

            balance_probabilities:
                Whether to balance the probabilities based on the class distribution
                in the training data. This can help to improve predictive performance
                when the classes are highly imbalanced. This is only applied when
                predicting during a post-processing step.

            average_before_softmax:
                Only used if `n_estimators > 1`. Whether to average the predictions of
                the estimators before applying the softmax function. This can help to
                improve predictive performance when there are many classes or when
                calibrating the model's confidence. This is only applied when predicting
                during a post-processing step. If None, the server default is used.

            ignore_pretraining_limits:
                Whether to ignore the pre-training limits of the model. The TabPFN
                models have been pre-trained on a specific range of input data. If the
                input data is outside of this range, the model may not perform well. You
                may ignore our limits to use the model on data outside the pre-training
                range.

            inference_precision:
                The precision to use for inference. This can dramatically affect the
                speed and reproducibility of the inference. `"autocast"` enables
                mixed-precision autocast; `"auto"` decides based on the device. If None,
                the server default is used.

            random_state:
                Controls the randomness of the model. Pass an int for reproducible
                results; pass `None` to use a fresh random seed each run.

            inference_config:
                For advanced users, additional advanced arguments that adjust the
                behavior of the model interface. See `InferenceConfig` in the tabpfn
                package for details and options. For the client, the inference_config
                and the preprocess transforms need to be dictionaries.

            categorical_features_indices:
                The indices of the columns that should be treated as categorical.
                If None, the model infers which columns are categorical.

            fit_mode:
                Controls what the server persists at fit time. None defers to the
                server default, which is "fit_preprocessors".

                - If `"fit_preprocessors"`, only the preprocessing state is fitted, so
                  every predict re-runs the forward pass from the uploaded train set.
                - If `"fit_with_cache"`, a server-side KV cache is additionally built
                  and persisted, keyed by the resulting fitted-train-set id. Later
                  predicts against that id (stored on the estimator as `model_id_`, and
                  persisted across runs by `save_model()`) are served from the cache
                  instead of re-fitting.

            paper_version:
                If True, will use the model described in the paper, instead of the
                newest version available on the API, which e.g. handles text features
                better. Cannot be combined with thinking mode.

            thinking_mode:
                If True, spend extra fit-time compute for higher precision.
                Equivalent to passing `thinking_effort="medium"`; setting any
                `thinking_effort` value also enables thinking, so this flag is
                optional when you've set the level explicitly.

            thinking_effort:
                Effort level for thinking mode. When set, thinking is enabled
                (you don't also need `thinking_mode=True`). When None and
                `thinking_mode=True`, defaults to "medium".

            thinking_timeout_s:
                Budget for the fit, in seconds. Only consulted when thinking is
                enabled. Capped at 2400.

            thinking_metric:
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

            group_col:
                New since 0.6.0. Column(s) of `X` that identify groups of related
                rows, e.g. a patient or a session id. During the fit, the rows of
                one group are never split between training and validation. The fit
                may also use the other rows of a group to predict a row of it.
                Requires thinking mode, and `X` passed to `fit` and `predict` must
                be a pandas DataFrame that holds the column(s).

            time_col:
                New since 0.6.0. Column of `X` that holds time, as datetimes or
                numbers. During the fit, validation uses contiguous blocks of time.
                Cannot be combined with `group_col`. Requires thinking mode, and
                `X` must be a pandas DataFrame that holds the column.

            group_time_col:
                New since 0.6.0. Column of `X` that orders the rows within a group.
                Requires `group_col` and thinking mode, and `X` must be a pandas
                DataFrame that holds the column.

            api_mode:
                How the client calls the API.

                - `SYNC`: for small workloads, where latency matters. The client
                  waits on one open request.
                - `ASYNC`: for larger workloads. The server runs the work as a
                  background job, and the client polls it until it is done.
                - `AUTO`: `ASYNC` for large workloads, `SYNC` otherwise. A fit
                  runs as a job for a large training set. A prediction runs as
                  a job once the fit's training rows or the rows to predict
                  reach the thresholds the server sets; a model fitted with
                  `fit_mode="fit_with_cache"` predicts with one request.

                In every mode `fit()` and `predict()` return once the work is
                done. A thinking-mode fit always runs as a background job, as do
                `submit_fit()` and `submit_predict()`, whatever this setting is.
                In every mode, `fit()` waits for a thinking-mode fit for at most
                `TABPFN_CLIENT_ASYNC_POLL_TIMEOUT` seconds: by default the
                server's 2 hours, which outlasts the server's own deadline for
                a fit. Against a server that does not run predictions as
                background jobs, `predict()` sends one request in every mode.

            client_options:
                Client specific options (e.g. timeout, headers).

            text_handling:
                Text-processing preset. Both choices support text. Advanced preserves
                the default processing; simple is an alternative whose accuracy depends
                on the dataset. Applies when preprocessing is enabled.
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
        self.text_handling = text_handling
        self.paper_version = paper_version
        self.thinking_mode = thinking_mode
        self.thinking_effort = thinking_effort
        self.thinking_timeout_s = thinking_timeout_s
        self.thinking_metric = thinking_metric
        self.group_col = group_col
        self.time_col = time_col
        self.group_time_col = group_time_col
        self.api_mode = api_mode
        self.client_options = client_options or ClientOptions()

        self._last_trace_id = None
        self._last_meta = {}
        self._last_timings = None
        self._fit_count = 0

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        description: str | None = None,
    ):
        """Fit the model.

        Uploads the training data to the server, which fits the model on it,
        and returns once the fit has completed. Every call starts a new fit,
        also on a model with a pending fit. To start a fit without waiting for
        it, use `submit_fit()`.

        The model takes the new fit as soon as the server has accepted it. A
        fit that `fit()` polls for, as an ASYNC or thinking-mode one, is then
        pending until the poll sees it complete: if the poll stops, for a
        timeout or an interrupt, the model stays pending on the new fit, and
        `wait_for_fit()` resumes the wait.

        Args:
            X: The input data.
            y: The target variable.
            description: Description of the dataset and task for the server.

        Returns:
            self

        Raises:
            ValueError: If the data or the estimator's parameters are invalid,
                or the data exceeds the model limits.
            FitFailedError: If the server reports that the fit failed while
                `fit()` polled for it. The model is then unfitted.
            FittedModelNotFoundError: If the server no longer has the fit
                while `fit()` polls for it. The model is then unfitted.
            TimeoutError: If a fit that `fit()` polls for has not completed
                within `TABPFN_CLIENT_ASYNC_POLL_TIMEOUT` seconds. The model is
                then pending on the new fit.
        """
        return self._fit(X, y, description, submit=False)

    def _fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        description: str | None,
        *,
        submit: bool,
    ) -> Self:
        """Fit as `fit()` does or, with `submit`, as `submit_fit()` does."""
        text_handling = validate_text_handling(self.text_handling)
        # assert init() is called
        init()
        tabpfn_config = self._get_tabpfn_config()

        validate_train_set(X, y, model_path=tabpfn_config.model_path)
        X_clean = _clean_text_features(X)
        classes = self._validate_targets_and_classes(y)

        # NOTE: Always resolve, do not re-assign to self.thinking_mode, user could change effort.
        thinking_mode = _resolve_thinking_mode(self.thinking_mode, self.thinking_effort)
        # Only thinking needs 2 classes; a standard fit accepts one, like the local
        # package. Checked here so the caller doesn't wait for the upload to find out.
        if thinking_mode and len(classes) < 2:
            raise ValueError(
                "y contains only one class; thinking mode needs at least 2 classes."
            )
        task_config = _build_fit_task_config(tabpfn_config)
        tabpfn_systems = _build_tabpfn_systems(self.paper_version, thinking_mode)
        thinking_config = _build_thinking_config(
            X,
            enabled=thinking_mode,
            effort=self.thinking_effort,
            timeout_secs=self.thinking_timeout_s,
            metric=self.thinking_metric,
            group_col=self.group_col,
            time_col=self.time_col,
            group_time_col=self.group_time_col,
        )

        if Config.use_server:
            # NOTE(@trace_id)
            # Create a new sentry trace at every fit, provided that:
            # - The user has not explicitly set a sentry-trace header.
            # - In any case if we have already called .fit() on this instance.
            if self._fit_count > 0 or "sentry-trace" not in self.client_options.headers:
                self.client_options.headers["sentry-trace"] = uuid4().hex

            self._last_trace_id = self.client_options.headers["sentry-trace"]

            fit_result = InferenceClient.fit_with_result(
                X_clean,
                y,
                task_config=task_config,
                tabpfn_systems=tabpfn_systems,
                text_handling=text_handling,
                thinking_config=thinking_config,
                api_mode=_fit_api_mode(
                    self.api_mode, submit=submit, thinking=thinking_mode
                ),
                client_options=self.client_options,
                description=description,
                # Waited for below, once the model holds the new fit.
                wait=False,
            )
            self._set_fit(fit_result, n_train_rows=X.shape[0])
            # Set only after the server has accepted the fit, so a failed refit
            # can't pair new classes with the old model_id_.
            self.classes_ = classes
            self._fit_count += 1
        else:
            raise NotImplementedError(
                "Only server mode is supported at the moment for init(use_server=False)"
            )
        if not submit and self._fit_pending:
            self.wait_for_fit()
        return self

    def predict(self, X):
        """Predict the class labels for the provided input samples.

        Once the server has accepted the prediction, it replaces the model's
        pending prediction, as `submit_predict()` does. While `predict()` waits
        for a prediction the server runs as a background job, the model holds
        it: if the wait stops, for a timeout or an interrupt, the prediction
        stays pending, for `wait_for_predict()` to resume and `save_model()` to
        save.

        Args:
            X: The input data for prediction.

        Returns:
            The predicted class labels as a NumPy array.

        Raises:
            NotFittedError: If the model has no fit, or its fit is pending:
                call `wait_for_fit()` first.
            PredictFailedError: If the server reports that the prediction
                failed, in answer to the prediction request or to a poll for it.
                Another failed request raises RuntimeError, which
                `PredictFailedError` subclasses.
            RefitRequiredError: If the fitted model can no longer predict:
                call `fit()` again.
            TimeoutError: If a prediction the server runs as a background job
                has not completed within `TABPFN_CLIENT_ASYNC_POLL_TIMEOUT`
                seconds. The model then holds it: call `wait_for_predict()`.
        """
        return self._predict(X, output_type="preds")

    def predict_proba(self, X):
        """Predict the probabilities of the classes for the provided input samples.

        Once the server has accepted the prediction, it replaces the model's
        pending prediction, as `submit_predict()` does. While `predict()` waits
        for a prediction the server runs as a background job, the model holds
        it: if the wait stops, for a timeout or an interrupt, the prediction
        stays pending, for `wait_for_predict()` to resume and `save_model()` to
        save.

        Args:
            X: The input data for prediction.

        Returns:
            The predicted probabilities of the classes as a NumPy array.
            Shape (n_samples, n_classes).

        Raises:
            NotFittedError: If the model has no fit, or its fit is pending:
                call `wait_for_fit()` first.
            PredictFailedError: If the server reports that the prediction
                failed, in answer to the prediction request or to a poll for it.
                Another failed request raises RuntimeError, which
                `PredictFailedError` subclasses.
            RefitRequiredError: If the fitted model can no longer predict:
                call `fit()` again.
            TimeoutError: If a prediction the server runs as a background job
                has not completed within `TABPFN_CLIENT_ASYNC_POLL_TIMEOUT`
                seconds. The model then holds it: call `wait_for_predict()`.
        """
        return self._predict(X, output_type="probas")

    def submit_predict(
        self,
        X: pd.DataFrame | np.ndarray,
        output_type: ClassifierOutputLiteral = "preds",
    ) -> Self:
        """Start predicting on the server, without waiting for the predictions.

        Checks and uploads `X` as `predict()` does, but returns as soon as the
        server has accepted the prediction, which it then runs as a background
        job, whatever `api_mode` is. `wait_for_predict()` waits for it and
        returns the predictions. The model holds one pending prediction: a new
        `submit_predict()` or `predict()` replaces it once the server has
        accepted the new prediction, and a new fit drops it. `save_model()`
        saves it, and a pickled or copied model keeps it; `clone()` does not.

        Args:
            X: The input data for prediction.
            output_type: `"preds"` for the class labels `predict()` returns,
                `"probas"` for the probabilities `predict_proba()` returns.

        Returns:
            self

        Raises:
            NotFittedError: If the model has no fit, or its fit is pending:
                call `wait_for_fit()` first.
            ValueError: If `output_type` is not `"preds"` or `"probas"`, `X`
                exceeds the model limits or the group columns are invalid.
            RuntimeError: If the server does not run predictions as background
                jobs.
            PredictFailedError: If the server reports that the prediction
                failed at once, for example because it could not queue it.
        """
        # IMPORTANT: self._get_predict_params() should be called first to make sure
        # we capture the original user-provided values.
        predict_params = self._get_predict_params(locals())
        X_clean, task_config = self._prepare_predict(X, output_type, predict_params)
        fitted_test_set_id = InferenceClient.submit_predict(
            X_clean,
            fitted_train_set_id=self.model_id_,
            task_config=task_config,
            client_options=self.client_options,
        )
        self._set_pending_predict(_PendingPredict(fitted_test_set_id, output_type))
        return self

    def wait_for_predict(self) -> np.ndarray:
        """Wait for the pending prediction, and return it.

        The pending prediction is the one `submit_predict()` started, or one
        `predict()` stopped waiting for. Polls the server until the prediction
        has completed. Until then the prediction is pending, so a wait that
        stops can be resumed by calling `wait_for_predict()` again.

        Returns:
            The predictions, as `predict()` returns them for `"preds"` and
            `predict_proba()` for `"probas"`.

        Raises:
            RuntimeError: If no prediction is pending, polling for it fails,
                or the server no longer has it. In the last case, it is then
                no longer pending.
            PredictFailedError: If the server reports that the prediction
                failed. It is then no longer pending.
            RefitRequiredError: If the fitted model can no longer predict:
                call `fit()` again. The prediction is then no longer pending.
            TimeoutError: If the prediction has not completed within
                `TABPFN_CLIENT_ASYNC_POLL_TIMEOUT` seconds.
        """
        _, result = self._await_pending_predict()
        output = self._finish_predict(result)
        self._pending_predict = None
        return output

    def _predict(
        self,
        X,
        output_type: ClassifierOutputLiteral,
    ) -> np.ndarray:
        """Validate `X` and request `output_type` predictions from the server."""
        # IMPORTANT: self._get_predict_params() should be called first to make sure
        # we capture the original user-provided values.
        predict_params = self._get_predict_params(locals())
        X_clean, task_config = self._prepare_predict(X, output_type, predict_params)
        result = self._predict_holding(
            output_type,
            lambda on_accepted: InferenceClient.predict(
                X_clean,
                fitted_train_set_id=self.model_id_,
                task_config=task_config,
                client_options=self.client_options,
                api_mode=self.api_mode,
                n_train_rows=self._n_train_rows,
                thinking=_resolve_thinking_mode(
                    self.thinking_mode, self.thinking_effort
                ),
                on_accepted=on_accepted,
            ),
        )
        return self._finish_predict(result)

    def _prepare_predict(
        self,
        X: Any,
        output_type: str,
        predict_params: ClassifierPredictParams,
    ) -> tuple[Any, ClassifierConfig]:
        """Check that the model can predict `X`, and build the request for it.

        Returns:
            `X` with its text cleaned, and the task config to send with it.
        """
        # Before `init()`, so that a model with nothing to predict with fails
        # without authenticating first.
        self._check_fitted()
        _validate_output_type(output_type, ClassifierOutputLiteral)

        # An estimator restored by `load_model()` (or whose `model_id_` was
        # assigned directly) can reach `predict` without ever calling `fit()`,
        # so `init()` (which authorizes the HTTP client) must run here too. It
        # short-circuits after the first successful call.
        init()

        tabpfn_config = self._get_tabpfn_config()
        task_config = ClassifierConfig(
            tabpfn_config=tabpfn_config,
            predict_params=predict_params,
        )

        validate_test_set(
            X,
            output_type,
            tabpfn_config.model_path,
            train_rows=self._n_train_rows,
            inference_config=tabpfn_config.inference_config,
        )
        _validate_group_columns(
            X,
            enabled=_resolve_thinking_mode(self.thinking_mode, self.thinking_effort),
            group_col=self.group_col,
            time_col=self.time_col,
            group_time_col=self.group_time_col,
        )
        X_clean = _clean_text_features(X)

        if (
            "sentry-trace" not in self.client_options.headers
            and self._last_trace_id is not None
        ):
            self.client_options.headers["sentry-trace"] = self._last_trace_id
        return X_clean, task_config

    def _finish_predict(self, result: PredictionResult) -> np.ndarray:
        """Store the metadata and timings of a prediction, and return its output."""
        # Unpack and store metadata
        self._last_meta = result.metadata
        self._last_timings = result.timings

        return cast("np.ndarray", result.y_pred)

    def _get_tabpfn_config(self) -> ClassifierTabPFNConfig:
        """Build the model config from the constructor parameters that are not None."""
        init_params = self.get_params()
        cfg = {
            k: v
            for k, v in init_params.items()
            # Nones are treated as unset
            if k in ClassifierTabPFNConfig.model_fields and v is not None
        }
        return ClassifierTabPFNConfig.model_validate(cfg)

    def _get_predict_params(self, kwargs: dict[str, Any]) -> ClassifierPredictParams:
        """Build the predict parameters from the matching entries of `kwargs`."""
        params = {
            k: v for k, v in kwargs.items() if k in ClassifierPredictParams.model_fields
        }
        return ClassifierPredictParams.model_validate(params)

    def _validate_targets_and_classes(self, y) -> np.ndarray:
        """Validate the targets and return their classes without setting ``classes_``.

        `fit()` assigns fitted state only once the server fit succeeded, so a
        failed re-fit can't leave the old ``model_id_`` paired with the new
        targets' classes.

        Raises:
            ValueError: If `y` contains NaN, is not a valid classification
                target, or has more classes than the server supports.
        """
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


class TabPFNRegressor(
    RegressorMixin, _AsyncMixin, TabPFNModelSelection, _ServerTimingsMixin
):
    """Hosted TabPFN regressor with a scikit-learn-compatible interface.

    Usage guidance:
        - TabPFN-3 and later versions support up to 1,000,000 rows, subject to
          feature count and model/API limits.
        - For large datasets, use per-estimator subsampling,
          e.g. ``inference_config={"SUBSAMPLE_SAMPLES": 100_000}``.
        - Pass raw pandas DataFrames to ``fit`` and ``predict``. Categorical
          strings/categories, raw text, and missing feature values are handled
          automatically; no manual integer/one-hot encoding, imputation, scaling,
          or outlier removal is needed.
    """

    _AVAILABLE_MODELS = [
        *_DEFAULT_MODEL_NAMES,
        "v2.5_low-skew",
        "v2.5_quantiles",
        "v2.5_real-variant",
        "v2.5_real",
        "v2.5_small-samples",
        "v2.5_variant",
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
        fit_mode: FitModeLiteral | None = None,
        # end: tabpfn_config
        paper_version: bool = False,
        thinking_mode: bool = False,
        thinking_effort: ThinkingEffort | None = None,
        thinking_timeout_s: float | None = None,
        thinking_metric: str | None = None,
        group_col: str | list[str] | None = None,
        time_col: str | None = None,
        group_time_col: str | None = None,
        api_mode: ApiMode = ApiMode.AUTO,
        client_options: ClientOptions | None = None,
        text_handling: TextHandling = "advanced",
    ):
        """Construct a TabPFN regressor.

        This constructs a regressor using the latest model and settings. If you would
        like to use a previous model version, use `create_default_for_version()`
        instead. You can also use `model_path` to specify a particular model.

        Args:
            model_path:
                The name of the model to use. "auto" (or None) lets the server pick
                the latest default model; "default" is accepted as a backward-compatible
                alias. Use `create_default_for_version()` to pin to a specific major
                version, and `list_available_models()` for the accepted names.

            n_estimators:
                The number of estimators in the TabPFN ensemble. We aggregate the
                predictions of `n_estimators`-many forward passes of TabPFN. Each
                forward pass has (slightly) different input data. Think of this as an
                ensemble of `n_estimators`-many "prompts" of the input data. The
                server accepts at most 8. If None, it applies the model's default:

                - v3.5 (`"v3.5_default"`, the default model): 8.
                - v3.5-fast (`"v3.5-fast_default"`): 4.
                - v3 (`"v3_default"`): 8, raised on wide datasets until every
                  feature is seen by at least one estimator, up to 32.

                These are server defaults and may change with server releases.

            softmax_temperature:
                The temperature for the softmax function. This is used to control the
                confidence of the model's predictions. Lower values make the model's
                predictions more confident. This is only applied when predicting during
                a post-processing step. Set `softmax_temperature=1.0` for no effect. If
                None, the server default is used.

            average_before_softmax:
                Only used if `n_estimators > 1`. Whether to average the predictions of
                the estimators before applying the softmax function. This can help to
                improve predictive performance when calibrating the model's confidence.
                This is only applied when predicting during a post-processing step. If
                None, the server default is used.

            ignore_pretraining_limits:
                Whether to ignore the pre-training limits of the model. The TabPFN
                models have been pre-trained on a specific range of input data. If the
                input data is outside of this range, the model may not perform well. You
                may ignore our limits to use the model on data outside the pre-training
                range.

            inference_precision:
                The precision to use for inference. This can dramatically affect the
                speed and reproducibility of the inference. `"autocast"` enables
                mixed-precision autocast; `"auto"` decides based on the device. If None,
                the server default is used.

            random_state:
                Controls the randomness of the model. Pass an int for reproducible
                results; pass `None` to use a fresh random seed each run.

            inference_config:
                For advanced users, additional advanced arguments that adjust the
                behavior of the model interface. See `InferenceConfig` in the tabpfn
                package for details and options. For the client, the inference_config
                and the preprocess transforms need to be dictionaries.

            categorical_features_indices:
                The indices of the columns that should be treated as categorical.
                If None, the model infers which columns are categorical.

            fit_mode:
                Controls what the server persists at fit time. None defers to the
                server default, which is "fit_preprocessors".

                - If `"fit_preprocessors"`, only the preprocessing state is fitted, so
                  every predict re-runs the forward pass from the uploaded train set.
                - If `"fit_with_cache"`, a server-side KV cache is additionally built
                  and persisted, keyed by the resulting fitted-train-set id. Later
                  predicts against that id (stored on the estimator as `model_id_`, and
                  persisted across runs by `save_model()`) are served from the cache
                  instead of re-fitting.

            paper_version:
                If True, will use the model described in the paper, instead of the
                newest version available on the API, which e.g. handles text features
                better. Cannot be combined with thinking mode.

            thinking_mode:
                If True, spend extra fit-time compute for higher precision.
                Equivalent to passing `thinking_effort="medium"`; setting any
                `thinking_effort` value also enables thinking, so this flag is
                optional when you've set the level explicitly.

            thinking_effort:
                Effort level for thinking mode. When set, thinking is enabled
                (you don't also need `thinking_mode=True`). When None and
                `thinking_mode=True`, defaults to "medium".

            thinking_timeout_s:
                Budget for the fit, in seconds. Only consulted when thinking is
                enabled. Capped at 2400.

            thinking_metric:
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

            group_col:
                New since 0.6.0. Column(s) of `X` that identify groups of related
                rows, e.g. a patient or a session id. During the fit, the rows of
                one group are never split between training and validation. The fit
                may also use the other rows of a group to predict a row of it.
                Requires thinking mode, and `X` passed to `fit` and `predict` must
                be a pandas DataFrame that holds the column(s).

            time_col:
                New since 0.6.0. Column of `X` that holds time, as datetimes or
                numbers. During the fit, validation uses contiguous blocks of time.
                Cannot be combined with `group_col`. Requires thinking mode, and
                `X` must be a pandas DataFrame that holds the column.

            group_time_col:
                New since 0.6.0. Column of `X` that orders the rows within a group.
                Requires `group_col` and thinking mode, and `X` must be a pandas
                DataFrame that holds the column.

            api_mode:
                How the client calls the API.

                - `SYNC`: for small workloads, where latency matters. The client
                  waits on one open request.
                - `ASYNC`: for larger workloads. The server runs the work as a
                  background job, and the client polls it until it is done.
                - `AUTO`: `ASYNC` for large workloads, `SYNC` otherwise. A fit
                  runs as a job for a large training set. A prediction runs as
                  a job once the fit's training rows or the rows to predict
                  reach the thresholds the server sets; a model fitted with
                  `fit_mode="fit_with_cache"` predicts with one request.

                In every mode `fit()` and `predict()` return once the work is
                done. A thinking-mode fit always runs as a background job, as do
                `submit_fit()` and `submit_predict()`, whatever this setting is.
                In every mode, `fit()` waits for a thinking-mode fit for at most
                `TABPFN_CLIENT_ASYNC_POLL_TIMEOUT` seconds: by default the
                server's 2 hours, which outlasts the server's own deadline for
                a fit. Against a server that does not run predictions as
                background jobs, `predict()` sends one request in every mode.

            client_options:
                Client specific options (e.g. timeout, headers).

            text_handling:
                Text-processing preset. Both choices support text. Advanced preserves
                the default processing; simple is an alternative whose accuracy depends
                on the dataset. Applies when preprocessing is enabled.
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
        self.text_handling = text_handling
        self.paper_version = paper_version
        self.thinking_mode = thinking_mode
        self.thinking_effort = thinking_effort
        self.thinking_timeout_s = thinking_timeout_s
        self.thinking_metric = thinking_metric
        self.group_col = group_col
        self.time_col = time_col
        self.group_time_col = group_time_col
        self.api_mode = api_mode
        self.client_options = client_options or ClientOptions()

        self._last_trace_id = None
        self._last_meta = {}
        self._last_timings = None
        self._fit_count = 0

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        description: str | None = None,
    ):
        """Fit the model.

        Uploads the training data to the server, which fits the model on it,
        and returns once the fit has completed. Every call starts a new fit,
        also on a model with a pending fit. To start a fit without waiting for
        it, use `submit_fit()`.

        The model takes the new fit as soon as the server has accepted it. A
        fit that `fit()` polls for, as an ASYNC or thinking-mode one, is then
        pending until the poll sees it complete: if the poll stops, for a
        timeout or an interrupt, the model stays pending on the new fit, and
        `wait_for_fit()` resumes the wait.

        Args:
            X: The input data.
            y: The target variable.
            description: Description of the dataset and task for the server.

        Returns:
            self

        Raises:
            ValueError: If the data or the estimator's parameters are invalid,
                or the data exceeds the model limits.
            FitFailedError: If the server reports that the fit failed while
                `fit()` polled for it. The model is then unfitted.
            FittedModelNotFoundError: If the server no longer has the fit
                while `fit()` polls for it. The model is then unfitted.
            TimeoutError: If a fit that `fit()` polls for has not completed
                within `TABPFN_CLIENT_ASYNC_POLL_TIMEOUT` seconds. The model is
                then pending on the new fit.
        """
        return self._fit(X, y, description, submit=False)

    def _fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        description: str | None,
        *,
        submit: bool,
    ) -> Self:
        """Fit as `fit()` does or, with `submit`, as `submit_fit()` does."""
        text_handling = validate_text_handling(self.text_handling)
        # assert init() is called
        init()
        tabpfn_config = self._get_tabpfn_config()

        validate_train_set(X, y, model_path=tabpfn_config.model_path)
        self._validate_targets(y)
        X_clean = _clean_text_features(X)

        thinking_mode = _resolve_thinking_mode(self.thinking_mode, self.thinking_effort)
        task_config = _build_fit_task_config(tabpfn_config)
        tabpfn_systems = _build_tabpfn_systems(self.paper_version, thinking_mode)
        thinking_config = _build_thinking_config(
            X,
            enabled=thinking_mode,
            effort=self.thinking_effort,
            timeout_secs=self.thinking_timeout_s,
            metric=self.thinking_metric,
            group_col=self.group_col,
            time_col=self.time_col,
            group_time_col=self.group_time_col,
        )

        if Config.use_server:
            if self._fit_count > 0 or "sentry-trace" not in self.client_options.headers:
                self.client_options.headers["sentry-trace"] = uuid4().hex

            self._last_trace_id = self.client_options.headers["sentry-trace"]

            fit_result = InferenceClient.fit_with_result(
                X_clean,
                y,
                task_config=task_config,
                tabpfn_systems=tabpfn_systems,
                text_handling=text_handling,
                thinking_config=thinking_config,
                api_mode=_fit_api_mode(
                    self.api_mode, submit=submit, thinking=thinking_mode
                ),
                client_options=self.client_options,
                description=description,
                # Waited for below, once the model holds the new fit.
                wait=False,
            )
            self._set_fit(fit_result, n_train_rows=X.shape[0])
            self._fit_count += 1
        else:
            raise NotImplementedError(
                "Only server mode is supported at the moment for init(use_server=False)"
            )
        if not submit and self._fit_pending:
            self.wait_for_fit()
        return self

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        output_type: RegressorOutputLiteral = "mean",
        quantiles: list[float] | None = None,  # NOTE: captured in _get_predict_params()
    ) -> np.ndarray | list[np.ndarray] | dict[str, np.ndarray]:
        """Predict regression target for X.

        Once the server has accepted the prediction, it replaces the model's
        pending prediction, as `submit_predict()` does. While `predict()` waits
        for a prediction the server runs as a background job, the model holds
        it: if the wait stops, for a timeout or an interrupt, the prediction
        stays pending, for `wait_for_predict()` to resume and `save_model()` to
        save. A `"full"` output split over several requests is not held: if the
        wait for it stops, the client logs the ids of the predictions still
        running.

        Args:
            X: The input data, of shape (n_samples, n_features).
            output_type: Determines the type of output to return.

                - If `"mean"`, we return the mean over the predicted distribution.
                - If `"median"`, we return the median over the predicted
                  distribution.
                - If `"mode"`, we return the mode over the predicted distribution.
                - If `"quantiles"`, we return the quantiles of the predicted
                  distribution. The parameter `quantiles` determines which
                  quantiles are returned.
                - If `"main"`, we return all the output types above in a dict.
                - If `"full"`, we return the full output of the model, including
                  the logits, the borders and all the output types from "main".
                  The criterion is included when the optional `tabpfn` and
                  `torch` packages are installed. A full output too large for
                  one response is split over several requests; those that run
                  as background jobs overlap, up to 4 at once.

            quantiles: The quantiles to return if `output_type="quantiles"`.
                By default, the `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]`
                quantiles are returned.

        Returns:
            An array of predictions for `"mean"`, `"median"` and `"mode"`; a
            list with one array per quantile for `"quantiles"`; a dict of arrays
            for `"main"` and `"full"`.

        Raises:
            NotFittedError: If the model has no fit, or its fit is pending:
                call `wait_for_fit()` first.
            ValueError: If `output_type` is not one of the above, `X` exceeds
                the model limits or the group columns are invalid.
            RuntimeError: If a `"full"` prediction split across several requests
                returns fields that cannot be combined.
            PredictFailedError: If the server reports that the prediction
                failed, in answer to the prediction request or to a poll for it.
                Another failed request raises RuntimeError, which
                `PredictFailedError` subclasses.
            RefitRequiredError: If the fitted model can no longer predict:
                call `fit()` again.
            TimeoutError: If a prediction the server runs as a background job
                has not completed within `TABPFN_CLIENT_ASYNC_POLL_TIMEOUT`
                seconds. The model then holds it: call `wait_for_predict()`.
        """
        # IMPORTANT: self._get_predict_params() should be called first to make sure
        # we capture the original user-provided values.
        predict_params = self._get_predict_params(locals())
        task_config, rows_per_call = self._prepare_predict(
            X, output_type, predict_params, split_full_output=True
        )

        thinking = _resolve_thinking_mode(self.thinking_mode, self.thinking_effort)
        if rows_per_call is None:
            result = self._predict_holding(
                output_type,
                lambda on_accepted: InferenceClient.predict(
                    _clean_text_features(X),
                    fitted_train_set_id=self.model_id_,
                    task_config=task_config,
                    client_options=self.client_options,
                    api_mode=self.api_mode,
                    n_train_rows=self._n_train_rows,
                    thinking=thinking,
                    on_accepted=on_accepted,
                ),
            )
            results = [result]
        else:
            results = self._predict_holding(
                output_type,
                lambda on_accepted: InferenceClient.predict_chunks(
                    # A generator, so that each chunk is sliced only when it is
                    # sent.
                    (
                        _clean_text_features(
                            _row_slice(X, start, start + rows_per_call)
                        )
                        for start in range(0, X.shape[0], rows_per_call)
                    ),
                    fitted_train_set_id=self.model_id_,
                    task_config=task_config,
                    client_options=self.client_options,
                    api_mode=self.api_mode,
                    n_train_rows=self._n_train_rows,
                    thinking=thinking,
                    on_accepted=on_accepted,
                ),
                # Several jobs, whose ids the client logs if the wait stops.
                hold=False,
            )
        return self._finish_predict(results, output_type)

    def submit_predict(
        self,
        X: pd.DataFrame | np.ndarray,
        output_type: RegressorOutputLiteral = "mean",
        quantiles: list[float] | None = None,  # NOTE: captured in _get_predict_params()
    ) -> Self:
        """Start predicting on the server, without waiting for the predictions.

        Checks and uploads `X` as `predict()` does, but returns as soon as the
        server has accepted the prediction, which it then runs as a background
        job, whatever `api_mode` is. `wait_for_predict()` waits for it and
        returns the predictions. The model holds one pending prediction: a new
        `submit_predict()` or `predict()` replaces it once the server has
        accepted the new prediction, and a new fit drops it. `save_model()`
        saves it, and a pickled or copied model keeps it; `clone()` does not.

        Args:
            X: The input data, of shape (n_samples, n_features).
            output_type: The type of output to return, as for `predict()`. A
                `"full"` output must fit in one response; `predict()` splits a
                larger one over several requests.
            quantiles: The quantiles to return if `output_type="quantiles"`, as
                for `predict()`.

        Returns:
            self

        Raises:
            NotFittedError: If the model has no fit, or its fit is pending:
                call `wait_for_fit()` first.
            ValueError: If `output_type` is not one `predict()` takes, `X`
                exceeds the model limits, the group columns are invalid, or a
                `"full"` output for `X` does not fit in one response.
            RuntimeError: If the server does not run predictions as background
                jobs.
            PredictFailedError: If the server reports that the prediction
                failed at once, for example because it could not queue it.
        """
        # IMPORTANT: self._get_predict_params() should be called first to make sure
        # we capture the original user-provided values.
        predict_params = self._get_predict_params(locals())
        task_config, _ = self._prepare_predict(
            X, output_type, predict_params, split_full_output=False
        )
        fitted_test_set_id = InferenceClient.submit_predict(
            _clean_text_features(X),
            fitted_train_set_id=self.model_id_,
            task_config=task_config,
            client_options=self.client_options,
        )
        self._set_pending_predict(_PendingPredict(fitted_test_set_id, output_type))
        return self

    def wait_for_predict(self) -> np.ndarray | list[np.ndarray] | dict[str, np.ndarray]:
        """Wait for the pending prediction, and return it.

        The pending prediction is the one `submit_predict()` started, or one
        `predict()` stopped waiting for. Polls the server until the prediction
        has completed. Until then the prediction is pending, so a wait that
        stops can be resumed by calling `wait_for_predict()` again.

        Returns:
            The predictions, as `predict()` returns them for the `output_type`
            and `quantiles` the prediction was started with.

        Raises:
            RuntimeError: If no prediction is pending, polling for it fails,
                or the server no longer has it. In the last case, it is then
                no longer pending.
            PredictFailedError: If the server reports that the prediction
                failed. It is then no longer pending.
            RefitRequiredError: If the fitted model can no longer predict:
                call `fit()` again. The prediction is then no longer pending.
            TimeoutError: If the prediction has not completed within
                `TABPFN_CLIENT_ASYNC_POLL_TIMEOUT` seconds.
        """
        pending, result = self._await_pending_predict()
        output = self._finish_predict([result], pending.output_type)
        self._pending_predict = None
        return output

    def _prepare_predict(
        self,
        X: Any,
        output_type: str,
        predict_params: RegressorPredictParams,
        *,
        split_full_output: bool,
    ) -> tuple[RegressorConfig, int | None]:
        """Check that the model can predict `X`, and build the request for it.

        `split_full_output` says whether the caller splits a `"full"` output
        that one response cannot cover, as `predict()` does; otherwise such an
        output raises `ValueError`.

        Returns:
            The task config to send, and the rows one request may cover when a
            `"full"` output for `X` has to be split, else None.
        """
        # Before `init()`, so that a model with nothing to predict with fails
        # without authenticating first.
        self._check_fitted()
        _validate_output_type(output_type, RegressorOutputLiteral)

        # An estimator restored by `load_model()` (or whose `model_id_` was
        # assigned directly) can reach `predict` without ever calling `fit()`,
        # so `init()` (which authorizes the HTTP client) must run here too. It
        # short-circuits after the first successful call.
        init()

        tabpfn_config = self._get_tabpfn_config()
        task_config = RegressorConfig(
            tabpfn_config=tabpfn_config,
            predict_params=predict_params,
        )

        # A full-output response carries one logit per histogram bar for every
        # test row, so the server caps the rows a single response may cover.
        # Splitting the call keeps the returned arrays identical to what one
        # unrestricted call would have produced.
        rows_per_call = (
            _full_output_row_limit(tabpfn_config.model_path)
            if output_type == "full"
            else None
        )
        chunked = rows_per_call is not None and X.shape[0] > rows_per_call
        validate_test_set(
            X,
            output_type,
            tabpfn_config.model_path,
            train_rows=self._n_train_rows,
            inference_config=tabpfn_config.inference_config,
            split_full_output=chunked,
        )
        # A pending prediction holds one id, so a background prediction is one
        # request.
        if chunked and not split_full_output:
            raise ValueError(
                f"The full output for {X.shape[0]} test rows is too large for one "
                f"background prediction, which covers at most {rows_per_call} "
                "rows; use predict(), which splits it."
            )
        _validate_group_columns(
            X,
            enabled=_resolve_thinking_mode(self.thinking_mode, self.thinking_effort),
            group_col=self.group_col,
            time_col=self.time_col,
            group_time_col=self.group_time_col,
        )

        # NOTE(@trace_id)
        # If this instance reuses a previous fit (restored by `load_model()` or
        # via a directly-assigned `model_id_`) we assume this is a
        # fit-once-predict-many scenario, so we won't try to link all operations
        # under the same trace. In this case we will let the server create a new
        # trace for every prediction or use the user-supplied one.
        if (
            "sentry-trace" not in self.client_options.headers
            and self._last_trace_id is not None
        ):
            self.client_options.headers["sentry-trace"] = self._last_trace_id
        return task_config, rows_per_call if chunked else None

    def _finish_predict(
        self,
        results: list[PredictionResult],
        output_type: str,
    ) -> np.ndarray | list[np.ndarray] | dict[str, np.ndarray]:
        """Store the metadata and timings of a prediction, and shape its output.

        `results` holds one result per request the prediction was sent in.
        """
        if len(results) > 1:
            # Metadata describes the request, and every chunk shares the same
            # config; the last one stands for the whole prediction.
            self._last_meta = results[-1].metadata
            self._last_timings = _combine_timings([r.timings for r in results])
            output = _merge_full_outputs(
                [cast("dict[str, np.ndarray]", r.y_pred) for r in results]
            )
        else:
            # Unpack and store metadata
            self._last_meta = results[0].metadata
            self._last_timings = results[0].timings
            output = results[0].y_pred

        if output_type == "quantiles" and isinstance(output, np.ndarray):
            return list(output) if output.ndim == 2 else [output]
        if output_type == "full":
            # `criterion` is a bar distribution rather than an array, so the
            # full output is looser than the declared `dict[str, np.ndarray]`.
            full = cast("dict[str, Any]", output)
            if "logits" in full:
                full["logits"] = _restore_masked_logits(full["logits"])
            try:
                from tabpfn.regressor import FullSupportBarDistribution  # type: ignore
                import torch  # type: ignore

                full["criterion"] = FullSupportBarDistribution(
                    borders=torch.tensor(full["borders"])
                )
            except ImportError:
                warnings.warn(
                    "Optional dependencies 'tabpfn' and 'torch' are required to "
                    "construct the criterion when output_type='full'. Skipping criterion.",
                    # Points at the user's call of `predict()` or
                    # `wait_for_predict()`, which call this.
                    stacklevel=3,
                )
            return full

        return output

    def _get_tabpfn_config(self) -> RegressorTabPFNConfig:
        """Build the model config from the constructor parameters that are not None."""
        init_params = self.get_params()
        cfg = {
            k: v
            for k, v in init_params.items()
            # Nones are treated as unset
            if k in RegressorTabPFNConfig.model_fields and v is not None
        }
        return RegressorTabPFNConfig.model_validate(cfg)

    def _get_predict_params(self, kwargs: dict[str, Any]) -> RegressorPredictParams:
        """Build the predict parameters from the non-None entries of `kwargs`."""
        params = {
            k: v
            for k, v in kwargs.items()
            # Nones are treated as unset
            if k in RegressorPredictParams.model_fields and v is not None
        }
        return RegressorPredictParams.model_validate(params)

    def _validate_targets(self, y) -> None:
        """Check that `y` is one-dimensional and free of NaN.

        Raises:
            ValueError: If `y` contains NaN.
        """
        y_ = column_or_1d(y, warn=True)
        if sum(pd.isnull(y_)) > 0:
            raise ValueError("Input y contains NaN.")


def validate_train_set(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | None = None,
    model_path: str | None = None,
) -> None:
    """Check the integrity of the training data.

    Args:
        X: The training input data.
        y: The training targets, if available.
        model_path: The model the data is fitted with; selects the limits.

    Raises:
        ValueError: If `X` and `y` differ in length, or `X` exceeds the model's
            row, column or upload-cell limits.
    """
    # check if the number of samples is consistent (ValueError)
    if y is not None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

    limit = _limit_for_model_path(model_path)
    if limit is None:
        return

    if X.shape[0] > limit.train_set_max_rows:
        raise ValueError(
            f"The number of train rows ({X.shape[0]}) exceeds the maximum of {limit.train_set_max_rows}."
        )
    if X.shape[1] > limit.max_cols:
        raise ValueError(
            f"The number of train columns ({X.shape[1]}) exceeds the maximum of {limit.max_cols}."
        )
    n_cells = X.shape[0] * X.shape[1]
    upload_limit = limit.train_set_max_upload_cells
    if n_cells > upload_limit:
        raise ValueError(
            f"The training upload has {n_cells} cells, exceeding the maximum of {upload_limit}."
        )


def _limit_for_model_path(model_path: str | None) -> ModelLimit | None:
    """Return the row/cell caps for `model_path`, or None if unknown."""
    api_settings = ServiceClient.get_settings()
    if api_settings is None:
        return None
    if model_path in _AUTO_MODEL_PATH_ALIASES:
        return api_settings.model_limits[api_settings.default_model_version]
    model_version = model_version_from_path(model_path)
    return model_limit_from_version(model_version, api_settings.model_limits)


def _full_output_row_limit(model_path: str | None) -> int | None:
    """Rows a single `output_type="full"` response may cover, None if unknown."""
    limit = _limit_for_model_path(model_path)
    return limit.test_set_max_rows_w_full_regression_output if limit else None


def _test_row_limit(
    limit: ModelLimit,
    train_rows: int | None,
    inference_config: dict[str, Any] | None,
) -> int:
    """Most test rows one predict call accepts, before the cell and byte limits."""
    max_rows = limit.test_set_max_rows
    subsample = (inference_config or {}).get("SUBSAMPLE_SAMPLES")
    # The server validates context-dependent limits for subsampled requests.
    if train_rows and subsample is None:
        max_rows = min(max_rows, limit.predict_row_pairs_budget // train_rows)
    return max_rows


def validate_test_set(
    X: pd.DataFrame | np.ndarray,
    output_type: str | None,
    model_path: str | None = None,
    train_rows: int | None = None,
    split_full_output: bool = False,
    inference_config: dict[str, Any] | None = None,
):
    """Check the integrity of the test data.

    Args:
        X: The test input data.
        output_type: The requested prediction output type.
        model_path: The model the data is predicted with; selects the limits.
        train_rows: Rows of the fitted train set. Caps the test rows through the
            server's row-pairs budget unless `inference_config` subsamples.
        split_full_output: Whether the caller honours the full-output row cap by
            splitting the request, so that cap is not enforced here.
        inference_config: The estimator's inference config.

    Raises:
        ValueError: If `X` exceeds the model's row, column or cell limits.
    """
    limit = _limit_for_model_path(model_path)
    if limit is None:
        return

    max_rows = _test_row_limit(limit, train_rows, inference_config)
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
    if output_type == "full" and not split_full_output:
        if X.shape[0] > limit.test_set_max_rows_w_full_regression_output:
            raise ValueError(
                f"The number of test rows ({X.shape[0]}) exceeds the maximum of {limit.test_set_max_rows_w_full_regression_output} "
                "for full regression output."
            )


def _restore_masked_logits(logits: np.ndarray) -> np.ndarray:
    """Return `logits` with bars outside a row's support back at -inf.

    The response encoding carries no representation for -inf, so those bars
    arrive as null and land in the array as NaN.
    """
    logits = np.asarray(logits, dtype=float)
    return np.where(np.isnan(logits), -np.inf, logits)


def _row_slice(X: Any, start: int, stop: int) -> Any:
    """Return rows `[start:stop)` of `X`, keeping its container type."""
    if isinstance(X, pd.DataFrame):
        return X.iloc[start:stop]
    return X[start:stop]


# Axis along which each `output_type="full"` array runs over test rows. None
# marks a row-independent array: `borders` describes the histogram of the
# fitted target, so every chunk returns the same one.
_FULL_OUTPUT_ROW_AXIS: dict[str, int | None] = {
    "mean": 0,
    "median": 0,
    "mode": 0,
    "logits": 0,
    "quantiles": 1,
    "borders": None,
}


def _merge_full_outputs(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stitch per-chunk full-output predictions back into one result."""
    unknown = sorted(set(parts[0]) - set(_FULL_OUTPUT_ROW_AXIS))
    if unknown:
        raise RuntimeError(
            f"Cannot combine full regression output across calls: the server "
            f"returned unrecognised field(s) {unknown}. Upgrade tabpfn-client, "
            f"or split the test set yourself and merge the results."
        )
    merged: dict[str, np.ndarray] = {}
    for key, axis in _FULL_OUTPUT_ROW_AXIS.items():
        if key not in parts[0]:
            continue
        merged[key] = (
            parts[0][key]
            if axis is None
            else np.concatenate([p[key] for p in parts], axis=axis)
        )
    return merged


@overload
def _clean_text_features(X: pd.DataFrame) -> pd.DataFrame: ...
@overload
def _clean_text_features(X: np.ndarray) -> np.ndarray: ...
def _clean_text_features(X):
    """Clean text features in the input data.

    Removes commas and collapses whitespace in text columns so the data
    serializes cleanly, and truncates text to 2500 characters. Numeric arrays
    are returned unchanged and torch tensors are converted to numpy.
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
        # Only object columns hold text. Other dtypes, such as periods, make
        # pd.to_numeric raise TypeError.
        if df[col].dtype != object:
            continue
        # check if we can't convert to float
        try:
            pd.to_numeric(df[col])
        except ValueError:
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


def _build_fit_task_config(tabpfn_config: TabPFNConfig) -> FitTaskConfig:
    """Wrap a classifier or regressor model config in its fit task config."""
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
    """Return the server-side systems to run for the given mode.

    Raises:
        ValueError: If both `paper_version` and `thinking_mode` are set.
    """
    if paper_version and thinking_mode:
        raise ValueError(
            "Paper version and thinking mode cannot be enabled at the same time"
        )
    if paper_version:
        return []
    if thinking_mode:
        return ["preprocessing", "text", "thinking"]
    return ["preprocessing", "text"]


def _validate_output_type(output_type: str, allowed: Any) -> None:
    """Check `output_type` against the `Literal` of the outputs an estimator predicts.

    Raises:
        ValueError: If `output_type` is not among them.
    """
    choices = get_args(allowed)
    if output_type not in choices:
        raise ValueError(
            f"output_type must be one of {', '.join(map(repr, choices))}, "
            f"not {output_type!r}."
        )


def _resolve_thinking_mode(enabled: bool, effort: str | None = None) -> bool:
    """Return whether thinking is enabled, either explicitly or by setting an effort."""
    # To honour previous contract, setting effort alone is enough to enable thinking.
    if enabled:
        return True
    if effort:
        return True
    return False


def _fit_api_mode(api_mode: ApiMode, *, submit: bool, thinking: bool) -> ApiMode:
    """Return the mode a fit runs in, for `fit()` or, with `submit`, `submit_fit()`.

    Raises:
        ValueError: If `api_mode` is not a valid mode.
    """
    # Checked on every path, so that an invalid mode fails before any upload,
    # not at the first predict() after a billed fit.
    api_mode = ApiMode(api_mode)
    # Only an ASYNC fit is accepted before it runs, as submit needs. A thinking
    # fit can take most of an hour, too long to hold one request open for.
    if submit or thinking:
        return ApiMode.ASYNC
    return api_mode


def _build_thinking_config(
    X: pd.DataFrame | np.ndarray,
    *,
    enabled: bool,
    effort: str | None = None,
    timeout_secs: float | None = None,
    metric: str | None = None,
    group_col: str | list[str] | None = None,
    time_col: str | None = None,
    group_time_col: str | None = None,
) -> ThinkingConfig | None:
    """Validate the group columns and build the thinking config, or None if disabled."""
    _validate_group_columns(
        X,
        enabled=enabled,
        group_col=group_col,
        time_col=time_col,
        group_time_col=group_time_col,
    )
    if not enabled:
        return None
    return ThinkingConfig(
        effort=effort,
        timeout_secs=timeout_secs,
        metric=metric,
        group_col=group_col,
        time_col=time_col,
        group_time_col=group_time_col,
    )


def _validate_group_columns(
    X: Any,
    *,
    enabled: bool,
    group_col: str | list[str] | None,
    time_col: str | None,
    group_time_col: str | None,
) -> None:
    """Check `group_col`, `time_col` and `group_time_col` against `X` and thinking mode.

    The server reads these columns by name from the data passed to `fit` and to
    `predict`, so `X` must be a DataFrame that holds them. For now only thinking mode
    uses them.

    Raises:
        ValueError: A column is named without thinking mode, the combination is
            invalid, `X` is not a DataFrame, or `X` lacks a named column.
    """
    group_cols = [group_col] if isinstance(group_col, str) else list(group_col or [])
    named = [*group_cols, *(c for c in (time_col, group_time_col) if c is not None)]
    if not named:
        return
    if not enabled:
        raise ValueError(
            "`group_col`, `time_col` and `group_time_col` are only supported in "
            "thinking mode. Set `thinking_mode=True`."
        )
    if group_col is not None and time_col is not None:
        raise ValueError("`group_col` and `time_col` cannot be combined.")
    if group_time_col is not None and group_col is None:
        raise ValueError("`group_time_col` requires `group_col`.")
    if not isinstance(X, pd.DataFrame):
        raise ValueError(
            "`group_col`, `time_col` and `group_time_col` name columns, so X must be "
            f"a pandas DataFrame; got {type(X).__name__}."
        )
    missing = [col for col in named if col not in X.columns]
    if missing:
        raise ValueError(f"X is missing the columns {missing}.")
