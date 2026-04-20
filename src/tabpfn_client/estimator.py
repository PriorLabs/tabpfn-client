#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import logging
import sys
import time
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Literal, Optional, Union
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
)
from tabpfn_client.config import Config, init
from tabpfn_client.constants import (
    ci_mode_enabled,
    URL_TABPFN_EXTENSIONS_GITHUB_MANY_CLASS_CODE,
    ModelVersion,
)
from tabpfn_client.service_wrapper import InferenceClient

try:
    from torch import Tensor

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

# Special string used to identify v2.5 models in model paths.
V_2_5_IDENTIFIER = "v2.5"

DEFAULT_V2_MODEL_PATH = "v2_default"
DEFAULT_V2_5_MODEL_PATH = "v2.5_default"


class TabPFNModelSelection:
    """Base class for TabPFN model selection and path handling."""

    _AVAILABLE_MODELS: list[str] = []
    _VALID_TASKS = {"classification", "regression"}

    @classmethod
    def list_available_models(cls) -> list[str]:
        return cls._AVAILABLE_MODELS

    @classmethod
    def _validate_model_name(cls, model_name: str) -> None:
        if model_name != "default" and model_name not in cls._AVAILABLE_MODELS:
            raise ValueError(
                f"Invalid model name: {model_name}. "
                f"Available models are: {cls.list_available_models()}"
            )

    @classmethod
    def _model_name_to_path(
        cls, task: Literal["classification", "regression"], model_name: str
    ) -> Optional[str]:
        cls._validate_model_name(model_name)
        model_name_task = "classifier" if task == "classification" else "regressor"
        # Let the server handle the default model. This enables v2.5 as well.
        if model_name == "default":
            return None
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
        options = {
            "n_estimators": 8,
            "softmax_temperature": 0.9,
        }
        if version == ModelVersion.V2:
            options["model_path"] = DEFAULT_V2_MODEL_PATH
        elif version == ModelVersion.V2_5:
            options["model_path"] = DEFAULT_V2_5_MODEL_PATH
        else:
            raise ValueError(f"Unknown version: {version}")

        options.update(overrides)

        return cls(**options)

    def _get_estimator_params_with_model_path(
        self, task: Literal["classification", "regression"]
    ) -> Dict:
        """Get estimator parameters with the model_path resolved to full path.

        Parameters
        ----------
        task : {"classification", "regression"}
            The task type to determine the correct model path.

        Returns
        -------
        Dict
            Dictionary of estimator parameters with model_path updated to full path.
        """
        estimator_param = self.get_params()
        estimator_param["model_path"] = self._model_name_to_path(task, self.model_path)
        return estimator_param


class TabPFNClassifier(ClassifierMixin, BaseEstimator, TabPFNModelSelection):
    _AVAILABLE_MODELS = [
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
        "default",
        "gn2p4bpt",
        "llderlii",
        "od3j1g5m",
        "vutqq28w",
        "znskzxi4",
    ]

    def __init__(
        self,
        model_path: str = "default",
        n_estimators: int = 8,
        softmax_temperature: float = 0.9,
        balance_probabilities: bool = False,
        average_before_softmax: bool = False,
        ignore_pretraining_limits: bool = True,
        inference_precision: Literal["autocast", "auto"] = "auto",
        random_state: Optional[
            Union[int, np.random.RandomState, np.random.Generator]
        ] = None,
        inference_config: Optional[Dict] = None,
        paper_version: bool = False,
    ):
        """Construct a TabPFN classifier.

        This constructs a classifier using the latest model and settings. If you would
        like to use a previous model version, use `create_default_for_version()`
        instead. You can also use `model_path` to specify a particular model

        Parameters
        ----------
        model_path: str, default="default"
            The name of the model to use.
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
        inference_precision: "autocast" or "auto", default="auto"
            The precision to use for inference. This can dramatically affect the
            speed and reproducibility of the inference.
        random_state: int or RandomState or RandomGenerator or None, default=None
            Controls the randomness of the model. Pass an int for reproducible results.
        inference_config: dict or None, default=None
            Additional advanced arguments for model interface. See the doc of InferenceConfig
            in the tabpfn package for more details. For the client, the inference_config and the
            preprocess transforms need to be dictionaries.
        paper_version: bool, default=False
            If True, will use the model described in the paper, instead of the newest
            version available on the API, which e.g handles text features better.
        """
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
        self.last_trace_id = None
        self.last_fitted_train_set_id = None
        self.last_train_X = None
        self.last_train_y = None
        self.last_meta = {}
        self.last_train_set_description = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        description: str | None = None,
        client_options: ClientOptions | None = None,
    ):
        # assert init() is called
        init()

        validate_train_set(X, y)
        X = _clean_text_features(X)
        self._validate_targets_and_classes(y)
        _check_paper_version(self.paper_version, X)

        estimator_param = self._get_estimator_params_with_model_path("classification")
        if Config.use_server:
            client_options = client_options or ClientOptions()
            if "sentry-trace" not in client_options.headers:
                client_options.headers["sentry-trace"] = uuid4().hex

            self.last_trace_id = client_options.headers["sentry-trace"]
            self.last_train_set_description = description

            def fit_task() -> UUID:
                return InferenceClient.fit(
                    X,
                    y,
                    task="classification",
                    tabpfn_config=estimator_param,
                    description=description,
                    client_options=client_options,
                )

            self.last_fitted_train_set_id = run_task(fit_task, "Fitting")
            self.last_train_X = X
            self.last_train_y = y
            self.fitted_ = True
        else:
            raise NotImplementedError(
                "Only server mode is supported at the moment for init(use_server=False)"
            )
        return self

    def predict(
        self,
        X,
        client_options: ClientOptions | None = None,
    ):
        """Predict class labels for samples in X.

        Args:
            X: The input samples.

        Returns:
            The predicted class labels.
        """
        return self._predict(
            X,
            output_type="preds",
            client_options=client_options,
        )

    def predict_proba(
        self,
        X,
        client_options: ClientOptions | None = None,
    ):
        """Predict class probabilities for X.

        Args:
            X: The input samples.

        Returns:
            The class probabilities of the input samples.
        """
        return self._predict(
            X,
            output_type="probas",
            client_options=client_options,
        )

    def _predict(
        self,
        X,
        output_type,
        client_options: ClientOptions | None = None,
    ) -> dict[str, np.ndarray]:
        check_is_fitted(self)
        validate_test_set(X, output_type)
        _check_paper_version(self.paper_version, X)
        X = _clean_text_features(X)

        estimator_param = self._get_estimator_params_with_model_path("classification")

        client_options = client_options or ClientOptions()
        if "sentry-trace" not in client_options.headers:
            client_options.headers["sentry-trace"] = self.last_trace_id

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
                        X,
                        fitted_train_set_id=self.last_fitted_train_set_id,
                        task="classification",
                        tabpfn_config=estimator_param,
                        predict_params={"output_type": output_type},
                        client_options=client_options,
                    )
                except NeedsRefittingError as exc:
                    last_exc = exc
                    refit_attempts += 1
                    self.last_fitted_train_set_id = InferenceClient.fit(
                        self.last_train_X,
                        self.last_train_y,
                        task="classification",
                        tabpfn_config=estimator_param,
                        description=self.last_train_set_description,
                        client_options=client_options,
                        is_refitting=True,
                    )

        result = run_task(predict_task, "Predicting")
        # Unpack and store metadata
        self.last_meta = result.metadata

        return result.y_pred

    def _validate_targets_and_classes(self, y) -> np.ndarray:
        y_ = column_or_1d(y, warn=True)
        if sum(pd.isnull(y_)) > 0:
            raise ValueError("Input y contains NaN.")
        check_classification_targets(y)
        # Get classes and encode before type conversion to guarantee correct class labels.
        # TODO: should pass this from the server
        self.classes_ = np.unique(y_)
        # TODO: these things should ideally be shared with the local package
        limits = ServiceClient.get_dataset_limits()
        if limits is not None:
            if len(self.classes_) > limits.dataset_max_classes:
                raise ValueError(
                    f"Number of classes {len(self.classes_)} exceeds the maximal number of "
                    f"{limits.dataset_max_classes} classes supported by TabPFN. Consider using "
                    "the many_class extension to reduce the number of classes. For code see "
                    f"{URL_TABPFN_EXTENSIONS_GITHUB_MANY_CLASS_CODE}"
                )


class TabPFNRegressor(RegressorMixin, BaseEstimator, TabPFNModelSelection):
    _AVAILABLE_MODELS = [
        DEFAULT_V2_5_MODEL_PATH,
        "v2.5_low-skew",
        "v2.5_quantiles",
        "v2.5_real-variant",
        "v2.5_real",
        "v2.5_small-samples",
        "v2.5_variant",
        DEFAULT_V2_MODEL_PATH,
        "default",
        "2noar4o2",
        "5wof9ojf",
        "09gpqh39",
        "wyl4o83o",
    ]

    def __init__(
        self,
        model_path: str = "default",
        n_estimators: int = 8,
        softmax_temperature: float = 0.9,
        average_before_softmax: bool = False,
        ignore_pretraining_limits: bool = False,
        inference_precision: Literal["autocast", "auto"] = "auto",
        random_state: Optional[
            Union[int, np.random.RandomState, np.random.Generator]
        ] = None,
        inference_config: Optional[Dict] = None,
        paper_version: bool = False,
    ):
        """Construct a TabPFN regressor.

        This constructs a regressor using the latest model and settings. If you would
        like to use a previous model version, use `create_default_for_version()`
        instead. You can also use `model_path` to specify a particular model.

        Parameters
        ----------
        model_path: str, default="default"
            The name to the model to use.
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
        random_state: int or RandomState or RandomGenerator or None, default=None
            Controls the randomness of the model. Pass an int for reproducible results.
        inference_config: dict or None, default=None
            Additional advanced arguments for model interface. See the doc of InferenceConfig
            in the tabpfn package for more details. For the client, the inference_config and the
            preprocess transforms need to be dictionaries.
        paper_version: bool, default=False
            If True, will use the model described in the paper, instead of the newest
            version available on the API, which e.g handles text features better.
        """
        self.model_path = model_path
        self.n_estimators = n_estimators
        self.softmax_temperature = softmax_temperature
        self.average_before_softmax = average_before_softmax
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.inference_precision = inference_precision
        self.random_state = random_state
        self.inference_config = inference_config
        self.paper_version = paper_version
        self.last_trace_id = None
        self.last_fitted_train_set_id = None
        self.last_train_X = None
        self.last_train_y = None
        self.last_meta = {}
        self.last_train_set_description = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        description: str | None = None,
        client_options: ClientOptions | None = None,
    ):
        # assert init() is called
        init()

        validate_train_set(X, y)
        self._validate_targets(y)
        X = _clean_text_features(X)
        _check_paper_version(self.paper_version, X)

        estimator_param = self._get_estimator_params_with_model_path("regression")
        if Config.use_server:
            client_options = client_options or ClientOptions()
            if "sentry-trace" not in client_options.headers:
                client_options.headers["sentry-trace"] = uuid4().hex

            self.last_trace_id = client_options.headers["sentry-trace"]
            self.last_train_set_description = description

            def fit_task() -> UUID:
                return InferenceClient.fit(
                    X,
                    y,
                    task="regression",
                    tabpfn_config=estimator_param,
                    description=description,
                    client_options=client_options,
                )

            self.last_fitted_train_set_id = run_task(fit_task, "Fitting")
            self.last_train_X = X
            self.last_train_y = y
            self.fitted_ = True
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
        quantiles: Optional[list[float]] = None,
        client_options: ClientOptions | None = None,
    ) -> Union[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]:
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
        check_is_fitted(self)
        validate_test_set(X, output_type)
        X = _clean_text_features(X)
        _check_paper_version(self.paper_version, X)

        # Add new parameters
        predict_params = {
            "output_type": output_type,
            "quantiles": quantiles,
        }

        estimator_param = self._get_estimator_params_with_model_path("regression")

        client_options = client_options or ClientOptions()
        if "sentry-trace" not in client_options.headers:
            client_options.headers["sentry-trace"] = self.last_trace_id

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
                        X,
                        fitted_train_set_id=self.last_fitted_train_set_id,
                        task="regression",
                        tabpfn_config=estimator_param,
                        predict_params=predict_params,
                        client_options=client_options,
                    )
                except NeedsRefittingError as exc:
                    last_exc = exc
                    refit_attempts += 1
                    self.last_fitted_train_set_id = InferenceClient.fit(
                        self.last_train_X,
                        self.last_train_y,
                        task="regression",
                        tabpfn_config=estimator_param,
                        description=self.last_train_set_description,
                        client_options=client_options,
                        is_refitting=True,
                    )

        result = run_task(predict_task, "Predicting")
        # Unpack and store metadata
        self.last_meta = result.metadata

        output = result.y_pred
        if output_type == "full":
            try:
                from tabpfn.regressor import FullSupportBarDistribution
                import torch

                output["criterion"] = FullSupportBarDistribution(
                    borders=torch.tensor(output["borders"])
                )
            except ImportError:
                logger.warning(
                    "Optional dependencies 'tabpfn' and 'torch' are required to "
                    "construct the criterion when output_type='full'. Skipping criterion."
                )

        return output

    def _validate_targets(self, y) -> np.ndarray:
        y_ = column_or_1d(y, warn=True)
        if sum(pd.isnull(y_)) > 0:
            raise ValueError("Input y contains NaN.")


def validate_train_set(X: np.ndarray, y: Union[np.ndarray, None] = None):
    """Check the integrity of the training data."""

    # check if the number of samples is consistent (ValueError)
    if y is not None:
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")

    limits = ServiceClient.get_dataset_limits()
    if limits is None:
        return

    if X.shape[0] > limits.train_set_max_rows:
        raise ValueError(
            f"The number of train rows ({X.shape[0]}) exceeds the maximum of {limits.train_set_max_rows}."
        )
    if X.shape[1] > limits.dataset_max_cols:
        raise ValueError(
            f"The number of train columns ({X.shape[1]}) exceeds the maximum of {limits.dataset_max_cols}."
        )
    n_cells = X.shape[0] * X.shape[1]
    if n_cells > limits.train_set_max_cells:
        raise ValueError(
            f"The number of train cells ({n_cells}) exceeds the maximum of {limits.train_set_max_cells}."
        )


def validate_test_set(X: np.ndarray, output_type: str):
    """Check the integrity of the test data."""

    limits = ServiceClient.get_dataset_limits()
    if limits is None:
        return

    if X.shape[0] > limits.test_set_max_rows:
        raise ValueError(
            f"The number of test rows ({X.shape[0]}) exceeds the maximum of {limits.test_set_max_rows}. "
            "Split the test set across multiple calls to reduce the number of rows."
        )
    if X.shape[1] > limits.dataset_max_cols:
        raise ValueError(
            f"The number of test columns ({X.shape[1]}) exceeds the maximum of {limits.dataset_max_cols}."
        )
    n_cells = X.shape[0] * X.shape[1]
    if n_cells > limits.test_set_max_cells:
        raise ValueError(
            f"The number of test cells ({n_cells}) exceeds the maximum of {limits.test_set_max_cells}. "
            "Split the test set across multiple calls to reduce the number of cells."
        )
    if output_type == "full":
        if X.shape[0] > limits.test_set_max_rows_w_full_regression_output:
            raise ValueError(
                f"The number of test rows ({X.shape[0]}) exceeds the maximum of {limits.test_set_max_rows_w_full_regression_output} "
                "for full regression output."
            )


def _check_paper_version(paper_version, X):
    pass


def _clean_text_features(X):
    """
    Clean text features in the input data. This is used to avoid
    serialization errors, which happens when the input data contains
    commas or weird spaces, and to limit the length of the text features.
    """
    # Convert numpy array to pandas DataFrame if necessary
    # not necessary if numpy array of numbers
    if TORCH_AVAILABLE and isinstance(X, Tensor):
        if X.requires_grad:
            X = X.detach()
        if X.is_cuda:
            X = X.cpu()

        X = X.numpy()

    if isinstance(X, np.ndarray):
        if np.issubdtype(X.dtype, np.number):
            return X
        else:
            X_ = pd.DataFrame(X.copy())
    else:
        X_ = X.copy()

    # limit to 2500 chars and remove commas for text features
    for col in X_.columns:
        # check if we can't convert to float
        try:
            pd.to_numeric(X_[col])
        except ValueError:
            if X_[col].dtype == object:  # only process string/object columns
                X_[col] = (
                    X_[col]
                    .str.replace(",", "")
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                    .str.slice(0, 2500)
                )

    # Convert back to numpy if input was numpy (or tensor that was converted to numpy)
    if isinstance(X, np.ndarray):
        return X_.to_numpy()
    return X_


def run_task(task: Callable, message: str, with_spinner: bool = True) -> Any:
    if not with_spinner or ci_mode_enabled():
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
