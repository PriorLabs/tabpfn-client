# Generated with codegen tooling. Do not edit by hand.

from typing import Literal, Sequence, TypedDict
from typing_extensions import NotRequired


class BaseTabPFNConfigDict(TypedDict):
    n_estimators: NotRequired[int]
    categorical_features_indices: NotRequired[Sequence[int]]
    softmax_temperature: NotRequired[float]
    average_before_softmax: NotRequired[bool]
    random_state: NotRequired[int]
    inference_config: NotRequired[dict]
    inference_precision: NotRequired[Literal["autocast", "auto"]]
    ignore_pretraining_limits: bool
    n_preprocessing_jobs: int
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache", "batched"]
    device: NotRequired[str]
    memory_saving_mode: NotRequired[bool]


class ClassifierTabPFNConfigDict(BaseTabPFNConfigDict):
    model_path: NotRequired[str]
    balance_probabilities: NotRequired[bool]


class RegressorTabPFNConfigDict(BaseTabPFNConfigDict):
    model_path: NotRequired[str]
