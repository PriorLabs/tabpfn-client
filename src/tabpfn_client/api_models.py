# Generated code. Do not edit by hand.
#
# Requires the `eval-type-backport` package on Python < 3.10: pydantic uses it
# to evaluate the `X | Y` annotations under `from __future__ import annotations`.
#
# Forward-compat note: enum-typed fields are widened to `EnumName | UnknownEnum`
# (and `Literal[...] | str` for inline literals) so the SDK does not
# reject response payloads when the server adds a new enum value. Known
# values still deserialize to the enum member; unrecognized values flow
# through as `UnknownEnum` (a `str` subclass) instead of raising a
# ValidationError. Widened fields are wrapped in
# `Annotated[..., Field(union_mode="left_to_right")]` because pydantic's
# default smart mode would land known values in the wider branch.
# Discriminator `const` fields are intentionally left non-forward-compatible.
#
# Nullified-defaults note: every non-required field is projected as
# `<type> | None = None`. The server's defaults (literal or default_factory)
# are intentionally dropped so an omitted (None) value lets the server apply
# its own. `const` discriminator fields keep their fixed value.

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Union
from uuid import UUID

from pydantic import BaseModel, Field


class UnknownEnum(str):
    """Sentinel for enum values not known to this SDK — see header."""

    @property
    def value(self) -> str:
        return str(self)

    @classmethod
    def __get_pydantic_core_schema__(cls, _source, _handler):
        from pydantic_core import core_schema

        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())


DatasetFileType = Literal["csv", "parquet"]


TabPFNSystem = Literal["preprocessing", "text", "thinking"]


ThinkingEffort = Literal["medium", "high"]


class AsyncSettings(BaseModel):
    use_above_trainset_size_bytes: int
    poll_timeout_secs: float


class ClassifierOutputType(str, Enum):
    PROBAS = "probas"
    PREDS = "preds"
    TOP_K = "top_k"


class ClassifierPredictParams(BaseModel):
    output_type: (
        Annotated[ClassifierOutputType | UnknownEnum, Field(union_mode="left_to_right")] | None
    ) = None
    top_k: int | None = None


class FitMode(str, Enum):
    FIT_PREPROCESSORS = "fit_preprocessors"
    FIT_WITH_CACHE = "fit_with_cache"


class ClassifierTabPFNConfig(BaseModel):
    n_estimators: int | None = None
    categorical_features_indices: list[int] | None = None
    softmax_temperature: float | None = None
    average_before_softmax: bool | None = None
    random_state: int | None = None
    inference_config: dict[str, Any] | None = Field(
        default=None, description="Refer to tabpfn.inference_config.InferenceConfig for more details."
    )
    inference_precision: (
        Annotated[Literal["autocast", "auto"] | str, Field(union_mode="left_to_right")] | None
    ) = None
    ignore_pretraining_limits: bool | None = None
    fit_mode: Annotated[FitMode | UnknownEnum, Field(union_mode="left_to_right")] | None = None
    model_path: str | None = None
    balance_probabilities: bool | None = None


class PredictionTask(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ClassifierConfig(BaseModel):
    task: Literal[PredictionTask.CLASSIFICATION] = PredictionTask.CLASSIFICATION
    tabpfn_config: ClassifierTabPFNConfig | None = None
    predict_params: ClassifierPredictParams | None = None


class ClassifierFitTaskConfig(BaseModel):
    task: Literal[PredictionTask.CLASSIFICATION] = PredictionTask.CLASSIFICATION
    tabpfn_config: ClassifierTabPFNConfig | None = None


class ClassifierMetadata(BaseModel):
    test_set_num_rows: int
    test_set_num_cols: int
    n_estimators: int | None = None
    task: Literal[PredictionTask.CLASSIFICATION] = PredictionTask.CLASSIFICATION
    package_version: str
    tabpfn_config: ClassifierTabPFNConfig
    classes: list[str | int | float | bool] | None = None
    top_k: int | None = None


class FileInfo(BaseModel):
    format: Annotated[DatasetFileType | str, Field(union_mode="left_to_right")]
    hash: str | None = Field(
        default=None, description="The crc32c hash of the file, used to deduplicate the file."
    )
    size_bytes: int | None = Field(
        default=None,
        description="The size of the file in bytes, used to compute the optimal number of chunks when chunking is enabled.",
    )
    use_chunks: bool | None = Field(
        default=None,
        description="Whether to split the the file into chunks and upload them in parallel.",
    )


class FileUploadInfo(BaseModel):
    signed_urls: list[str]
    expires_at: float
    required_headers: dict[str, str]


class FitStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class ModelLimit(BaseModel):
    train_set_max_rows: int
    train_set_max_cells: int
    test_set_max_rows: int
    max_classes: int
    max_cols: int
    test_set_max_rows_w_full_regression_output: int
    predict_row_pairs_budget: int
    test_set_max_cells: int


class ModelVersion(str, Enum):
    V2 = "v2"
    V2_5 = "v2.5"
    V2_6 = "v2.6"
    V3 = "v3"
    V3_5 = "v3.5"


class RegressorOutputType(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    QUANTILES = "quantiles"
    FULL = "full"
    MAIN = "main"


class RegressorPredictParams(BaseModel):
    output_type: (
        Annotated[RegressorOutputType | UnknownEnum, Field(union_mode="left_to_right")] | None
    ) = None
    quantiles: list[float] | None = None


class RegressorTabPFNConfig(BaseModel):
    n_estimators: int | None = None
    categorical_features_indices: list[int] | None = None
    softmax_temperature: float | None = None
    average_before_softmax: bool | None = None
    random_state: int | None = None
    inference_config: dict[str, Any] | None = Field(
        default=None, description="Refer to tabpfn.inference_config.InferenceConfig for more details."
    )
    inference_precision: (
        Annotated[Literal["autocast", "auto"] | str, Field(union_mode="left_to_right")] | None
    ) = None
    ignore_pretraining_limits: bool | None = None
    fit_mode: Annotated[FitMode | UnknownEnum, Field(union_mode="left_to_right")] | None = None
    model_path: str | None = None


class RegressorConfig(BaseModel):
    task: Literal[PredictionTask.REGRESSION] = PredictionTask.REGRESSION
    tabpfn_config: RegressorTabPFNConfig | None = None
    predict_params: RegressorPredictParams | None = None


class RegressorFitTaskConfig(BaseModel):
    task: Literal[PredictionTask.REGRESSION] = PredictionTask.REGRESSION
    tabpfn_config: RegressorTabPFNConfig | None = None


class RegressorMetadata(BaseModel):
    test_set_num_rows: int
    test_set_num_cols: int
    n_estimators: int | None = None
    task: Literal[PredictionTask.REGRESSION] = PredictionTask.REGRESSION
    package_version: str
    tabpfn_config: RegressorTabPFNConfig


class ThinkingConfig(BaseModel):
    effort: Annotated[ThinkingEffort | str, Field(union_mode="left_to_right")] | None = None
    timeout_secs: float | None = None
    metric: str | None = None


Prediction = Union[list[Any], list[list[Any]], dict[str, Union[list[Any], list[list[Any]]]]]


FitTaskConfig = Annotated[
    Union[ClassifierFitTaskConfig, RegressorFitTaskConfig], Field(discriminator="task")
]


Metadata = Annotated[Union[ClassifierMetadata, RegressorMetadata], Field(discriminator="task")]


TaskConfig = Annotated[Union[ClassifierConfig, RegressorConfig], Field(discriminator="task")]


class DuplicateTestSetErrorResponse(BaseModel):
    message: str
    error_code: str = "DUPLICATE_TEST_SET_UPLOAD"
    trace_id: UUID | None = None
    test_set_upload_id: UUID


class DuplicateTrainSetErrorResponse(BaseModel):
    message: str
    error_code: str = "DUPLICATE_TRAIN_SET_UPLOAD"
    trace_id: UUID | None = None
    train_set_upload_id: UUID


class FitRequest(BaseModel):
    task_config: FitTaskConfig
    tabpfn_systems: list[Annotated[TabPFNSystem | str, Field(union_mode="left_to_right")]] | None = None
    thinking_config: ThinkingConfig | None = None
    train_set_upload_id: UUID


class FitResponse(BaseModel):
    fitted_train_set_id: UUID
    status: Annotated[FitStatus | UnknownEnum, Field(union_mode="left_to_right")]


class GetFitStatusRequest(BaseModel):
    fitted_train_set_id: UUID


class GetFitStatusResponse(BaseModel):
    fitted_train_set_id: UUID
    status: Annotated[FitStatus | UnknownEnum, Field(union_mode="left_to_right")]
    retry_in_secs: float | None = None
    error: str | None = None
    error_code: str | None = None


class GetSettingsResponse(BaseModel):
    default_model_version: Annotated[ModelVersion | UnknownEnum, Field(union_mode="left_to_right")]
    max_model_limit: ModelLimit
    model_limits: dict[
        Annotated[ModelVersion | UnknownEnum, Field(union_mode="left_to_right")], ModelLimit
    ]
    dataset_max_size_bytes: int
    async_settings: AsyncSettings


class NotFoundErrorResponse(BaseModel):
    message: str
    error_code: str = "NOT_FOUND"
    trace_id: UUID | None = None


class PredictRequest(BaseModel):
    test_set_upload_id: UUID
    fitted_train_set_id: UUID
    task_config: TaskConfig


class PredictResponse(BaseModel):
    prediction: Prediction
    metadata: Metadata


class PrepareTestSetUploadRequest(BaseModel):
    fitted_train_set_id: UUID
    x_test_info: FileInfo
    force_reupload: bool | None = Field(
        default=None,
        description="Whether to force the upload of the file even if a file with the same hash already exists.",
    )


class PrepareTestSetUploadResponse(BaseModel):
    test_set_upload_id: UUID
    x_test_info: FileUploadInfo


class PrepareTrainSetUploadRequest(BaseModel):
    x_train_info: FileInfo
    y_train_info: FileInfo
    description: str | None = None
    force_reupload: bool | None = Field(
        default=None,
        description="Whether to force the upload of the file even if a file with the same hash already exists.",
    )


class PrepareTrainSetUploadResponse(BaseModel):
    train_set_upload_id: UUID
    x_train_info: FileUploadInfo
    y_train_info: FileUploadInfo


class SubmitFitJobRequest(BaseModel):
    task_config: FitTaskConfig
    tabpfn_systems: list[Annotated[TabPFNSystem | str, Field(union_mode="left_to_right")]] | None = None
    thinking_config: ThinkingConfig | None = None
    train_set_upload_id: UUID


class SubmitFitJobResponse(BaseModel):
    fitted_train_set_id: UUID
