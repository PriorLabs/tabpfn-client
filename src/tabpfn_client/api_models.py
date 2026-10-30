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


TextHandling = Literal["advanced", "simple"]


ThinkingEffort = Literal["micro", "mini", "low", "medium", "high", "xhigh", "max"]


class PredictAsyncThresholds(BaseModel):
    train_rows: int
    test_rows: int


class AsyncSettings(BaseModel):
    use_above_trainset_size_bytes: int
    poll_timeout_secs: float
    predict_use_above: PredictAsyncThresholds
    thinking_predict_use_above: PredictAsyncThresholds


class ClassifierOutputType(str, Enum):
    PROBAS = "probas"
    PREDS = "preds"
    TOP_K = "top_k"


class ClassifierPredictParams(BaseModel):
    output_type: (
        Annotated[ClassifierOutputType | UnknownEnum, Field(union_mode="left_to_right")] | None
    ) = Field(
        default=None,
        description="What `prediction` contains:\n\n* `probas`: one probability for each class, for each row.\n* `preds`: one label for each row.\n* `top_k`: the most probable classes for each row.",
    )
    top_k: int | None = Field(
        default=None,
        description="Number of classes that `top_k` returns for each row. Required with `output_type=top_k`. Other output types do not permit it.",
    )


class FitMode(str, Enum):
    LOW_MEMORY = "low_memory"
    FIT_PREPROCESSORS = "fit_preprocessors"
    FIT_WITH_CACHE = "fit_with_cache"
    BATCHED = "batched"


class ClassifierTabPFNConfig(BaseModel):
    n_estimators: int | None = Field(
        default=None,
        description="Ensemble size. The model averages this number of forward passes, and each pass uses a different preprocessed view of the data. If unset, TabPFN selects the size. It selects 8 for most data. For wide tables, it selects more, so that each column gets to at least one ensemble member.",
    )
    categorical_features_indices: list[int] | None = Field(
        default=None,
        description="0-based indices of the categorical columns. If unset, the model finds the categorical columns.",
    )
    softmax_temperature: float | None = Field(
        default=None,
        description="A lower value makes the predicted probabilities more confident. The value 1.0 has no effect. If unset, the model uses its default value.",
    )
    average_before_softmax: bool | None = Field(
        default=None,
        description="If true, the model averages the logits of the ensemble members before the softmax. If false, it averages their probabilities after the softmax.",
    )
    random_state: int | None = Field(
        default=None,
        description="Random seed. The same seed gives the same results. If unset, TabPFN uses its default seed.",
    )
    inference_config: dict[str, Any] | None = Field(
        default=None,
        description="Advanced inference settings. See `tabpfn.inference_config.InferenceConfig` in the TabPFN package.",
    )
    inference_precision: (
        Annotated[Literal["autocast", "auto"] | str, Field(union_mode="left_to_right")] | None
    ) = Field(
        default=None,
        description="`autocast` always uses mixed precision. `auto` selects the precision from the hardware.",
    )
    ignore_pretraining_limits: bool | None = Field(
        default=None,
        description="If true, the server accepts data outside the size ranges of the model's pre-training, and predictions on that data may be worse. If false, the server rejects that data.",
    )
    fit_mode: Annotated[FitMode | UnknownEnum, Field(union_mode="left_to_right")] | None = Field(
        default=None,
        description="How the model prepares and keeps the fit. With `fit_with_cache`, the server keeps the fit, and later requests can use it again. Not all servers support `fit_with_cache`.",
    )
    model_path: str | None = Field(
        default=None,
        description="The checkpoint, by file name or model name. If unset, the server uses its default checkpoint.",
    )
    balance_probabilities: bool | None = Field(
        default=None,
        description="If true, the model adjusts the probabilities for the class distribution of the training data. This can help with imbalanced classes when the metric ignores class imbalance, such as balanced accuracy.",
    )


class PredictionTask(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ClassifierConfig(BaseModel):
    task: Literal[PredictionTask.CLASSIFICATION] = Field(
        default=PredictionTask.CLASSIFICATION, description="The type of prediction."
    )
    tabpfn_config: ClassifierTabPFNConfig | None = Field(
        default=None, description="Model settings for the fit."
    )
    predict_params: ClassifierPredictParams | None = Field(
        default=None, description="What the prediction returns."
    )


class ClassifierFitTaskConfig(BaseModel):
    task: Literal[PredictionTask.CLASSIFICATION] = PredictionTask.CLASSIFICATION
    tabpfn_config: ClassifierTabPFNConfig | None = None


class ClassifierMetadata(BaseModel):
    test_set_num_rows: int = Field(description="Number of rows that the model predicted.")
    test_set_num_cols: int = Field(
        description="Number of columns after encoding. This number can be different from the number of columns that you sent."
    )
    n_estimators: int | None = Field(
        default=None, description="Ensemble size that the request used. Null when no ensemble ran."
    )
    billing_model_version: str | None = Field(
        default=None,
        description="The model version for the usage count of the request. Null when not reported.",
    )
    execution_mode: (
        Annotated[Literal["standard", "thinking", "cache"] | str, Field(union_mode="left_to_right")]
        | None
    ) = Field(default=None, description="How the server ran the request. Null when not reported.")
    cache_outcome: (
        Annotated[
            Literal["not_requested", "hit", "miss", "fallback"] | str, Field(union_mode="left_to_right")
        ]
        | None
    ) = Field(default=None, description="Whether the request used a kept fit. Null when not reported.")
    task: Literal[PredictionTask.CLASSIFICATION] = Field(
        default=PredictionTask.CLASSIFICATION, description="The type of prediction."
    )
    package_version: str = Field(description="Version of the TabPFN package that made the prediction.")
    tabpfn_config: ClassifierTabPFNConfig = Field(
        description="The model settings that the request used."
    )
    classes: list[str | int | float | bool] | None = Field(
        default=None,
        description="Class labels in ascending order. They name the columns of `probas` and the labels of `preds`. Null with `top_k`, because `top_k` has its own `prediction.classes`.",
    )
    top_k: int | None = Field(default=None, description="The `top_k` value of the request.")


class ModelVersion(str, Enum):
    V2 = "v2"
    V2_5 = "v2.5"
    V2_6 = "v2.6"
    V3 = "v3"
    V3_5 = "v3.5"
    V3_5_FAST = "v3.5-fast"


class QuotaOperation(str, Enum):
    PREDICT = "predict"
    THINKING_PREDICT = "thinking_predict"
    THINKING_FIT = "thinking_fit"
    CACHE_PREDICT = "cache_predict"


class EstimateCostRequest(BaseModel):
    train_rows: int
    raw_columns: int
    test_rows: int | None = None
    model_version: Annotated[ModelVersion | UnknownEnum, Field(union_mode="left_to_right")] | None = None
    operation: Annotated[QuotaOperation | UnknownEnum, Field(union_mode="left_to_right")] | None = None
    n_estimators: int | None = None
    thinking_effort: Annotated[ThinkingEffort | str, Field(union_mode="left_to_right")] | None = None


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


class FitTimings(BaseModel):
    """Where a fit's time went. Queue wait covers everything outside worker
    processing: waiting for a free worker and transport.
    """

    elapsed_s: float = Field(
        description="Seconds from submission until the fit completed, or until now while it is pending."
    )
    queue_wait_s: float | None = Field(
        default=None, description="Seconds the fit waited for a worker. Set once the fit completed."
    )
    train_set_transform_s: float | None = Field(
        default=None,
        description="Seconds a worker spent loading, validating and preprocessing the train set. Set once the fit completed.",
    )
    fit_s: float | None = Field(
        default=None,
        description="Seconds a worker spent fitting the model; 0 when the model is fitted at predict time. Set once the fit completed.",
    )


class ModelLimit(BaseModel):
    train_set_max_rows: int
    train_set_max_cells: int = Field(
        description="Maximum training cells per estimator: selected rows times uploaded columns. Row subsampling can keep each context below this limit even when the full upload exceeds it."
    )
    train_set_max_upload_cells: int = Field(
        description="Maximum cells in the full training upload before row subsampling. This bounds the table processed before estimator contexts are selected."
    )
    test_set_max_rows: int
    max_classes: int
    max_cols: int
    predict_row_pairs_budget: int
    test_set_max_rows_w_full_regression_output: int
    test_set_max_cells: int


class PredictStatus(str, Enum):
    """Status of a predict job."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class PredictTimings(BaseModel):
    """Where a predict's time went, per stage. Queue wait covers everything
    outside worker processing: waiting for a free worker and transport.
    """

    test_set_transform_queue_wait_s: float | None = Field(
        default=None, description="Seconds the test set transform waited for a worker."
    )
    test_set_transform_s: float | None = Field(
        default=None,
        description="Seconds a worker spent loading, validating and preprocessing the test set.",
    )
    predict_queue_wait_s: float | None = Field(
        default=None, description="Seconds the prediction waited for a worker."
    )
    predict_s: float | None = Field(default=None, description="Seconds a worker spent predicting.")


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
    ) = Field(
        default=None,
        description="What `prediction` contains:\n\n* `mean`, `median` or `mode`: one value for each row.\n* `quantiles`: one list for each quantile.\n* `main`: the mean, median, mode and quantiles.\n* `full`: the full predicted distribution. It accepts fewer rows.",
    )
    quantiles: list[float] | None = Field(
        default=None,
        description="Quantile levels between 0 and 1, for `quantiles` and `main`. If unset, TabPFN uses its default list of 9 levels.",
    )


class RegressorTabPFNConfig(BaseModel):
    n_estimators: int | None = Field(
        default=None,
        description="Ensemble size. The model averages this number of forward passes, and each pass uses a different preprocessed view of the data. If unset, TabPFN selects the size. It selects 8 for most data. For wide tables, it selects more, so that each column gets to at least one ensemble member.",
    )
    categorical_features_indices: list[int] | None = Field(
        default=None,
        description="0-based indices of the categorical columns. If unset, the model finds the categorical columns.",
    )
    softmax_temperature: float | None = Field(
        default=None,
        description="A lower value makes the predicted probabilities more confident. The value 1.0 has no effect. If unset, the model uses its default value.",
    )
    average_before_softmax: bool | None = Field(
        default=None,
        description="If true, the model averages the logits of the ensemble members before the softmax. If false, it averages their probabilities after the softmax.",
    )
    random_state: int | None = Field(
        default=None,
        description="Random seed. The same seed gives the same results. If unset, TabPFN uses its default seed.",
    )
    inference_config: dict[str, Any] | None = Field(
        default=None,
        description="Advanced inference settings. See `tabpfn.inference_config.InferenceConfig` in the TabPFN package.",
    )
    inference_precision: (
        Annotated[Literal["autocast", "auto"] | str, Field(union_mode="left_to_right")] | None
    ) = Field(
        default=None,
        description="`autocast` always uses mixed precision. `auto` selects the precision from the hardware.",
    )
    ignore_pretraining_limits: bool | None = Field(
        default=None,
        description="If true, the server accepts data outside the size ranges of the model's pre-training, and predictions on that data may be worse. If false, the server rejects that data.",
    )
    fit_mode: Annotated[FitMode | UnknownEnum, Field(union_mode="left_to_right")] | None = Field(
        default=None,
        description="How the model prepares and keeps the fit. With `fit_with_cache`, the server keeps the fit, and later requests can use it again. Not all servers support `fit_with_cache`.",
    )
    model_path: str | None = Field(
        default=None,
        description="The checkpoint, by file name or model name. If unset, the server uses its default checkpoint.",
    )


class RegressorConfig(BaseModel):
    task: Literal[PredictionTask.REGRESSION] = Field(
        default=PredictionTask.REGRESSION, description="The type of prediction."
    )
    tabpfn_config: RegressorTabPFNConfig | None = Field(
        default=None, description="Model settings for the fit."
    )
    predict_params: RegressorPredictParams | None = Field(
        default=None, description="What the prediction returns."
    )


class RegressorFitTaskConfig(BaseModel):
    task: Literal[PredictionTask.REGRESSION] = PredictionTask.REGRESSION
    tabpfn_config: RegressorTabPFNConfig | None = None


class RegressorMetadata(BaseModel):
    test_set_num_rows: int = Field(description="Number of rows that the model predicted.")
    test_set_num_cols: int = Field(
        description="Number of columns after encoding. This number can be different from the number of columns that you sent."
    )
    n_estimators: int | None = Field(
        default=None, description="Ensemble size that the request used. Null when no ensemble ran."
    )
    billing_model_version: str | None = Field(
        default=None,
        description="The model version for the usage count of the request. Null when not reported.",
    )
    execution_mode: (
        Annotated[Literal["standard", "thinking", "cache"] | str, Field(union_mode="left_to_right")]
        | None
    ) = Field(default=None, description="How the server ran the request. Null when not reported.")
    cache_outcome: (
        Annotated[
            Literal["not_requested", "hit", "miss", "fallback"] | str, Field(union_mode="left_to_right")
        ]
        | None
    ) = Field(default=None, description="Whether the request used a kept fit. Null when not reported.")
    task: Literal[PredictionTask.REGRESSION] = Field(
        default=PredictionTask.REGRESSION, description="The type of prediction."
    )
    package_version: str = Field(description="Version of the TabPFN package that made the prediction.")
    tabpfn_config: RegressorTabPFNConfig = Field(description="The model settings that the request used.")


class ThinkingConfig(BaseModel):
    effort: Annotated[ThinkingEffort | str, Field(union_mode="left_to_right")] | None = None
    timeout_secs: float | None = None
    metric: str | None = None
    group_col: str | list[str] | None = None
    time_col: str | None = None
    group_time_col: str | None = None


class UsageByTokenEntry(BaseModel):
    api_token_id: str | None
    api_token_name: str | None
    request_count: int
    total_cost: int


class UsageDailyEntry(BaseModel):
    date: str
    total_cost: int
    request_count: int
    thinking_fit_count: int
    cost_by_operation: dict[str, int] | None = Field(
        default=None,
        description="Costs by requested operation (legacy fallback: operation_type), including unknown operations",
    )


class UsageHistoryEntry(BaseModel):
    datetime: str
    cost: int | None
    duration_in_ms: int | None
    error_message: str | None
    error_type: str | None
    operation_type: str
    requested_operation_type: str | None = None
    pricing_version: str | None = None
    api_token_id: str | None
    api_token_name: str | None


Prediction = Union[list[Any], list[list[Any]], dict[str, Union[list[Any], list[list[Any]]]]]


FitTaskConfig = Annotated[
    Union[ClassifierFitTaskConfig, RegressorFitTaskConfig], Field(discriminator="task")
]


Metadata = Annotated[Union[ClassifierMetadata, RegressorMetadata], Field(discriminator="task")]


TaskConfig = Annotated[Union[ClassifierConfig, RegressorConfig], Field(discriminator="task")]


class APIUsageData(BaseModel):
    daily_tokens_used: int | None = None
    daily_token_limit: int | None = None
    monthly_tokens_used: int | None = None
    monthly_token_limit: int | None = None
    current_usage: int
    usage_limit: int
    reset_time: str
    monthly_usage: int
    monthly_limit: int
    monthly_reset_time: str
    daily_usage: int
    daily_limit: int
    daily_reset_time: str
    thinking_fit_count: int
    thinking_fit_limit: int


class DuplicateTestSetErrorResponse(BaseModel):
    message: str = Field(description="What went wrong.")
    error_code: str = "DUPLICATE_TEST_SET_UPLOAD"
    trace_id: UUID | None = Field(default=None, description="Id to quote when reporting the error.")
    test_set_upload_id: UUID


class DuplicateTrainSetErrorResponse(BaseModel):
    message: str = Field(description="What went wrong.")
    error_code: str = "DUPLICATE_TRAIN_SET_UPLOAD"
    trace_id: UUID | None = Field(default=None, description="Id to quote when reporting the error.")
    train_set_upload_id: UUID


class EstimateCostResponse(BaseModel):
    estimated_cost: int = Field(
        description="Tokens the operation is charged. For thinking_fit, the most it can be charged: the effort level's fit time limit at the GPU-second rate, which is reserved at submit. The settled charge is the measured fit time at the same rate, up to this amount and at least the minimum charge."
    )
    pricing_version: str
    inputs: EstimateCostRequest


class FitRequest(BaseModel):
    task_config: FitTaskConfig
    tabpfn_systems: list[Annotated[TabPFNSystem | str, Field(union_mode="left_to_right")]] | None = None
    text_handling: Annotated[TextHandling | str, Field(union_mode="left_to_right")] | None = None
    thinking_config: ThinkingConfig | None = None
    train_set_upload_id: UUID


class FitResponse(BaseModel):
    fitted_train_set_id: UUID
    status: Annotated[FitStatus | UnknownEnum, Field(union_mode="left_to_right")]
    timings: FitTimings | None = None


class GetFitStatusRequest(BaseModel):
    fitted_train_set_id: UUID


class GetFitStatusResponse(BaseModel):
    fitted_train_set_id: UUID
    status: Annotated[FitStatus | UnknownEnum, Field(union_mode="left_to_right")]
    retry_in_secs: float | None = None
    error: str | None = None
    error_code: str | None = None
    timings: FitTimings | None = None


class GetPredictResultRequest(BaseModel):
    fitted_test_set_id: UUID


class GetPredictResultResponse(BaseModel):
    fitted_test_set_id: UUID
    status: Annotated[PredictStatus | UnknownEnum, Field(union_mode="left_to_right")]
    retry_in_secs: float | None = None
    error: str | None = None
    error_code: str | None = None
    prediction: Prediction | None = None
    metadata: Metadata | None = None
    timings: PredictTimings | None = None


class GetSettingsResponse(BaseModel):
    default_model_version: Annotated[ModelVersion | UnknownEnum, Field(union_mode="left_to_right")]
    max_model_limit: ModelLimit
    model_limits: dict[
        Annotated[ModelVersion | UnknownEnum, Field(union_mode="left_to_right")], ModelLimit
    ]
    dataset_max_size_bytes: int
    async_settings: AsyncSettings
    inline_test_set_max_parquet_bytes: int | None = None


class InlinePredictRequest(BaseModel):
    fitted_train_set_id: UUID
    task_config: TaskConfig
    x_test_parquet_base64: str = Field(
        description="The test set as a base64 Parquet file, in place of `test_set_upload_id`. The decoded file can be at most `inline_test_set_max_parquet_bytes` from `get_settings`. If that value is 0, the server does not accept inline test sets."
    )


class NotFoundErrorResponse(BaseModel):
    message: str = Field(description="What went wrong.")
    error_code: str = "NOT_FOUND"
    trace_id: UUID | None = Field(default=None, description="Id to quote when reporting the error.")


class PredictFailedErrorResponse(BaseModel):
    """A prediction that failed after the server created it. get_predict_result
    reports the same failure under its id.
    """

    message: str = Field(description="What went wrong.")
    error_code: str = Field(
        default="GENERAL_ERROR",
        description="Machine-readable error category, such as `BUSY`. `NEEDS_REFITTING` and `KV_CACHE_INVALID` mean the fitted model has to be fit again.",
    )
    trace_id: UUID | None = Field(default=None, description="Id to quote when reporting the error.")
    fitted_test_set_id: UUID = Field(description="Id of the prediction that failed.")


class PredictPendingResponse(BaseModel):
    """A predict still running when the wait for it ended."""

    fitted_test_set_id: UUID
    status: Annotated[PredictStatus | UnknownEnum, Field(union_mode="left_to_right")] | None = None
    retry_in_secs: float


class PredictRequest(BaseModel):
    fitted_train_set_id: UUID
    task_config: TaskConfig
    test_set_upload_id: UUID


class PredictResponse(BaseModel):
    prediction: Prediction
    metadata: Metadata
    timings: PredictTimings | None = None
    fitted_test_set_id: UUID | None = Field(
        default=None, description="Id of the prediction, to get it again with get_predict_result."
    )


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
    text_handling: Annotated[TextHandling | str, Field(union_mode="left_to_right")] | None = None
    thinking_config: ThinkingConfig | None = None
    train_set_upload_id: UUID


class SubmitFitJobResponse(BaseModel):
    fitted_train_set_id: UUID


class SubmitPredictJobRequest(BaseModel):
    fitted_train_set_id: UUID
    task_config: TaskConfig
    test_set_upload_id: UUID


class SubmitPredictJobResponse(BaseModel):
    fitted_test_set_id: UUID


class UsageByTokenResponse(BaseModel):
    entries: list[UsageByTokenEntry]


class UsageDailyResponse(BaseModel):
    entries: list[UsageDailyEntry]


class UsageHistoryResponse(BaseModel):
    entries: list[UsageHistoryEntry]
