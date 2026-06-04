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

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal
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

        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema()
        )


class PredictionTask(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ModelLimit(BaseModel):
    train_set_max_rows: int
    train_set_max_cells: int
    test_set_max_rows: int
    max_classes: int
    max_cols: int = 2000
    test_set_max_rows_w_full_regression_output: int = 400
    test_set_max_cells: int


class ModelVersion(str, Enum):
    V2 = "v2"
    V2_5 = "v2.5"
    V2_6 = "v2.6"
    V3 = "v3"


class ClassifierOutputType(str, Enum):
    PROBAS = "probas"
    PREDS = "preds"


class RegressorOutputType(str, Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    QUANTILES = "quantiles"
    FULL = "full"
    MAIN = "main"


class ClassifierPredictParams(BaseModel):
    output_type: Annotated[
        ClassifierOutputType | UnknownEnum, Field(union_mode="left_to_right")
    ] = ClassifierOutputType.PROBAS


class ClassifierTabPFNConfig(BaseModel):
    n_estimators: int | None = None
    categorical_features_indices: list[int] | None = None
    softmax_temperature: float | None = None
    average_before_softmax: bool | None = None
    random_state: int | None = None
    inference_config: dict[str, Any] | None = Field(
        default=None,
        description="Refer to tabpfn.inference_config.InferenceConfig for more details.",
    )
    inference_precision: (
        Annotated[Literal["autocast", "auto"] | str, Field(union_mode="left_to_right")]
        | None
    ) = None
    ignore_pretraining_limits: bool = True
    model_path: str | None = None
    balance_probabilities: bool | None = None


class ClassifierConfig(BaseModel):
    task: Literal["classification"] = "classification"
    tabpfn_config: ClassifierTabPFNConfig = Field(
        default_factory=ClassifierTabPFNConfig
    )
    predict_params: ClassifierPredictParams = Field(
        default_factory=ClassifierPredictParams
    )


class RegressorPredictParams(BaseModel):
    output_type: Annotated[
        RegressorOutputType | UnknownEnum, Field(union_mode="left_to_right")
    ] = RegressorOutputType.MEAN
    quantiles: list[float] | None = None


class RegressorTabPFNConfig(BaseModel):
    n_estimators: int | None = None
    categorical_features_indices: list[int] | None = None
    softmax_temperature: float | None = None
    average_before_softmax: bool | None = None
    random_state: int | None = None
    inference_config: dict[str, Any] | None = Field(
        default=None,
        description="Refer to tabpfn.inference_config.InferenceConfig for more details.",
    )
    inference_precision: (
        Annotated[Literal["autocast", "auto"] | str, Field(union_mode="left_to_right")]
        | None
    ) = None
    ignore_pretraining_limits: bool = True
    model_path: str | None = None


class RegressorConfig(BaseModel):
    task: Literal["regression"] = "regression"
    tabpfn_config: RegressorTabPFNConfig = Field(default_factory=RegressorTabPFNConfig)
    predict_params: RegressorPredictParams = Field(
        default_factory=RegressorPredictParams
    )


class ClassifierMetadata(BaseModel):
    test_set_num_rows: int
    test_set_num_cols: int
    task: Literal["classification"] = "classification"
    package_version: str
    tabpfn_config: ClassifierTabPFNConfig


class RegressorMetadata(BaseModel):
    test_set_num_rows: int
    test_set_num_cols: int
    task: Literal["regression"] = "regression"
    package_version: str
    tabpfn_config: RegressorTabPFNConfig


class FileInfo(BaseModel):
    format: Annotated[
        Literal["csv", "parquet"] | str, Field(union_mode="left_to_right")
    ]
    hash: str | None = Field(
        default=None,
        description="The crc32c hash of the file, used to deduplicate the file.",
    )
    size_bytes: int | None = Field(
        default=None,
        description="The size of the file in bytes, used to compute the optimal number of chunks when chunking is enabled.",
    )
    use_chunks: bool = Field(
        default=False,
        description="Whether to split the the file into chunks and upload them in parallel.",
    )


class FileUploadInfo(BaseModel):
    signed_urls: list[str]
    expires_at: float
    required_headers: dict[str, str]


Prediction = list[Any] | list[list[Any]] | dict[str, list[Any] | list[list[Any]]]


TaskConfig = Annotated[ClassifierConfig | RegressorConfig, Field(discriminator="task")]


Metadata = Annotated[
    ClassifierMetadata | RegressorMetadata, Field(discriminator="task")
]


class TransformTrainSetRequest(BaseModel):
    bucket: str
    x_train_key: str
    y_train_key: str
    fitted_train_set_prefix: str
    task: Annotated[PredictionTask | UnknownEnum, Field(union_mode="left_to_right")]
    tabpfn_systems: list[
        Annotated[
            Literal["preprocessing", "text", "thinking"] | str,
            Field(union_mode="left_to_right"),
        ]
    ]
    task_config: dict[str, Any] | None = None
    ag_time_limit_s: float = 1200.0
    thinking_effort_metric: str | None = None
    max_tabprep_configs: int | None = None


class FitRequest(BaseModel):
    train_set_upload_id: UUID
    task: Annotated[PredictionTask | UnknownEnum, Field(union_mode="left_to_right")]
    tabpfn_systems: list[
        Annotated[
            Literal["preprocessing", "text", "thinking"] | str,
            Field(union_mode="left_to_right"),
        ]
    ] = ["preprocessing", "text"]
    tabpfn_config: dict[str, Any] | None = None
    thinking_effort: (
        Annotated[Literal["medium", "high"] | str, Field(union_mode="left_to_right")]
        | None
    ) = None
    thinking_timeout_s: float | None = None
    thinking_effort_metric: str | None = None
    force_refit: bool = Field(
        default=False,
        description="Whether to force the fitting of the train set even if a fittedtrain set and transform states already exist.",
    )


class FitResponse(BaseModel):
    fitted_train_set_id: UUID


class GetModelLimitsResponse(BaseModel):
    default_model_version: Annotated[
        ModelVersion | UnknownEnum, Field(union_mode="left_to_right")
    ]
    max_model_limit: ModelLimit
    model_limits: dict[
        Annotated[ModelVersion | UnknownEnum, Field(union_mode="left_to_right")],
        ModelLimit,
    ]
    dataset_max_size_bytes: int


class TransformTestSetRequest(BaseModel):
    bucket: str
    x_test_key: str
    fitted_train_set_prefix: str
    fitted_test_set_prefix: str
    x_train_cols: int
    output_type: (
        Annotated[ClassifierOutputType | UnknownEnum, Field(union_mode="left_to_right")]
        | Annotated[
            RegressorOutputType | UnknownEnum, Field(union_mode="left_to_right")
        ]
    )
    tabpfn_systems: list[
        Annotated[
            Literal["preprocessing", "text", "thinking"] | str,
            Field(union_mode="left_to_right"),
        ]
    ]


class NotFoundErrorResponse(BaseModel):
    message: str
    error_code: str = "NOT_FOUND"
    trace_id: UUID | None = None
    detail: str | None = None


class PredictRequest(BaseModel):
    test_set_upload_id: UUID
    fitted_train_set_id: UUID
    task_config: TaskConfig
    force_refit: bool = Field(
        default=False,
        description="Whether to force the fitting of the test set even if a fittedtest set and transform states already exist.",
    )


class PredictResponse(BaseModel):
    prediction: Prediction
    metadata: Metadata


class ErrorResponse(BaseModel):
    message: str
    error_code: str = "GENERAL_ERROR"
    trace_id: UUID | None = None
    detail: str | None = None


class PrepareTestSetUploadRequest(BaseModel):
    fitted_train_set_id: UUID
    x_test_info: FileInfo
    force_reupload: bool = Field(
        default=False,
        description="Whether to force the upload of the file even if a file with the same hash already exists.",
    )


class PrepareTestSetUploadResponse(BaseModel):
    test_set_upload_id: UUID
    x_test_info: FileUploadInfo


class DuplicateTestSetErrorResponse(BaseModel):
    message: str
    error_code: str = "DUPLICATE_TEST_SET_UPLOAD"
    trace_id: UUID | None = None
    detail: str | None = None
    test_set_upload_id: UUID


class PrepareTrainSetUploadRequest(BaseModel):
    x_train_info: FileInfo
    y_train_info: FileInfo
    description: str | None = None
    force_reupload: bool = Field(
        default=False,
        description="Whether to force the upload of the file even if a file with the same hash already exists.",
    )


class PrepareTrainSetUploadResponse(BaseModel):
    train_set_upload_id: UUID
    x_train_info: FileUploadInfo
    y_train_info: FileUploadInfo


class DuplicateTrainSetErrorResponse(BaseModel):
    message: str
    error_code: str = "DUPLICATE_TRAIN_SET_UPLOAD"
    trace_id: UUID | None = None
    detail: str | None = None
    train_set_upload_id: UUID
