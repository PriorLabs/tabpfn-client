# Generated code. Do not edit by hand.
#
# Forward-compat note: enum-typed fields are widened to `Union[EnumName, Unknown]`
# (and `Union[Literal[...], Unknown]` for inline literals) so the SDK does not
# reject response payloads when the server adds a new enum value. Known
# values still deserialize to the enum member; unrecognized values flow
# through as `Unknown` (a `str` subclass) instead of raising a
# ValidationError. Widened fields are wrapped in
# `Annotated[..., Field(union_mode="left_to_right")]` because pydantic's
# default smart mode would land known values in the wider branch.
# Discriminator `const` fields are intentionally left non-forward-compatible.

from enum import StrEnum
from typing import Annotated, Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class Unknown(str):
    """Sentinel for enum/literal values not known to this SDK — see header."""

    @property
    def value(self) -> str:
        return str(self)

    @classmethod
    def __get_pydantic_core_schema__(cls, _source, _handler):
        from pydantic_core import core_schema

        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())


class PredictionTask(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ModelLimit(BaseModel):
    train_set_max_rows: int
    train_set_max_cells: int
    test_set_max_rows: int
    max_classes: int
    max_cols: int = 2000
    test_set_max_rows_w_full_regression_output: int = 400


class ModelVersion(StrEnum):
    V2 = "v2"
    V2_5 = "v2.5"
    V2_6 = "v2.6"
    V3 = "v3"


class ClassifierOutputType(StrEnum):
    PROBAS = "probas"
    PREDS = "preds"


class RegressorOutputType(StrEnum):
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    QUANTILES = "quantiles"
    FULL = "full"
    MAIN = "main"


class ClassifierPredictParams(BaseModel):
    output_type: Annotated[Union[ClassifierOutputType, Unknown], Field(union_mode="left_to_right")] = (
        ClassifierOutputType.PROBAS
    )


class ClassifierTabPFNConfig(BaseModel):
    n_estimators: Optional[int] = None
    categorical_features_indices: Optional[list[int]] = None
    softmax_temperature: Optional[float] = None
    average_before_softmax: Optional[bool] = None
    random_state: Optional[int] = None
    inference_config: Optional[dict[str, Any]] = Field(
        default=None, description="Refer to tabpfn.inference_config.InferenceConfig for more details."
    )
    inference_precision: Optional[
        Annotated[Union[Literal["autocast", "auto"], Unknown], Field(union_mode="left_to_right")]
    ] = None
    ignore_pretraining_limits: bool = True
    model_path: Optional[str] = None
    balance_probabilities: Optional[bool] = None


class ClassifierConfig(BaseModel):
    task: Literal["classification"] = "classification"
    tabpfn_config: ClassifierTabPFNConfig = Field(default_factory=ClassifierTabPFNConfig)
    predict_params: ClassifierPredictParams = Field(default_factory=ClassifierPredictParams)


class RegressorPredictParams(BaseModel):
    output_type: Annotated[Union[RegressorOutputType, Unknown], Field(union_mode="left_to_right")] = (
        RegressorOutputType.MEAN
    )
    quantiles: Optional[list[float]] = None
    model_id: Optional[str] = None


class RegressorTabPFNConfig(BaseModel):
    n_estimators: Optional[int] = None
    categorical_features_indices: Optional[list[int]] = None
    softmax_temperature: Optional[float] = None
    average_before_softmax: Optional[bool] = None
    random_state: Optional[int] = None
    inference_config: Optional[dict[str, Any]] = Field(
        default=None, description="Refer to tabpfn.inference_config.InferenceConfig for more details."
    )
    inference_precision: Optional[
        Annotated[Union[Literal["autocast", "auto"], Unknown], Field(union_mode="left_to_right")]
    ] = None
    ignore_pretraining_limits: bool = True
    model_path: Optional[str] = None


class RegressorConfig(BaseModel):
    task: Literal["regression"] = "regression"
    tabpfn_config: RegressorTabPFNConfig = Field(default_factory=RegressorTabPFNConfig)
    predict_params: RegressorPredictParams = Field(default_factory=RegressorPredictParams)


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
    format: Annotated[Union[Literal["csv", "parquet"], Unknown], Field(union_mode="left_to_right")]
    hash: Optional[str] = Field(
        default=None, description="The crc32c hash of the file, used to deduplicate the file."
    )
    size_bytes: Optional[int] = Field(
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


Prediction = Union[list[Any], list[list[Any]], dict[str, Union[list[Any], list[list[Any]]]]]


TaskConfig = Annotated[Union[ClassifierConfig, RegressorConfig], Field(discriminator="task")]


Metadata = Annotated[Union[ClassifierMetadata, RegressorMetadata], Field(discriminator="task")]


class TransformTrainSetRequest(BaseModel):
    bucket: str
    x_train_key: str
    y_train_key: str
    fitted_train_set_prefix: str
    task: Union[Annotated[Union[PredictionTask, Unknown], Field(union_mode="left_to_right")], str]
    tabpfn_systems: list[
        Union[
            Annotated[
                Union[Literal["preprocessing", "text", "thinking"], Unknown],
                Field(union_mode="left_to_right"),
            ],
            str,
        ]
    ]
    task_config: Optional[dict[str, Any]] = None
    ag_time_limit_s: float = 1200.0
    thinking_effort_metric: Optional[str] = None
    max_tabprep_configs: Optional[int] = None


class FitRequest(BaseModel):
    train_set_upload_id: str
    task: Annotated[Union[PredictionTask, Unknown], Field(union_mode="left_to_right")]
    tabpfn_systems: list[
        Annotated[
            Union[Literal["preprocessing", "text", "thinking"], Unknown],
            Field(union_mode="left_to_right"),
        ]
    ] = ["preprocessing", "text"]
    tabpfn_config: Optional[dict[str, Any]] = None
    thinking_effort: Optional[
        Annotated[Union[Literal["medium", "high"], Unknown], Field(union_mode="left_to_right")]
    ] = None
    thinking_timeout_s: Optional[float] = None
    thinking_effort_metric: Optional[str] = None
    force_refit: bool = Field(
        default=False,
        description="Whether to force the fitting of the train set even if a fittedtrain set and transform states already exist.",
    )


class FitResponse(BaseModel):
    fitted_train_set_id: str


class GetModelLimitsResponse(BaseModel):
    default_model_version: Annotated[Union[ModelVersion, Unknown], Field(union_mode="left_to_right")]
    max_model_limit: ModelLimit
    model_limits: dict[str, ModelLimit]
    dataset_max_size_bytes: int


class TransformTestSetRequest(BaseModel):
    bucket: str
    x_test_key: str
    fitted_train_set_prefix: str
    fitted_test_set_prefix: str
    x_train_cols: int
    output_type: Union[
        Annotated[Union[ClassifierOutputType, Unknown], Field(union_mode="left_to_right")],
        str,
        Annotated[Union[RegressorOutputType, Unknown], Field(union_mode="left_to_right")],
    ]
    tabpfn_systems: list[
        Union[
            Annotated[
                Union[Literal["preprocessing", "text", "thinking"], Unknown],
                Field(union_mode="left_to_right"),
            ],
            str,
        ]
    ]


class NotFoundErrorResponse(BaseModel):
    message: str
    error_code: str = "NOT_FOUND"
    trace_id: Optional[str] = None
    detail: Optional[str] = None


class PredictRequest(BaseModel):
    test_set_upload_id: str
    fitted_train_set_id: str
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
    trace_id: Optional[str] = None
    detail: Optional[str] = None


class PrepareTestSetUploadRequest(BaseModel):
    fitted_train_set_id: str
    x_test_info: FileInfo
    force_reupload: bool = Field(
        default=False,
        description="Whether to force the upload of the file even if a file with the same hash already exists.",
    )


class PrepareTestSetUploadResponse(BaseModel):
    test_set_upload_id: str
    x_test_info: FileUploadInfo


class DuplicateTestSetErrorResponse(BaseModel):
    message: str
    error_code: str = "DUPLICATE_TEST_SET_UPLOAD"
    trace_id: Optional[str] = None
    detail: Optional[str] = None
    test_set_upload_id: str


class PrepareTrainSetUploadRequest(BaseModel):
    x_train_info: FileInfo
    y_train_info: FileInfo
    description: Optional[str] = None
    force_reupload: bool = Field(
        default=False,
        description="Whether to force the upload of the file even if a file with the same hash already exists.",
    )


class PrepareTrainSetUploadResponse(BaseModel):
    train_set_upload_id: str
    x_train_info: FileUploadInfo
    y_train_info: FileUploadInfo


class DuplicateTrainSetErrorResponse(BaseModel):
    message: str
    error_code: str = "DUPLICATE_TRAIN_SET_UPLOAD"
    trace_id: Optional[str] = None
    detail: Optional[str] = None
    train_set_upload_id: str
