from uuid import UUID
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

# Classification output_type="preds" preserves the original label type, so
# scalar predictions must allow non-numeric JSON scalars in addition to floats.
# see: tabpfn/preprocessing/label_encoder.py:66
PredictionScalar = Union[str, int, bool]
Prediction = Union[
    List[PredictionScalar],
    List[float],
    List[List[float]],
    Dict[str, Union[List[Optional[float]], List[List[Optional[float]]]]],
]

TabPFNConfig = Optional[Dict[str, Any]]

PredictParams = Optional[Dict[str, Any]]


class TaskConfig(BaseModel):
    task: str
    tabpfn_config: TabPFNConfig
    predict_params: PredictParams


# ---------------------------------------------------------------------------
# Discriminated-union members for FitRequest.tabpfn_systems. Mirrors the
# server-side `prior.types.TabPFNSystemConfig` union; we send objects so each
# system can carry its own knobs (today: only `enhanced` has any).
# ---------------------------------------------------------------------------


class PreprocessingSystemConfig(BaseModel):
    name: Literal["preprocessing"] = "preprocessing"


class TextSystemConfig(BaseModel):
    name: Literal["text"] = "text"


class EnhancedSystemConfig(BaseModel):
    name: Literal["enhanced"] = "enhanced"
    eval_metric: Optional[str] = None
    fit_time_limit_s: Optional[float] = None


TabPFNSystemConfig = Union[
    PreprocessingSystemConfig,
    TextSystemConfig,
    EnhancedSystemConfig,
]


class Metadata(BaseModel):
    task: str
    package_version: str
    tabpfn_config: TabPFNConfig
    test_set_num_rows: int
    test_set_num_cols: int


class ErrorResponse(BaseModel):
    message: str
    error_code: str
    trace_id: Optional[str] = None


class FileInfo(BaseModel):
    format: str
    hash: Optional[str] = None
    size_bytes: Optional[int] = None
    use_chunks: bool = False


class FileUploadInfo(BaseModel):
    signed_urls: List[str] = Field(..., min_length=1)
    expires_at: float
    required_headers: Dict[str, str]


# ---------------------------------------------------------------------------
# /tabpfn/get_dataset_limits/
# ---------------------------------------------------------------------------
class GetDatasetLimitsResponse(BaseModel):
    dataset_max_size_bytes: int
    dataset_max_cols: int
    dataset_max_classes: int
    train_set_max_rows: int
    train_set_max_cells: int
    test_set_max_rows: int
    test_set_max_cells: int
    test_set_max_rows_w_full_regression_output: int


# ---------------------------------------------------------------------------
# /tabpfn/prepare_train_set_upload/
# ---------------------------------------------------------------------------
class PrepareTrainSetUploadRequest(BaseModel):
    x_train_info: FileInfo
    y_train_info: FileInfo
    description: Optional[str] = None
    force_reupload: bool = False


class PrepareTrainSetUploadResponse(BaseModel):
    train_set_upload_id: UUID
    x_train_info: FileUploadInfo
    y_train_info: FileUploadInfo


class DuplicateTrainSetErrorResponse(ErrorResponse):
    train_set_upload_id: UUID


# ---------------------------------------------------------------------------
# /tabpfn/fit/
# ---------------------------------------------------------------------------
class FitRequest(BaseModel):
    train_set_upload_id: UUID
    task: str
    # List of per-system configs. Server discriminates by `name`. Per-system
    # knobs (e.g. enhanced.eval_metric, enhanced.fit_time_limit_s) live on
    # the enhanced entry, not at the top level.
    tabpfn_systems: List[TabPFNSystemConfig]
    force_retransform: bool = False
    # Estimator-side configuration (model_path, hyperparameters). Persisted
    # by the server at fit time and read back at predict time, so changing
    # these per predict call no longer has any effect.
    tabpfn_config: TabPFNConfig = None


class FitResponse(BaseModel):
    fitted_train_set_id: UUID


# ---------------------------------------------------------------------------
# /tabpfn/prepare_test_set_upload/
# ---------------------------------------------------------------------------
class PrepareTestSetUploadRequest(BaseModel):
    fitted_train_set_id: UUID
    x_test_info: FileInfo
    force_reupload: bool = False


class PrepareTestSetUploadResponse(BaseModel):
    test_set_upload_id: UUID
    x_test_info: FileUploadInfo


class DuplicateTestSetErrorResponse(ErrorResponse):
    test_set_upload_id: UUID


# ---------------------------------------------------------------------------
# /tabpfn/predict/
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    test_set_upload_id: UUID
    fitted_train_set_id: UUID
    # Preferred: just the predict-time output knobs (output_type / quantiles).
    # Server reconstructs the full TaskConfig from the persisted fit row.
    predict_params: PredictParams = None
    # Deprecated, only sent by older clients. Server uses `predict_params`
    # if present, else falls back to `task_config.predict_params`. The
    # `tabpfn_config` inside is silently ignored when the persisted fit has
    # one of its own.
    task_config: Optional[TaskConfig] = None
    force_retransform: bool = False


class PredictResponse(BaseModel):
    prediction: Prediction
    metadata: Metadata
