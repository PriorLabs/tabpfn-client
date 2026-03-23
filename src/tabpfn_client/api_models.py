from uuid import UUID
from typing import Any
from pydantic import BaseModel, Field

# Classification output_type="preds" preserves the original label type, so
# scalar predictions must allow non-numeric JSON scalars in addition to floats.
# see: tabpfn/preprocessing/label_encoder.py:66
PredictionScalar = str | int | bool
Prediction = (
    list[PredictionScalar]
    | list[float]
    | list[list[float]]
    | dict[str, list[float] | list[list[float]]]
)

TabPFNConfig = dict[str, Any] | None

PredictParams = dict[str, Any] | None


class TaskConfig(BaseModel):
    task: str
    tabpfn_config: TabPFNConfig
    predict_params: PredictParams


class Metadata(BaseModel):
    task: str
    package_version: str
    tabpfn_config: TabPFNConfig
    test_set_num_rows: int
    test_set_num_cols: int


class ErrorResponse(BaseModel):
    message: str
    error_code: str
    trace_id: str | None = None


class FileInfo(BaseModel):
    format: str
    hash: str | None = None
    size_bytes: int | None = None
    use_chunks: bool = False


class FileUploadInfo(BaseModel):
    signed_urls: list[str] = Field(..., min_length=1)
    expires_at: float
    required_headers: dict[str, str]


# ---------------------------------------------------------------------------
# /tabpfn/get_constraints/
# ---------------------------------------------------------------------------
class DatasetConstraints(BaseModel):
    max_size_bytes: int
    max_cells: int
    max_cols: int
    max_classes: int


class GetConstraintsResponse(BaseModel):
    min_client_version: str
    datasets: DatasetConstraints


# ---------------------------------------------------------------------------
# /tabpfn/prepare_train_set_upload/
# ---------------------------------------------------------------------------
class PrepareTrainSetUploadRequest(BaseModel):
    x_train_info: FileInfo
    y_train_info: FileInfo
    description: str | None = None
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
    tabpfn_systems: list[str]
    force_retransform: bool = False


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
    task_config: TaskConfig
    force_retransform: bool = False


class PredictResponse(BaseModel):
    prediction: Prediction
    metadata: Metadata
