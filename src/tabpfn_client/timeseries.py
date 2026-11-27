#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

"""Time-series forecasts through the TabPFN API.

The input is a long table: one row per item and timestamp, with one or more target
columns. The client checks the structure of the request: the column roles, the
column names and the limits of the server's settings. The server checks the values,
and a job whose tables break a rule fails with a parse or validation error.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Annotated, Literal, Sequence, Union
from uuid import UUID

import httpx
import pandas as pd
from pydantic import (
    AliasChoices,
    Base64UrlBytes,
    BaseModel,
    Field,
    RootModel,
    ValidationError,
)
from typing_extensions import assert_never

from tabpfn_client.api_models import FileInfo, FileUploadInfo, ForecastLimits
from tabpfn_client.client import (
    _MAX_POLL_TIMEOUT_SECS,
    ServiceClient,
    _malformed_poll_body,
)
from tabpfn_client.config import init
from tabpfn_client.errors import (
    CappedRetryableServerError,
    InvalidSeriesError,
    RetryableServerError,
    TimeSeriesInputParseError,
    TimeSeriesInputValidationError,
    TimeSeriesJobFailedError,
)
from tabpfn_client.options import get_opts

__all__ = ["ForecastMetadata", "ForecastResult", "forecast"]

_PREPARE_UPLOAD_PATH = "/forecast/prepare_upload"
_FORECAST_PATH = "/forecast/forecast"
_JOB_RESULT_PATH = "/forecast/get_job_result"

# Output column names, of the forecast and of the backtest to come (`cutoff`,
# `actual`), with every name that starts with `_QUANTILE_PREFIX`. The id and
# timestamp columns are output columns too, so they may not take one. Target and
# covariate names never become column names of the result.
_QUANTILE_PREFIX = "q_"
_RESERVED_COLUMNS = frozenset({"target_column", "mean", "cutoff", "actual"})


# -- Wire models: the server's `prior.models.forecast`. --


class _ColumnRoles(BaseModel):
    id: str
    timestamp: str
    targets: list[str]
    past_covariates: list[str]
    known_covariates: list[str]
    static_covariates: list[str]

    @property
    def covariates(self) -> list[str]:
        return [*self.past_covariates, *self.known_covariates, *self.static_covariates]

    @property
    def names(self) -> list[str]:
        return [self.id, self.timestamp, *self.targets, *self.covariates]


class _ForecastTask(BaseModel):
    task: Literal["forecast"] = "forecast"
    horizon: int
    # None asks for the server's full quantile grid.
    quantiles: list[float] | None


class _JobSpec(BaseModel):
    freq: str
    # The server prices the job from it, and fails a job whose series table has
    # another count.
    n_items: int
    columns: _ColumnRoles
    task: _ForecastTask


class _PrepareUploadResponse(BaseModel):
    upload_id: UUID
    series_info: FileUploadInfo
    known_covariates_info: FileUploadInfo | None = None


class ForecastMetadata(BaseModel):
    """What the server reports about a forecast job, besides the forecast.

    Attributes:
        n_items: The number of items forecast.
        runtime_s: Seconds from the start of the job on the server to the end of
            the forecast.
    """

    n_items: int
    runtime_s: float


class _InlineResult(BaseModel):
    kind: Literal["inline"]
    parquet: Base64UrlBytes


class _StoredResult(BaseModel):
    kind: Literal["stored"]
    url: str


_Result = Annotated[Union[_InlineResult, _StoredResult], Field(discriminator="kind")]


class _JobPending(BaseModel):
    status: Literal["pending"]
    job_id: UUID
    retry_in_secs: float


class _JobSucceeded(BaseModel):
    status: Literal["succeeded"]
    job_id: UUID
    metadata: ForecastMetadata
    result: _Result


class _JobFailed(BaseModel):
    status: Literal["failed"]
    job_id: UUID
    error: str
    # What was wrong with the input; None when the service was at fault.
    error_kind: Literal["parse", "validation"] | None = None


_JobResult = Annotated[
    Union[_JobPending, _JobSucceeded, _JobFailed], Field(discriminator="status")
]


class _JobResultResponse(RootModel[_JobResult]):
    @property
    def retry_in_secs(self) -> float | None:
        match self.root:
            case _JobPending():
                return self.root.retry_in_secs
            case _JobSucceeded() | _JobFailed():
                return None
            case _:
                assert_never(self.root)


class _ErrorBody(BaseModel):
    """The body of an error response of the forecast endpoints."""

    # `detail` in an error streamed after the keepalive started.
    message: str = Field(validation_alias=AliasChoices("message", "detail"))
    error_code: str | None = None


# A job that failed is named in the `X-Job-Id` header, whatever the status.
_FORECAST_ERRORS = dict.fromkeys(range(400, 600), _ErrorBody)


@dataclass(frozen=True)
class ForecastResult:
    """A forecast and what the server reports about its job.

    Attributes:
        forecast: One row per item, target column and future timestamp: the id
            and timestamp columns, `target_column`, `mean`, and one `q_<level>`
            column per quantile.
        metadata: What the server reports about the job.
    """

    forecast: pd.DataFrame
    metadata: ForecastMetadata


# -- Public API. --


def forecast(
    series: pd.DataFrame,
    *,
    horizon: int,
    freq: str | None = None,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
    target_columns: Sequence[str] = ("target",),
    past_covariate_columns: Sequence[str] = (),
    static_covariate_columns: Sequence[str] = (),
    known_covariates: pd.DataFrame | None = None,
    quantiles: Sequence[float] | None = None,
) -> ForecastResult:
    """Forecast the next `horizon` steps of every item in `series`.

    Several target columns give one joint (multivariate) forecast per item.

    Args:
        series: Long table, one row per item and timestamp. Every column has one
            role: id, timestamp, target or covariate.
        horizon: Number of steps to forecast.
        freq: Pandas offset alias of the timestamps, for example "D" or "h". None
            infers it from the timestamps of the first items of `series`.
        id_column: Column that names the item.
        timestamp_column: Column with the timestamps.
        target_columns: Columns to forecast.
        past_covariate_columns: Columns of `series` known for the history only.
        static_covariate_columns: Columns of `series` with one value per item.
        known_covariates: Table of the id column, the timestamp column and the
            known covariates, with `horizon` rows per item right after its last
            row in `series`. Its other columns are the known covariates, and
            `series` has each of them for the history.
        quantiles: Quantiles to return, from the `forecast.quantiles` grid of
            `ServiceClient.get_settings()`. None returns the whole grid.

    Returns:
        The forecast and what the server reports about the job.

    Raises:
        InvalidSeriesError: If the column roles, the column names or the
            parameters break a rule of the service, or the server refuses the
            request before it creates a job.
        TimeSeriesInputParseError: If the server cannot read a table as the
            column roles name it.
        TimeSeriesInputValidationError: If the server reads the tables, but their
            values break a rule of the service.
        TimeSeriesJobFailedError: If the job fails for another reason.
        TimeoutError: If the job does not finish within the poll timeout.
        RuntimeError: If the server does not offer forecasts, or fails the
            request before it creates a job in any other way, such as 429 for a
            quota or rate limit.
    """
    init()
    limits = _forecast_limits()
    known_covariate_columns = (
        []
        if known_covariates is None
        else _known_covariate_columns(
            known_covariates, id_column=id_column, timestamp_column=timestamp_column
        )
    )
    columns = _ColumnRoles(
        id=id_column,
        timestamp=timestamp_column,
        targets=list(target_columns),
        past_covariates=list(past_covariate_columns),
        known_covariates=known_covariate_columns,
        static_covariates=list(static_covariate_columns),
    )
    task = _ForecastTask(
        horizon=horizon, quantiles=None if quantiles is None else list(quantiles)
    )
    _check_spec(columns, task, limits=limits)
    _check_series_columns(series, columns)
    if freq is None:
        freq = _infer_freq(
            series, id_column=id_column, timestamp_column=timestamp_column
        )
    spec = _JobSpec(
        freq=freq, n_items=int(series[id_column].nunique()), columns=columns, task=task
    )
    job = _start(
        spec,
        _parquet(series),
        known_covariates=None
        if known_covariates is None
        else _parquet(known_covariates),
        limits=limits,
    )
    return _result_of(job)


# -- Structure checks. --


_FREQ_SAMPLE_ITEMS = 100
_FREQ_HINT = 'Pass `freq`, for example "D" or "h".'


def _infer_freq(series: pd.DataFrame, *, id_column: str, timestamp_column: str) -> str:
    """Infer the frequency from the first 100 items (by id) with 3 or more rows."""
    freqs: set[str] = set()
    n_sampled = 0
    for _, rows in series.groupby(id_column, sort=True):
        if len(rows) < 3:
            continue
        timestamps = pd.DatetimeIndex(pd.to_datetime(rows[timestamp_column]))
        freq = pd.infer_freq(timestamps.sort_values())
        if freq is None:
            raise InvalidSeriesError(
                "Cannot infer `freq`: the timestamps of an item are irregular. "
                + _FREQ_HINT
            )
        freqs.add(freq)
        n_sampled += 1
        if n_sampled == _FREQ_SAMPLE_ITEMS:
            break
    if n_sampled == 0:
        raise InvalidSeriesError(
            "Cannot infer `freq`: no item has 3 rows. " + _FREQ_HINT
        )
    if len(freqs) > 1:
        raise InvalidSeriesError(
            f"Cannot infer `freq`: items disagree, found {sorted(freqs)}. " + _FREQ_HINT
        )
    return freqs.pop()


def _forecast_limits() -> ForecastLimits:
    settings = ServiceClient.get_settings()
    if settings is None:
        raise RuntimeError(
            "Could not read the settings of the TabPFN server, which hold the "
            "forecast limits. Try again later."
        )
    if settings.forecast is None:
        raise RuntimeError(
            "This TabPFN server does not offer forecasts, or not to your account."
        )
    return settings.forecast


def _check_spec(
    columns: _ColumnRoles, task: _ForecastTask, *, limits: ForecastLimits
) -> None:
    """Check the column roles and the task against the rules and `limits`."""
    if not columns.targets:
        raise InvalidSeriesError("Name at least one target column.")
    if len(columns.targets) > limits.max_targets:
        raise InvalidSeriesError(
            f"The job names {len(columns.targets)} target columns, and a job can "
            f"have at most {limits.max_targets}. Name fewer."
        )
    names = columns.names
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise InvalidSeriesError(
            f"Column(s) {repeated} have more than one role, and a column can have "
            "only one role. Name each column once."
        )
    reserved = sorted(
        {
            name
            for name in (columns.id, columns.timestamp)
            if name in _RESERVED_COLUMNS or name.startswith(_QUANTILE_PREFIX)
        }
    )
    if reserved:
        raise InvalidSeriesError(
            f"The id or timestamp column {reserved} has a reserved name: "
            f"{sorted(_RESERVED_COLUMNS)} and names that start with "
            f"{_QUANTILE_PREFIX!r} are output columns. Rename the column."
        )
    if len(columns.covariates) > limits.max_covariates:
        raise InvalidSeriesError(
            f"The job names {len(columns.covariates)} covariates, and a job can "
            f"have at most {limits.max_covariates}. Name fewer."
        )
    if not 1 <= task.horizon <= limits.max_horizon:
        raise InvalidSeriesError(
            f"horizon is {task.horizon}; it must be from 1 to {limits.max_horizon}."
        )
    if task.quantiles is None:
        return
    if not task.quantiles:
        raise InvalidSeriesError(
            "Give at least one quantile, or None for the whole grid."
        )
    off_grid = sorted(set(task.quantiles) - set(limits.quantiles))
    if off_grid:
        raise InvalidSeriesError(
            f"Quantiles {off_grid} are not on the grid {limits.quantiles}. Pick "
            "quantiles from the grid."
        )


def _check_series_columns(series: pd.DataFrame, columns: _ColumnRoles) -> None:
    """Check that `series` has each named column once, and no other column."""
    _check_unique_columns(series, label="series")
    names = columns.names
    missing = sorted(set(names) - set(series.columns))
    if missing:
        raise InvalidSeriesError(
            f"The series table has no column(s) {missing}, which the job names. Add "
            "them, or name the table's own columns."
        )
    extra = sorted(str(c) for c in set(series.columns) - set(names))
    if extra:
        raise InvalidSeriesError(
            f"The series table has column(s) {extra}, which the job does not name. "
            "Drop them, or give them a role."
        )


def _known_covariate_columns(
    known_covariates: pd.DataFrame, *, id_column: str, timestamp_column: str
) -> list[str]:
    """The known covariates: the columns of the table besides the id and timestamp."""
    _check_unique_columns(known_covariates, label="known covariates")
    missing = [
        name
        for name in (id_column, timestamp_column)
        if name not in known_covariates.columns
    ]
    if missing:
        raise InvalidSeriesError(
            f"The known covariates table has no column(s) {missing}. Add them."
        )
    names = [
        str(c)
        for c in known_covariates.columns
        if c not in (id_column, timestamp_column)
    ]
    if not names:
        raise InvalidSeriesError(
            "The known covariates table has no covariate column. Add the known "
            "covariates as columns, or send no table."
        )
    return names


def _check_unique_columns(table: pd.DataFrame, *, label: str) -> None:
    duplicated = sorted(str(c) for c in set(table.columns[table.columns.duplicated()]))
    if duplicated:
        raise InvalidSeriesError(
            f"The {label} table has more than one column named {duplicated}. Give "
            "each column of the table a different name."
        )


# -- Transport. --


def _parquet(table: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    # Snappy, pandas' default: the server reads snappy or uncompressed parquet only.
    table.to_parquet(buf, index=False, compression="snappy")
    return buf.getvalue()


def _start(
    spec: _JobSpec,
    series: bytes,
    *,
    known_covariates: bytes | None,
    limits: ForecastLimits,
) -> _JobResult:
    """Run the job with one request, which answers pending if the job outlives it."""
    data = {"spec": spec.model_dump_json(exclude_none=True)}
    inline_bytes = len(series) + (
        0 if known_covariates is None else len(known_covariates)
    )
    if inline_bytes <= limits.max_inline_table_bytes:
        files = {"series": ("series.parquet", series, "application/vnd.apache.parquet")}
        if known_covariates is not None:
            files["known_covariates"] = (
                "known_covariates.parquet",
                known_covariates,
                "application/vnd.apache.parquet",
            )
        res = ServiceClient.httpx_client.post(_FORECAST_PATH, data=data, files=files)
    else:
        data["upload_id"] = str(_upload(series, known_covariates=known_covariates))
        res = ServiceClient.httpx_client.post(_FORECAST_PATH, data=data)
    resp = ServiceClient._validate_response(
        res,
        "forecast",
        success_model=_JobResultResponse,
        error_models=_FORECAST_ERRORS,
    )
    match resp:
        case _JobResultResponse():
            return resp.root
        case _ErrorBody():
            raise _failure(res, resp)
        case _:
            assert_never(resp)


def _failure(res: httpx.Response, error: _ErrorBody) -> Exception:
    """The error for a failed request: a failed job if the server named one."""
    header = res.headers.get("X-Job-Id")
    if header is None:
        if res.status_code in (413, 422):
            return InvalidSeriesError(error.message)
        # No job exists, so this is a quota or rate limit (429) or an outage: the
        # error of a prediction request that fails before its job.
        body, message = ServiceClient._read_json_body(res)
        ServiceClient._raise_http_error(res, "forecast", body, message)
    job_id = UUID(header)
    match error.error_code:
        case "INPUT_PARSE_ERROR":
            return TimeSeriesInputParseError(error.message, job_id=job_id)
        case "INPUT_VALIDATION_ERROR":
            return TimeSeriesInputValidationError(error.message, job_id=job_id)
        case _:
            return TimeSeriesJobFailedError(error.message, job_id=job_id)


def _upload(series: bytes, *, known_covariates: bytes | None) -> UUID:
    """Write the tables to signed URLs, and return the job they are for."""
    request = {"series_info": _file_info(series)}
    if known_covariates is not None:
        request["known_covariates_info"] = _file_info(known_covariates)
    res = ServiceClient.httpx_client.post(_PREPARE_UPLOAD_PATH, json=request)
    up = ServiceClient._validate_response(
        res, "prepare_upload", success_model=_PrepareUploadResponse
    )
    ServiceClient._upload_to_gcs("series", series, up.series_info)
    if known_covariates is not None:
        if up.known_covariates_info is None:
            raise RuntimeError(
                "The server sent no upload URL for the known covariates table."
            )
        ServiceClient._upload_to_gcs(
            "known_covariates", known_covariates, up.known_covariates_info
        )
    return up.upload_id


def _file_info(table: bytes) -> dict[str, object]:
    return FileInfo(format="parquet", size_bytes=len(table)).model_dump(
        mode="json", exclude_none=True
    )


def _result_of(job: _JobResult) -> ForecastResult:
    """The forecast of `job`, once it is done."""
    match job:
        case _JobPending():
            return _result_of(_wait_for_job(job))
        case _JobSucceeded():
            return ForecastResult(
                forecast=_read_result(job.result), metadata=job.metadata
            )
        case _JobFailed():
            raise _job_failed(job)
        case _:
            assert_never(job)


def _job_failed(job: _JobFailed) -> TimeSeriesJobFailedError:
    match job.error_kind:
        case "parse":
            error_type = TimeSeriesInputParseError
        case "validation":
            error_type = TimeSeriesInputValidationError
        case None:
            error_type = TimeSeriesJobFailedError
        case _:
            assert_never(job.error_kind)
    return error_type(job.error, job_id=job.job_id)


def _read_result(result: _Result) -> pd.DataFrame:
    match result:
        case _InlineResult():
            return pd.read_parquet(io.BytesIO(result.parquet))
        case _StoredResult():
            return ServiceClient._download_parquet(
                result.url, timeout=get_opts().TABPFN_CLIENT_TIMEOUT
            )
        case _:
            assert_never(result)


def _wait_for_job(job: _JobPending) -> _JobResult:
    """Poll the job until it succeeds or fails, and return that answer.

    The first poll comes after the job's `retry_in_secs`, and each later one after
    the hint of the answer before it.

    Poll errors are classified as for a prediction: 408, 502, 503, 504 and network
    errors are retried until the poll timeout; 429, 500 and a malformed body up to
    3 times in a row; anything else, such as an unknown job (404), raises at once.
    """
    resp = ServiceClient._poll_until_done(
        lambda: _get_job_result(job.job_id),
        lambda r: not isinstance(r.root, _JobPending),
        subject=f"forecast job {job.job_id}",
        transient_errors=(
            httpx.NetworkError,
            httpx.TimeoutException,
            RetryableServerError,
        ),
        capped_errors=(CappedRetryableServerError,),
        first_poll_delay_secs=job.retry_in_secs,
    )
    return resp.root


def _get_job_result(job_id: UUID) -> _JobResultResponse:
    res = ServiceClient.httpx_client.post(
        _JOB_RESULT_PATH, json={"job_id": str(job_id)}, timeout=_MAX_POLL_TIMEOUT_SECS
    )
    if res.status_code in (429, 500):
        raise CappedRetryableServerError(
            f"Fail to call get_job_result: [HTTP {res.status_code}] {res.text}."
        )
    try:
        return ServiceClient._validate_response(
            res, "get_job_result", success_model=_JobResultResponse
        )
    except ValidationError as exc:
        raise _malformed_poll_body("get_job_result", exc) from exc
