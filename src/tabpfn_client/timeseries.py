"""PROTOTYPE: time-series forecasts through the TabPFN API.

The input is a long table: one row per item and timestamp, with one or more target
columns. The client checks the table before upload; the server checks it again.
"""

from __future__ import annotations

import io
import time
from typing import Annotated, Literal, Sequence, Union
from uuid import UUID

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, TypeAdapter

from tabpfn_client.client import ServiceClient
from tabpfn_client.config import init

# The quantiles the service can return. Keep in step with the server.
QUANTILE_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

_SUBMIT_PATH = "/forecast/v0/submit_job"
_RESULT_PATH = "/forecast/v0/get_job_result"


class InvalidSeriesError(ValueError):
    """The series table or the job parameters break a rule of the service."""


class TimeSeriesJobFailedError(RuntimeError):
    """The server reports that a time-series job failed."""

    def __init__(self, message: str, *, job_id: UUID, user_error: bool):
        """Initialize the error.

        Args:
            message: The server's reason.
            job_id: The failed job.
            user_error: True when the input was at fault, False when the service was.
        """
        super().__init__(message)
        self.job_id = job_id
        self.user_error = user_error


# --- wire models (mirror the server's `prior.models.forecast`) ---------------------


class _ColumnRoles(BaseModel):
    id: str
    timestamp: str
    targets: list[str]


class _ForecastTask(BaseModel):
    task: Literal["forecast"] = "forecast"
    horizon: int
    quantiles: list[float]


class _BacktestTask(BaseModel):
    task: Literal["backtest"] = "backtest"
    horizon: int
    quantiles: list[float]
    n_windows: int
    step_size: int | None = None


class _JobSpec(BaseModel):
    freq: str
    columns: _ColumnRoles
    task: Annotated[Union[_ForecastTask, _BacktestTask], Field(discriminator="task")]
    preset: str | None = None


class _SubmitJobResponse(BaseModel):
    job_id: UUID


class _JobPending(BaseModel):
    status: Literal["pending"]
    job_id: UUID


class _JobSucceeded(BaseModel):
    status: Literal["succeeded"]
    job_id: UUID
    result_url: str
    metadata: dict


class _JobFailed(BaseModel):
    status: Literal["failed"]
    job_id: UUID
    error: str
    user_error: bool


_JobResult = TypeAdapter(
    Annotated[
        Union[_JobPending, _JobSucceeded, _JobFailed], Field(discriminator="status")
    ]
)


# --- checks -------------------------------------------------------------------------


def validate_series(
    series: pd.DataFrame,
    *,
    freq: str,
    id_column: str,
    timestamp_column: str,
    target_columns: Sequence[str],
) -> None:
    """Raise `InvalidSeriesError` if the server would reject `series`.

    The rules are the same as the worker's: every column has exactly one role, the
    timestamps parse, the targets are finite numbers, and each item's timestamps are
    on the `freq` grid with no gaps or duplicates.
    """
    roles = [id_column, timestamp_column, *target_columns]
    if not target_columns:
        raise InvalidSeriesError("give at least one target column")
    if len(set(roles)) != len(roles):
        raise InvalidSeriesError("a column can have only one role")
    missing = sorted(set(roles) - set(series.columns))
    if missing:
        raise InvalidSeriesError(f"the series table has no column(s) {missing}")
    extra = sorted(set(series.columns) - set(roles))
    if extra:
        raise InvalidSeriesError(f"the series table has columns with no role: {extra}")
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except ValueError as e:
        raise InvalidSeriesError(f"freq {freq!r} is not a pandas offset alias") from e
    try:
        stamps = pd.to_datetime(series[timestamp_column])
    except (ValueError, TypeError) as e:
        raise InvalidSeriesError(
            f"column {timestamp_column!r} does not parse as timestamps"
        ) from e
    for t in target_columns:
        if not pd.api.types.is_numeric_dtype(series[t]):
            raise InvalidSeriesError(f"target column {t!r} is not numeric")
        if not np.isfinite(series[t].to_numpy(dtype=np.float64)).all():
            raise InvalidSeriesError(f"target column {t!r} has NaN or inf values")

    frame = pd.DataFrame(
        {"id": series[id_column].astype(str), "ts": stamps}
    ).sort_values(["id", "ts"])
    for item_id, g in frame.groupby("id", sort=True):
        ts = pd.DatetimeIndex(g["ts"])
        if ts.has_duplicates:
            raise InvalidSeriesError(f"item {item_id!r} has duplicate timestamps")
        expected = pd.date_range(start=ts[0], periods=len(ts), freq=offset)
        if not ts.equals(expected):
            raise InvalidSeriesError(
                f"item {item_id!r} has gaps, or timestamps off the {freq!r} grid"
            )


def _check_quantiles(quantiles: Sequence[float]) -> list[float]:
    off_grid = sorted(set(quantiles) - set(QUANTILE_GRID))
    if off_grid:
        raise InvalidSeriesError(
            f"quantiles must be from {list(QUANTILE_GRID)}, got {off_grid}"
        )
    if not quantiles:
        raise InvalidSeriesError("give at least one quantile")
    return sorted(set(quantiles))


# --- transport ----------------------------------------------------------------------


def _submit(series: pd.DataFrame, spec: _JobSpec) -> UUID:
    buf = io.BytesIO()
    series.to_parquet(buf, index=False)
    res = ServiceClient.httpx_client.post(
        _SUBMIT_PATH,
        data={"spec": spec.model_dump_json(exclude_none=True)},
        files={
            "series": (
                "series.parquet",
                buf.getvalue(),
                "application/vnd.apache.parquet",
            )
        },
    )
    if res.status_code in (413, 422):
        raise InvalidSeriesError(res.json().get("detail", res.text))
    return ServiceClient._validate_response(
        res, "submit_job", success_model=_SubmitJobResponse
    ).job_id


def _poll(job_id: UUID) -> _JobPending | _JobSucceeded | _JobFailed:
    res = ServiceClient.httpx_client.post(_RESULT_PATH, json={"job_id": str(job_id)})
    return ServiceClient._validate_response(
        res, "get_job_result", success_model=_JobResult
    )


def _download(url: str) -> pd.DataFrame:
    res = ServiceClient.storage_client.get(url)
    if res.status_code != 200:
        raise RuntimeError(f"result download failed: {res.status_code} {res.text}")
    return pd.read_parquet(io.BytesIO(res.content))


def _run(
    series: pd.DataFrame,
    spec: _JobSpec,
    *,
    timeout: float,
    poll_interval: float,
) -> tuple[pd.DataFrame, dict]:
    init()
    job_id = _submit(series, spec)
    deadline = time.monotonic() + timeout
    while True:
        result = _poll(job_id)
        match result:
            case _JobSucceeded(result_url=url, metadata=metadata):
                return _download(url), metadata
            case _JobFailed(error=error, user_error=user_error):
                raise TimeSeriesJobFailedError(
                    error, job_id=job_id, user_error=user_error
                )
            case _JobPending():
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"job {job_id} is still pending after {timeout:.0f} s"
                    )
                time.sleep(poll_interval)


def _spec(
    series: pd.DataFrame,
    task: _ForecastTask | _BacktestTask,
    *,
    freq: str,
    id_column: str,
    timestamp_column: str,
    target_columns: Sequence[str],
    preset: str | None,
) -> _JobSpec:
    validate_series(
        series,
        freq=freq,
        id_column=id_column,
        timestamp_column=timestamp_column,
        target_columns=target_columns,
    )
    return _JobSpec(
        freq=freq,
        columns=_ColumnRoles(
            id=id_column, timestamp=timestamp_column, targets=list(target_columns)
        ),
        task=task,
        preset=preset,
    )


# --- public API ---------------------------------------------------------------------


def forecast(
    series: pd.DataFrame,
    *,
    horizon: int,
    freq: str,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
    target_columns: Sequence[str] = ("target",),
    quantiles: Sequence[float] = QUANTILE_GRID,
    preset: str | None = None,
    timeout: float = 1800.0,
    poll_interval: float = 5.0,
) -> tuple[pd.DataFrame, dict]:
    """Forecast the next `horizon` steps of every item in `series`.

    Several target columns give one joint (multivariate) forecast per item.

    Args:
        series: Long table, one row per item and timestamp.
        horizon: Number of steps to forecast.
        freq: Pandas offset alias of the timestamps, for example "D" or "h".
        id_column: Column that names the item.
        timestamp_column: Column with the timestamps.
        target_columns: Columns to forecast.
        quantiles: Quantiles to return, from `QUANTILE_GRID`.
        preset: Ensemble size. None lets the server choose.
        timeout: Seconds to wait for the job.
        poll_interval: Seconds between status requests.

    Returns:
        The forecast, one row per item, target column and future timestamp, with a
        `mean` column and one `q_<level>` column per quantile; and the job metadata.

    Raises:
        InvalidSeriesError: If the table or the parameters break a rule.
        TimeSeriesJobFailedError: If the server reports the job failed.
        TimeoutError: If the job does not finish within `timeout`.
    """
    if horizon < 1:
        raise InvalidSeriesError("horizon must be at least 1")
    task = _ForecastTask(horizon=horizon, quantiles=_check_quantiles(quantiles))
    spec = _spec(
        series,
        task,
        freq=freq,
        id_column=id_column,
        timestamp_column=timestamp_column,
        target_columns=target_columns,
        preset=preset,
    )
    return _run(series, spec, timeout=timeout, poll_interval=poll_interval)


def backtest(
    series: pd.DataFrame,
    *,
    horizon: int,
    freq: str,
    n_windows: int = 1,
    step_size: int | None = None,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
    target_columns: Sequence[str] = ("target",),
    quantiles: Sequence[float] = QUANTILE_GRID,
    preset: str | None = None,
    timeout: float = 1800.0,
    poll_interval: float = 5.0,
) -> tuple[pd.DataFrame, dict]:
    """Forecast the last part of `series` as if it were not known yet.

    Window `i` cuts `horizon + i * step_size` steps off the end of every item and
    forecasts the next `horizon` steps from there. The result has the forecast next
    to the known values, so the caller can score it.

    Args:
        series: Long table, one row per item and timestamp.
        horizon: Number of steps each window forecasts.
        freq: Pandas offset alias of the timestamps.
        n_windows: Number of windows, from 1 to 10.
        step_size: Steps between window starts. None means `horizon`.
        id_column: Column that names the item.
        timestamp_column: Column with the timestamps.
        target_columns: Columns to forecast.
        quantiles: Quantiles to return, from `QUANTILE_GRID`.
        preset: Ensemble size. None lets the server choose.
        timeout: Seconds to wait for the job.
        poll_interval: Seconds between status requests.

    Returns:
        The same columns as `forecast`, plus `cutoff` (the last known timestamp of
        the window) and `actual`; and the job metadata.

    Raises:
        InvalidSeriesError: If the table or the parameters break a rule.
        TimeSeriesJobFailedError: If the server reports the job failed.
        TimeoutError: If the job does not finish within `timeout`.
    """
    if horizon < 1:
        raise InvalidSeriesError("horizon must be at least 1")
    if not 1 <= n_windows <= 10:
        raise InvalidSeriesError("n_windows must be from 1 to 10")
    if step_size is not None and step_size < 1:
        raise InvalidSeriesError("step_size must be at least 1")
    task = _BacktestTask(
        horizon=horizon,
        quantiles=_check_quantiles(quantiles),
        n_windows=n_windows,
        step_size=step_size,
    )
    spec = _spec(
        series,
        task,
        freq=freq,
        id_column=id_column,
        timestamp_column=timestamp_column,
        target_columns=target_columns,
        preset=preset,
    )
    return _run(series, spec, timeout=timeout, poll_interval=poll_interval)
