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
from typing_extensions import assert_never

from tabpfn_client.api_models import FileInfo, FileUploadInfo, ForecastLimits
from tabpfn_client.client import ServiceClient
from tabpfn_client.config import init

_UPLOAD_PATH = "/forecast/prepare_upload"
_SUBMIT_PATH = "/forecast/submit_job"
_RESULT_PATH = "/forecast/get_job_result"


class InvalidSeriesError(ValueError):
    """The series table or the job parameters break a rule of the service."""


class TimeSeriesJobFailedError(RuntimeError):
    """The server reports that a time-series job failed."""

    def __init__(self, message: str, *, job_id: UUID):
        """Initialize the error.

        Args:
            message: The server's reason.
            job_id: The failed job.
        """
        super().__init__(message)
        self.job_id = job_id


class TimeSeriesInputParseError(TimeSeriesJobFailedError):
    """The server cannot read the table as the columns and types the job names."""


class TimeSeriesInputValidationError(TimeSeriesJobFailedError):
    """The server reads the table, but its values break a rule of the service."""


# --- wire models (mirror the server's `prior.models.forecast`) ---------------------


class _ColumnRoles(BaseModel):
    id: str
    timestamp: str
    targets: list[str]
    past_covariates: list[str]
    known_covariates: list[str]
    static_covariates: list[str]


class _ForecastTask(BaseModel):
    task: Literal["forecast"] = "forecast"
    horizon: int
    # None asks for the server's full quantile grid.
    quantiles: list[float] | None = None


class _JobSpec(BaseModel):
    freq: str
    columns: _ColumnRoles
    # Tagged, so more tasks can join as a discriminated union later.
    task: _ForecastTask


class _SubmitJobResponse(BaseModel):
    job_id: UUID


class _PrepareUploadResponse(BaseModel):
    job_id: UUID
    series_info: FileUploadInfo
    known_covariates_info: FileUploadInfo | None = None


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
    # What was wrong with the input; None when the service was at fault.
    error_kind: Literal["parse", "validation"] | None = None


_JobResult = TypeAdapter(
    Annotated[
        Union[_JobPending, _JobSucceeded, _JobFailed], Field(discriminator="status")
    ]
)


_MAX_NAMED_ITEMS = 10


# --- checks -------------------------------------------------------------------------


def validate_series(
    series: pd.DataFrame,
    *,
    freq: str,
    id_column: str,
    timestamp_column: str,
    target_columns: Sequence[str],
    past_covariate_columns: Sequence[str] = (),
    known_covariate_columns: Sequence[str] = (),
    static_covariate_columns: Sequence[str] = (),
) -> None:
    """Raise `InvalidSeriesError` if the server would reject `series`.

    The rules are the same as the worker's, in its order: no column has two roles,
    the column names are unique, every column has a role, the table has rows, every item has an observed value in each target, the targets
    are finite
    numbers, the timestamps parse, each item's timestamps are on the `freq` grid with no gaps or duplicates, and each static covariate has one value per item.
    """
    roles = [
        id_column,
        timestamp_column,
        *target_columns,
        *past_covariate_columns,
        *known_covariate_columns,
        *static_covariate_columns,
    ]
    if not target_columns:
        raise InvalidSeriesError("Name at least one target column.")
    repeated = sorted({name for name in roles if roles.count(name) > 1})
    if repeated:
        raise InvalidSeriesError(
            f"Column(s) {repeated} have more than one role, and a column can have "
            "only one role. Name each column once."
        )
    duplicated = sorted(set(series.columns[series.columns.duplicated()]))
    if duplicated:
        raise InvalidSeriesError(
            f"The series table has more than one column named {duplicated}. Give "
            "each column of the table a different name."
        )
    missing = sorted(set(roles) - set(series.columns))
    if missing:
        raise InvalidSeriesError(
            f"The series table has no column(s) {missing}, which the job names. Add "
            "them, or name the table's own columns in the job."
        )
    extra = sorted(set(series.columns) - set(roles))
    if extra:
        raise InvalidSeriesError(
            f"The series table has column(s) {extra}, which the job does not name. "
            "Drop them, or name them in the job."
        )
    if len(series) == 0:
        raise InvalidSeriesError(
            "The series table has no rows. Send at least one row per item."
        )
    for t in target_columns:
        if not pd.api.types.is_numeric_dtype(series[t]):
            raise InvalidSeriesError(
                f"Target column {t!r} has type {series[t].dtype}, not a number type. "
                "Convert it to int or float."
            )
    _check_items_have_data(series, id_column=id_column, target_columns=target_columns)
    for t in target_columns:
        # In float32, the precision the service computes in, so a float64 value too large
        # for it is refused.
        with np.errstate(over="ignore"):
            values = series[t].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise InvalidSeriesError(
                f"Target column {t!r} has NaN or inf values, or values beyond +-3.4e38. "
                "Fill them, for example by interpolation."
            )
    try:
        stamps = pd.to_datetime(series[timestamp_column])
    except (ValueError, TypeError, OverflowError) as e:
        raise InvalidSeriesError(
            f"Column {timestamp_column!r} of the series table does not parse as "
            "timestamps. Send datetime values or ISO 8601 strings."
        ) from e
    if stamps.hasnans:
        raise InvalidSeriesError(
            f"Column {timestamp_column!r} of the series table has missing timestamps. "
            "Fill them or drop their rows."
        )
    offset = _offset(freq)

    frame = pd.DataFrame(
        {"id": series[id_column].astype(str), "ts": stamps}
    ).sort_values(["id", "ts"])
    for item_id, g in frame.groupby("id", sort=True):
        _check_item(
            pd.DatetimeIndex(g["ts"]), item_id=item_id, freq=freq, offset=offset
        )
    _check_static_covariates(
        series, id_column=id_column, static_covariate_columns=static_covariate_columns
    )


def _check_static_covariates(
    series: pd.DataFrame, *, id_column: str, static_covariate_columns: Sequence[str]
) -> None:
    """Refuse a static covariate that takes more than one value in an item."""
    ids = series[id_column].astype(str)
    for name in static_covariate_columns:
        # `dropna=False`: NaN counts as a value.
        varying = series[name].groupby(ids).nunique(dropna=False) > 1
        if varying.any():
            shown = [str(i) for i in varying.index[varying][:_MAX_NAMED_ITEMS]]
            raise InvalidSeriesError(
                f"Static covariate {name!r} has more than one value in item(s) "
                f"{shown}. Give each item one value, or name the column as a past "
                "covariate."
            )


def validate_known_covariates(
    known_covariates: pd.DataFrame,
    series: pd.DataFrame,
    *,
    horizon: int,
    freq: str,
    id_column: str,
    timestamp_column: str,
) -> None:
    """Raise `InvalidSeriesError` if the known-covariates table breaks a rule.

    Per item of `series`, the table needs `horizon` rows on the `freq` grid, from the
    step after the item's last timestamp.
    """
    table = known_covariates
    duplicated = sorted(set(table.columns[table.columns.duplicated()]))
    if duplicated:
        raise InvalidSeriesError(
            f"The known covariates table has more than one column named {duplicated}. "
            "Give each column of the table a different name."
        )
    missing = sorted({id_column, timestamp_column} - set(table.columns))
    if missing:
        raise InvalidSeriesError(
            f"The known covariates table has no column(s) {missing}. Add them."
        )
    if len(table.columns) == 2:
        raise InvalidSeriesError(
            "The known covariates table has no covariate column. Add the known "
            "covariates as columns, or send no table."
        )
    try:
        stamps = pd.to_datetime(table[timestamp_column])
    except (ValueError, TypeError, OverflowError) as e:
        raise InvalidSeriesError(
            f"Column {timestamp_column!r} of the known covariates table does not parse "
            "as timestamps. Send datetime values or ISO 8601 strings."
        ) from e
    if stamps.hasnans:
        raise InvalidSeriesError(
            f"Column {timestamp_column!r} of the known covariates table has missing "
            "timestamps. Fill them or drop their rows."
        )
    offset = _offset(freq)

    series_ids = series[id_column].astype(str)
    last = pd.to_datetime(series[timestamp_column]).groupby(series_ids).max()
    frame = pd.DataFrame({"id": table[id_column].astype(str), "ts": stamps})
    absent = sorted(set(last.index) - set(frame["id"]))
    if absent:
        raise InvalidSeriesError(
            f"The known covariates table has no rows for item(s) {absent[:_MAX_NAMED_ITEMS]}. "
            "Send `horizon` rows for every item of the series table."
        )
    unknown = sorted(set(frame["id"]) - set(last.index))
    if unknown:
        raise InvalidSeriesError(
            f"The known covariates table has item(s) {unknown[:_MAX_NAMED_ITEMS]} that "
            "the series table does not have. Drop their rows."
        )
    for item_id, g in frame.sort_values(["id", "ts"]).groupby("id", sort=True):
        ts = pd.DatetimeIndex(g["ts"])
        if ts.has_duplicates:
            raise InvalidSeriesError(
                f"Item {item_id!r} has duplicate timestamps in the known covariates "
                "table. Keep one row per item and timestamp."
            )
        if len(ts) != horizon:
            raise InvalidSeriesError(
                f"Item {item_id!r} has {len(ts)} rows in the known covariates table, "
                f"not horizon={horizon}. Send one row per forecast step."
            )
        try:
            expected = pd.date_range(
                start=last[item_id] + offset, periods=horizon, freq=offset
            )
        except (ValueError, OverflowError) as e:
            raise InvalidSeriesError(
                f"With freq {freq!r}, the forecast steps of item {item_id!r} run past "
                "the year 2262, the last date the service can hold."
            ) from e
        off = np.flatnonzero(ts != expected)
        if len(off) > 0:
            row = int(off[0])
            raise InvalidSeriesError(
                f"Item {item_id!r} has timestamp {ts[row]} in the known covariates "
                f"table where {expected[row]} is due. The rows must be the {horizon} "
                f"{freq!r} steps that follow the item's last series timestamp, "
                f"{last[item_id]}."
            )


def _check_items_have_data(
    series: pd.DataFrame, *, id_column: str, target_columns: Sequence[str]
) -> None:
    """Refuse the request if an item has no observed value in some target column."""
    ids = series[id_column].astype(str)
    observed = series[list(target_columns)].notna().groupby(ids).any()
    empty = [str(i) for i in observed.index[~observed.all(axis=1)]]
    if not empty:
        return
    shown = ", ".join(repr(i) for i in empty[:_MAX_NAMED_ITEMS])
    if len(empty) > _MAX_NAMED_ITEMS:
        shown += f" and {len(empty) - _MAX_NAMED_ITEMS} more"
    raise InvalidSeriesError(
        f"{len(empty)} item(s) have no data: {shown}. Every item needs at least one "
        "observed (non-NaN) value in each target. Drop these items, or send their "
        "values."
    )


def _offset(freq: str) -> pd.DateOffset:
    """`freq` as a pandas offset, refused unless it is a positive step."""
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except OverflowError as e:
        raise InvalidSeriesError(
            f"freq {freq!r} is too large a step. Use the step between your timestamps, "
            "such as 'D' or '15min'."
        ) from e
    except ValueError as e:
        raise InvalidSeriesError(
            f"freq {freq!r} is not a pandas offset alias. Use an alias such as 'D', "
            "'h', 'W-SUN' or 'MS'."
        ) from e
    if offset.n <= 0:
        raise InvalidSeriesError(
            f"freq {freq!r} is not a positive step. Use a positive multiple, such as "
            "'D' or '15min'."
        )
    return offset


def _check_item(
    ts: pd.DatetimeIndex, *, item_id: str, freq: str, offset: pd.DateOffset
) -> None:
    """Refuse an item whose sorted timestamps are not one `freq` step apart."""
    if ts.has_duplicates:
        raise InvalidSeriesError(
            f"Item {item_id!r} has duplicate timestamps. Keep one row per item and "
            "timestamp."
        )
    try:
        expected = pd.date_range(start=ts[0], periods=len(ts), freq=offset)
    except (ValueError, OverflowError) as e:
        raise InvalidSeriesError(
            f"With freq {freq!r}, the timestamps of item {item_id!r} run past the "
            "year 2262, the last date the service can hold. Check that freq is the "
            "step between your timestamps."
        ) from e
    # `!=` and not `equals`: `equals` is False across units, e.g. `us` data against
    # an `ns` `date_range`.
    off = np.flatnonzero(ts != expected)
    if len(off) == 0:
        return
    row = int(off[0])
    raise InvalidSeriesError(
        f"Item {item_id!r} has a gap or a timestamp off the {freq!r} grid: after "
        f"{ts[row - 1]} comes {ts[row]}, not {expected[row]}. Send one row per "
        f"{freq!r} step from an item's first timestamp to its last, and check that "
        "freq is the step between your timestamps."
    )


def _check_covariates(
    *, names: Sequence[str], has_table: bool, limits: ForecastLimits
) -> None:
    """Check the covariates against the server's limits."""
    if not limits.covariates:
        if len(names) > 0 or has_table:
            raise InvalidSeriesError(
                "Covariates are not supported yet. Name no covariate columns and send "
                "no known covariates table."
            )
    elif len(names) > limits.max_covariates:
        raise InvalidSeriesError(
            f"at most {limits.max_covariates} covariates are allowed, got {len(names)}"
        )


def _check_task(
    *, horizon: int, quantiles: list[float] | None, limits: ForecastLimits | None
) -> None:
    """Check the task against the server's limits.

    Without limits (no settings, or none for forecasts) the server does the checks.
    """
    if horizon < 1:
        raise InvalidSeriesError("horizon must be at least 1")
    if quantiles is not None and not quantiles:
        raise InvalidSeriesError("give at least one quantile")
    if limits is None:
        return
    if horizon > limits.max_horizon:
        raise InvalidSeriesError(f"horizon must be at most {limits.max_horizon}")
    if quantiles is not None:
        off_grid = sorted(set(quantiles) - set(limits.quantiles))
        if off_grid:
            raise InvalidSeriesError(
                f"quantiles must be from {limits.quantiles}, got {off_grid}"
            )


_QUANTILE_PREFIX = "q_"
# Output column names, of the forecast and of the backtest to come (`cutoff`, `actual`),
# with every name that starts with `_QUANTILE_PREFIX`. No input column may take one.
_RESERVED_COLUMNS = frozenset({"target_column", "mean", "cutoff", "actual"})


def _is_reserved_column(name: str) -> bool:
    return name in _RESERVED_COLUMNS or name.startswith(_QUANTILE_PREFIX)


def _check_reserved_names(*, names: Sequence[str]) -> None:
    """Raise if a caller's column name is reserved for the output."""
    reserved = sorted({name for name in names if _is_reserved_column(name)})
    if reserved:
        raise InvalidSeriesError(
            f"column(s) {reserved} have reserved names: {sorted(_RESERVED_COLUMNS)} and "
            f"names that start with {_QUANTILE_PREFIX!r} are output columns. Rename "
            "those columns."
        )


# --- transport ----------------------------------------------------------------------


def _parquet(table: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    table.to_parquet(buf, index=False)
    return buf.getvalue()


def _submit(
    series: pd.DataFrame,
    spec: _JobSpec,
    *,
    known_covariates: pd.DataFrame | None,
    inline_max_bytes: int,
) -> UUID:
    tables = {"series": _parquet(series)}
    if known_covariates is not None:
        tables["known_covariates"] = _parquet(known_covariates)
    data = {"spec": spec.model_dump_json(exclude_none=True)}
    if sum(len(t) for t in tables.values()) <= inline_max_bytes:
        res = ServiceClient.httpx_client.post(
            _SUBMIT_PATH,
            data=data,
            files={
                name: (f"{name}.parquet", t, "application/vnd.apache.parquet")
                for name, t in tables.items()
            },
        )
    else:
        data["job_id"] = str(_upload(tables))
        res = ServiceClient.httpx_client.post(_SUBMIT_PATH, data=data)
    if res.status_code in (413, 422):
        raise InvalidSeriesError(ServiceClient._read_json_body(res)[1])
    return ServiceClient._validate_response(
        res, "submit_job", success_model=_SubmitJobResponse
    ).job_id


def _upload(tables: dict[str, bytes]) -> UUID:
    """Write the tables to signed URLs; return the job they are for."""
    res = ServiceClient.httpx_client.post(
        _UPLOAD_PATH,
        json={
            f"{name}_info": FileInfo(format="parquet", size_bytes=len(t)).model_dump(
                exclude_none=True
            )
            for name, t in tables.items()
        },
    )
    if res.status_code == 422:
        raise InvalidSeriesError(ServiceClient._read_json_body(res)[1])
    up = ServiceClient._validate_response(
        res, "prepare_upload", success_model=_PrepareUploadResponse
    )
    ServiceClient._upload_to_gcs("series", tables["series"], up.series_info)
    if "known_covariates" in tables:
        if up.known_covariates_info is None:
            raise RuntimeError("prepare_upload returned no known_covariates_info")
        ServiceClient._upload_to_gcs(
            "known_covariates", tables["known_covariates"], up.known_covariates_info
        )
    return up.job_id


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
    known_covariates: pd.DataFrame | None,
    inline_max_bytes: int,
    timeout: float,
    poll_interval: float,
) -> tuple[pd.DataFrame, dict]:
    job_id = _submit(
        series,
        spec,
        known_covariates=known_covariates,
        inline_max_bytes=inline_max_bytes,
    )
    deadline = time.monotonic() + timeout
    while True:
        result = _poll(job_id)
        match result:
            case _JobSucceeded(result_url=url, metadata=metadata):
                return _download(url), metadata
            case _JobFailed(error=error, error_kind=kind):
                match kind:
                    case "parse":
                        error_type = TimeSeriesInputParseError
                    case "validation":
                        error_type = TimeSeriesInputValidationError
                    case None:
                        error_type = TimeSeriesJobFailedError
                    case _:
                        assert_never(kind)
                raise error_type(error, job_id=job_id)
            case _JobPending():
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"job {job_id} is still pending after {timeout:.0f} s"
                    )
                time.sleep(poll_interval)


def _spec(
    series: pd.DataFrame,
    task: _ForecastTask,
    *,
    freq: str,
    id_column: str,
    timestamp_column: str,
    target_columns: Sequence[str],
    past_covariate_columns: Sequence[str],
    known_covariate_columns: Sequence[str],
    static_covariate_columns: Sequence[str],
) -> _JobSpec:
    validate_series(
        series,
        freq=freq,
        id_column=id_column,
        timestamp_column=timestamp_column,
        target_columns=target_columns,
        past_covariate_columns=past_covariate_columns,
        known_covariate_columns=known_covariate_columns,
        static_covariate_columns=static_covariate_columns,
    )
    return _JobSpec(
        freq=freq,
        columns=_ColumnRoles(
            id=id_column,
            timestamp=timestamp_column,
            targets=list(target_columns),
            past_covariates=list(past_covariate_columns),
            known_covariates=list(known_covariate_columns),
            static_covariates=list(static_covariate_columns),
        ),
        task=task,
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
    past_covariate_columns: Sequence[str] = (),
    static_covariate_columns: Sequence[str] = (),
    known_covariates: pd.DataFrame | None = None,
    quantiles: Sequence[float] | None = None,
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
        past_covariate_columns: Columns of `series` that are known for the history
            only.
        static_covariate_columns: Columns of `series` with one value per item.
        known_covariates: Table of the id column, the timestamp column and the known
            covariates, one row per item and forecast step. Its other columns are the
            known covariates, and each is also a column of `series`.
        quantiles: Quantiles to return, from the `forecast.quantiles` grid of
            `ServiceClient.get_settings()`. None returns the whole grid.
        timeout: Seconds to wait for the job.
        poll_interval: Seconds between status requests.

    Returns:
        The forecast, one row per item, target column and future timestamp, with a
        `mean` column and one `q_<level>` column per quantile; and the job metadata.

    Raises:
        InvalidSeriesError: If the table or the parameters break a rule, or a column
            name is one the result adds.
        TimeSeriesInputParseError: If the server cannot read the table as the job
            names it.
        TimeSeriesInputValidationError: If the server reads the table, but its values
            break a rule.
        TimeSeriesJobFailedError: If the server reports the job failed for another
            reason.
        TimeoutError: If the job does not finish within `timeout`.
    """
    init()
    settings = ServiceClient.get_settings()
    limits = None if settings is None else settings.forecast
    requested = None if quantiles is None else sorted(set(quantiles))
    _check_task(horizon=horizon, quantiles=requested, limits=limits)
    known_columns = (
        []
        if known_covariates is None
        else [
            c
            for c in known_covariates.columns
            if c not in (id_column, timestamp_column)
        ]
    )
    covariates = [*past_covariate_columns, *known_columns, *static_covariate_columns]
    if limits is not None:
        _check_covariates(
            names=covariates, has_table=known_covariates is not None, limits=limits
        )
    _check_reserved_names(
        names=[id_column, timestamp_column, *target_columns, *covariates]
    )
    task = _ForecastTask(horizon=horizon, quantiles=requested)
    spec = _spec(
        series,
        task,
        freq=freq,
        id_column=id_column,
        timestamp_column=timestamp_column,
        target_columns=target_columns,
        past_covariate_columns=past_covariate_columns,
        known_covariate_columns=known_columns,
        static_covariate_columns=static_covariate_columns,
    )
    if known_covariates is not None:
        validate_known_covariates(
            known_covariates,
            series,
            horizon=horizon,
            freq=freq,
            id_column=id_column,
            timestamp_column=timestamp_column,
        )
    return _run(
        series,
        spec,
        known_covariates=known_covariates,
        # Without limits, upload every table: the signed URL takes any size.
        inline_max_bytes=0 if limits is None else limits.max_inline_table_bytes,
        timeout=timeout,
        poll_interval=poll_interval,
    )
