"""Time-series jobs: the local checks, and submit, poll and download over HTTP."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from unittest.mock import patch
from urllib.parse import parse_qs
from uuid import UUID

import numpy as np
import pandas as pd
import pytest
import respx

from tabpfn_client import timeseries
from tabpfn_client.api_models import ForecastLimits, GetSettingsResponse
from tabpfn_client.client import ServiceClient
from tabpfn_client.timeseries import (
    InvalidSeriesError,
    TimeSeriesJobFailedError,
    validate_series,
)
from tests.mock_tabpfn_server import MockTabPFNServer
from tests.unit.test_client import _api_settings_payload

JOB_ID = UUID("00000000-0000-0000-0000-0000000000f1")
COLS = dict(id_column="item_id", timestamp_column="timestamp", target_columns=["y"])


GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _settings(**limits) -> GetSettingsResponse:
    forecast = ForecastLimits(
        **{
            "max_horizon": 64,
            "max_inline_table_bytes": 1_000_000,
            "max_upload_bytes": 10_000_000,
            "max_items": 100,
            "max_rows": 10_000,
            "quantiles": GRID,
            **limits,
        }
    )
    return GetSettingsResponse(**_api_settings_payload(), forecast=forecast)


def _series(n_items: int = 2, length: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": np.repeat([f"i{k}" for k in range(n_items)], length),
            "timestamp": np.tile(
                pd.date_range("2024-01-01", periods=length, freq="D"), n_items
            ),
            "y": np.arange(n_items * length, dtype=float),
        }
    )


@pytest.fixture(autouse=True)
def authorized_client() -> Iterator[None]:
    ServiceClient.reset_authorization()
    ServiceClient.authorize("dummy_token")
    with (
        patch.object(timeseries, "init"),
        patch.object(timeseries.time, "sleep"),
        patch.object(ServiceClient, "get_settings", return_value=_settings()),
    ):
        yield
    ServiceClient.reset_authorization()


@pytest.fixture
def server() -> Iterator[MockTabPFNServer]:
    with MockTabPFNServer() as s:
        yield s


# --- validate_series ----------------------------------------------------------------


def test__validate_series__valid_table__passes():
    validate_series(_series(), freq="D", **COLS)


def test__validate_series__microsecond_timestamps__passes():
    df = _series()
    validate_series(
        df.assign(timestamp=df["timestamp"].astype("datetime64[us]")), freq="D", **COLS
    )


def test__validate_series__unsorted_rows__passes():
    validate_series(_series().sample(frac=1, random_state=0), freq="D", **COLS)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda df: df.drop(columns="y"), "no column"),
        (lambda df: df.assign(extra=1), "which the job does not name"),
        (lambda df: df.assign(y=df["y"].astype(str)), "not a number type"),
        (lambda df: df.assign(y=df["y"].where(df.index != 3)), "NaN or inf"),
        (lambda df: df.assign(y=1e300), "beyond \\+-3.4e38"),
        (lambda df: df.assign(timestamp="not a date"), "parse as timestamps"),
        (
            lambda df: df.assign(timestamp=df["timestamp"].where(df.index != 3)),
            "missing timestamps",
        ),
        (
            lambda df: df.drop(index=2),
            "Item 'i0' has a gap or a timestamp off the 'D' grid: after 2024-01-02 "
            "00:00:00 comes 2024-01-04 00:00:00, not 2024-01-03 00:00:00",
        ),
        (lambda df: pd.concat([df, df.iloc[[0]]]), "duplicate"),
    ],
    ids=[
        "missing",
        "extra",
        "text",
        "nan",
        "float32_overflow",
        "bad_ts",
        "missing_ts",
        "gap",
        "duplicate",
    ],
)
def test__validate_series__bad_table__raises(mutate, match):
    with pytest.raises(InvalidSeriesError, match=match):
        validate_series(mutate(_series()), freq="D", **COLS)


def test__validate_series__off_grid_freq__raises():
    with pytest.raises(InvalidSeriesError, match="grid"):
        validate_series(_series(), freq="h", **COLS)


@pytest.mark.parametrize(
    ("freq", "match"),
    [
        ("fortnight", "is not a pandas offset alias"),
        ("0D", "is not a positive step"),
        ("-1h", "is not a positive step"),
        ("99999999999999999999D", "is too large a step"),
        ("100000D", "run past the year 2262"),
    ],
    ids=["unknown", "zero", "negative", "overflow", "past_2262"],
)
def test__validate_series__bad_freq__raises(freq, match):
    with pytest.raises(InvalidSeriesError, match=match):
        validate_series(_series(), freq=freq, **COLS)


def test__validate_series__column_in_two_roles__raises():
    with pytest.raises(InvalidSeriesError, match="one role"):
        validate_series(
            _series(),
            freq="D",
            id_column="item_id",
            timestamp_column="timestamp",
            target_columns=["y", "item_id"],
        )


# --- forecast over HTTP -----------------------------------------------------------


def _result_parquet() -> bytes:
    buf = io.BytesIO()
    pd.DataFrame({"item_id": ["i0"], "mean": [1.0], "q_0.5": [1.0]}).to_parquet(buf)
    return buf.getvalue()


def test__forecast__job_succeeds__returns_result_and_metadata(server):
    router = server.router
    submit = router.post("/forecast/submit_job").respond(
        200, json={"job_id": str(JOB_ID)}
    )
    url = f"{server.base_url}/signed/result.parquet"
    router.post("/forecast/get_job_result").mock(
        side_effect=[
            respx.MockResponse(200, json={"status": "pending", "job_id": str(JOB_ID)}),
            respx.MockResponse(
                200,
                json={
                    "status": "succeeded",
                    "job_id": str(JOB_ID),
                    "result_url": url,
                    "metadata": {"n_items": 2},
                },
            ),
        ]
    )
    router.get("/signed/result.parquet").respond(200, content=_result_parquet())

    out, metadata = timeseries.forecast(
        _series(), horizon=3, freq="D", target_columns=["y"], quantiles=[0.5]
    )

    assert list(out.columns) == ["item_id", "mean", "q_0.5"]
    assert metadata == {"n_items": 2}
    body = submit.calls.last.request.content
    assert b'{"task":"forecast","horizon":3,"quantiles":[0.5]}' in body
    assert b"preset" not in body


def test__forecast__job_fails__raises_with_user_error(server):
    server.router.post("/forecast/submit_job").respond(
        200, json={"job_id": str(JOB_ID)}
    )
    server.router.post("/forecast/get_job_result").respond(
        200,
        json={
            "status": "failed",
            "job_id": str(JOB_ID),
            "error": "Item 'i0' has duplicate timestamps.",
            "user_error": True,
        },
    )

    with pytest.raises(TimeSeriesJobFailedError, match="duplicate") as exc:
        timeseries.forecast(_series(), horizon=3, freq="D", target_columns=["y"])

    assert exc.value.user_error is True
    assert exc.value.job_id == JOB_ID


def test__forecast__server_rejects_spec__raises_invalid_series_with_its_message(
    server,
):
    server.router.post("/forecast/submit_job").respond(
        422,
        json={
            "message": "task.horizon: too large",
            "error_code": "REQUEST_VALIDATION_ERROR",
            "trace_id": "t",
        },
    )

    with pytest.raises(InvalidSeriesError) as exc:
        timeseries.forecast(_series(), horizon=3, freq="D", target_columns=["y"])

    assert str(exc.value) == "task.horizon: too large"


def _mock_upload_then_fail(server: MockTabPFNServer) -> tuple[respx.Route, ...]:
    router = server.router
    prepare = router.post("/forecast/prepare_upload").respond(
        200,
        json={
            "job_id": str(JOB_ID),
            "series_info": {
                "signed_urls": [f"{server.base_url}/signed/series.parquet"],
                "expires_at": 0.0,
                "required_headers": {"X-Goog-Content-Length-Range": "0,10"},
            },
        },
    )
    put = router.put("/signed/series.parquet").respond(200)
    submit = router.post("/forecast/submit_job").respond(
        200, json={"job_id": str(JOB_ID)}
    )
    router.post("/forecast/get_job_result").respond(
        200,
        json={
            "status": "failed",
            "job_id": str(JOB_ID),
            "error": "stop here",
            "user_error": False,
        },
    )

    return prepare, put, submit


def test__forecast__table_over_inline_limit__uploads_it_then_submits_the_job_id(
    server,
):
    prepare, put, submit = _mock_upload_then_fail(server)

    with (
        patch.object(
            ServiceClient,
            "get_settings",
            return_value=_settings(max_inline_table_bytes=1),
        ),
        pytest.raises(TimeSeriesJobFailedError),
    ):
        timeseries.forecast(_series(), horizon=3, freq="D", target_columns=["y"])

    assert b'"format":"parquet"' in prepare.calls.last.request.content
    assert put.calls.last.request.headers["X-Goog-Content-Length-Range"] == "0,10"
    body = submit.calls.last.request.content
    assert str(JOB_ID).encode() in body
    assert b"series.parquet" not in body


def test__forecast__bad_table__raises_before_any_request(server):
    # The mock server has no routes, so any request fails with another error.
    with pytest.raises(InvalidSeriesError):
        timeseries.forecast(
            _series().drop(index=2), horizon=3, freq="D", target_columns=["y"]
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"horizon": 0}, "at least 1"),
        ({"horizon": 65}, "at most 64"),
        ({"quantiles": [0.15]}, "quantiles must be from"),
        ({"quantiles": []}, "at least one quantile"),
    ],
    ids=["horizon_zero", "horizon_too_long", "off_grid_quantile", "no_quantile"],
)
def test__forecast__bad_parameters__raises(server, kwargs, match):
    args = {"horizon": 3, **kwargs}
    with pytest.raises(InvalidSeriesError, match=match):
        timeseries.forecast(_series(), freq="D", target_columns=["y"], **args)


def test__forecast__no_settings__uploads_and_leaves_limit_checks_to_the_server(
    server,
):
    prepare, _, submit = _mock_upload_then_fail(server)

    with (
        patch.object(ServiceClient, "get_settings", return_value=None),
        pytest.raises(TimeSeriesJobFailedError),
    ):
        timeseries.forecast(_series(), horizon=5000, freq="D", target_columns=["y"])

    assert prepare.called
    form = parse_qs(submit.calls.last.request.content.decode())
    assert json.loads(form["spec"][0])["task"] == {"task": "forecast", "horizon": 5000}


def _renamed(column: str, new: str) -> dict:
    roles = {"item_id": "id_column", "timestamp": "timestamp_column"}
    match column:
        case "y":
            return {
                "series": _series().rename(columns={"y": new}),
                "target_columns": [new],
            }
        case _:
            return {
                "series": _series().rename(columns={column: new}),
                "target_columns": ["y"],
                roles[column]: new,
            }


@pytest.mark.parametrize(
    ("column", "new", "quantiles"),
    [
        ("item_id", "mean", [0.5]),
        ("timestamp", "target_column", [0.5]),
        ("y", "q_0.5", [0.5]),
        ("y", "q_0.9", None),
    ],
    ids=["id_mean", "timestamp_target_column", "target_requested_q", "target_grid_q"],
)
def test__forecast__column_name_taken_by_result__raises_before_any_request(
    server, column, new, quantiles
):
    with pytest.raises(InvalidSeriesError, match=f"\\['{new}'\\] are taken"):
        timeseries.forecast(
            horizon=3, freq="D", quantiles=quantiles, **_renamed(column, new)
        )


def test__forecast__quantile_column_name_not_requested__submits(server):
    submit = server.router.post("/forecast/submit_job").respond(
        422, json={"detail": "stop here"}
    )

    with pytest.raises(InvalidSeriesError, match="stop here"):
        timeseries.forecast(
            horizon=3, freq="D", quantiles=[0.5], **_renamed("y", "q_0.9")
        )

    assert submit.called
