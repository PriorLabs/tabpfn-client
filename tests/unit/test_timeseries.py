"""Time-series jobs: the local checks, and submit, poll and download over HTTP."""

from __future__ import annotations

import io
from collections.abc import Iterator
from unittest.mock import patch
from uuid import UUID

import numpy as np
import pandas as pd
import pytest
import respx

from tabpfn_client import timeseries
from tabpfn_client.client import ServiceClient
from tabpfn_client.timeseries import (
    InvalidSeriesError,
    TimeSeriesJobFailedError,
    validate_series,
)
from tests.mock_tabpfn_server import MockTabPFNServer

JOB_ID = UUID("00000000-0000-0000-0000-0000000000f1")
COLS = dict(id_column="item_id", timestamp_column="timestamp", target_columns=["y"])


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
        (lambda df: df.assign(extra=1), "no role"),
        (lambda df: df.assign(y=df["y"].astype(str)), "not numeric"),
        (lambda df: df.assign(y=df["y"].where(df.index != 3)), "NaN or inf"),
        (lambda df: df.assign(timestamp="not a date"), "parse as timestamps"),
        (lambda df: df.drop(index=2), "gaps"),
        (lambda df: pd.concat([df, df.iloc[[0]]]), "duplicate"),
    ],
    ids=["missing", "extra", "text", "nan", "bad_ts", "gap", "duplicate"],
)
def test__validate_series__bad_table__raises(mutate, match):
    with pytest.raises(InvalidSeriesError, match=match):
        validate_series(mutate(_series()), freq="D", **COLS)


def test__validate_series__off_grid_freq__raises():
    with pytest.raises(InvalidSeriesError, match="grid"):
        validate_series(_series(), freq="h", **COLS)


def test__validate_series__unknown_freq__raises():
    with pytest.raises(InvalidSeriesError, match="offset alias"):
        validate_series(_series(), freq="fortnight", **COLS)


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
    submit = router.post("/forecast/v0/submit_job").respond(
        200, json={"job_id": str(JOB_ID)}
    )
    url = f"{server.base_url}/signed/result.parquet"
    router.post("/forecast/v0/get_job_result").mock(
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
    server.router.post("/forecast/v0/submit_job").respond(
        200, json={"job_id": str(JOB_ID)}
    )
    server.router.post("/forecast/v0/get_job_result").respond(
        200,
        json={
            "status": "failed",
            "job_id": str(JOB_ID),
            "error": "preset 'medium' is not available",
            "user_error": True,
        },
    )

    with pytest.raises(TimeSeriesJobFailedError, match="medium") as exc:
        timeseries.forecast(_series(), horizon=3, freq="D", target_columns=["y"])

    assert exc.value.user_error is True
    assert exc.value.job_id == JOB_ID


def test__forecast__server_rejects_spec__raises_invalid_series(server):
    server.router.post("/forecast/v0/submit_job").respond(
        422, json={"detail": "task.horizon: too large"}
    )

    with pytest.raises(InvalidSeriesError, match="too large"):
        timeseries.forecast(_series(), horizon=3, freq="D", target_columns=["y"])


def test__forecast__bad_table__raises_before_any_request(server):
    # The mock server has no routes, so any request fails with another error.
    with pytest.raises(InvalidSeriesError):
        timeseries.forecast(
            _series().drop(index=2), horizon=3, freq="D", target_columns=["y"]
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"horizon": 0}, {"quantiles": [0.15]}, {"quantiles": []}],
    ids=["horizon", "off_grid_quantile", "no_quantile"],
)
def test__forecast__bad_parameters__raises(kwargs):
    args = {"horizon": 3, **kwargs}
    with pytest.raises(InvalidSeriesError):
        timeseries.forecast(_series(), freq="D", target_columns=["y"], **args)
