"""Time-series jobs: the local checks, and submit, poll and download over HTTP."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterator
from typing import Any
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
    TimeSeriesInputParseError,
    TimeSeriesInputValidationError,
    TimeSeriesJobFailedError,
    validate_known_covariates,
    validate_series,
)
from tests.mock_tabpfn_server import MockTabPFNServer
from tests.unit.test_client import _api_settings_payload

JOB_ID = UUID("00000000-0000-0000-0000-0000000000f1")
COLS: dict[str, Any] = dict(
    id_column="item_id", timestamp_column="timestamp", target_columns=["y"]
)


GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _settings(**limits) -> GetSettingsResponse:
    forecast = ForecastLimits(
        **{
            "max_horizon": 64,
            "max_inline_table_bytes": 1_000_000,
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


def _covariate_series() -> pd.DataFrame:
    df = _series()
    return df.assign(
        price=df["y"] * 2, promo=df["y"] % 2, region=df["item_id"].map({"i0": "a"})
    ).assign(region=lambda d: d["region"].fillna("b"))


COVARIATES: dict[str, Any] = dict(
    past_covariate_columns=["price"],
    static_covariate_columns=["region"],
)
HORIZON = 3


def _known() -> pd.DataFrame:
    """The next `HORIZON` days of every item, with one known covariate."""
    last = _series()["timestamp"].max()
    steps = pd.date_range(last + pd.Timedelta(days=1), periods=HORIZON, freq="D")
    items = sorted(_series()["item_id"].unique())
    return pd.DataFrame(
        {
            "item_id": np.repeat(items, HORIZON),
            "timestamp": np.tile(steps, len(items)),
            "promo": 1.0,
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
        (lambda df: df.iloc[:0], "The series table has no rows"),
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
        "empty",
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


def test__validate_series__items_without_data__raises_naming_them():
    df = _series(n_items=3).assign(y=lambda d: d["y"].mask(d["item_id"] != "i1"))
    with pytest.raises(InvalidSeriesError) as e:
        validate_series(df, freq="D", **COLS)
    assert str(e.value) == (
        "2 item(s) have no data: 'i0', 'i2'. Every item needs at least one observed "
        "(non-NaN) value in each target. Drop these items, or send their values."
    )


def test__validate_series__items_without_data__names_ten_and_counts_the_rest():
    df = _series(n_items=13).assign(y=float("nan"))
    with pytest.raises(InvalidSeriesError, match="13 item") as e:
        validate_series(df, freq="D", **COLS)
    assert "'i6' and 3 more." in str(e.value)
    assert "'i7'" not in str(e.value)


def test__validate_series__one_target_empty_for_an_item__raises():
    df = _series().assign(z=lambda d: d["y"].mask(d["item_id"] == "i0"))
    with pytest.raises(InvalidSeriesError, match=r"1 item\(s\) have no data: 'i0'"):
        validate_series(df, freq="D", **{**COLS, "target_columns": ["y", "z"]})


# The worker takes a target of any type `is_numeric_dtype` accepts, bools included.
_NUMBER_TARGETS = {
    "int64": lambda y: y.astype("int64"),
    "float32": lambda y: y.astype("float32"),
    "bool": lambda y: y > 3,
    "Int64": lambda y: y.astype("Int64"),
    "Float64": lambda y: y.astype("Float64"),
    "boolean": lambda y: (y > 3).astype("boolean"),
    "int64_pyarrow": lambda y: y.astype("int64[pyarrow]"),
    "double_pyarrow": lambda y: y.astype("double[pyarrow]"),
}


@pytest.mark.parametrize("to_dtype", _NUMBER_TARGETS.values(), ids=_NUMBER_TARGETS)
def test__validate_series__number_target_dtype__passes(to_dtype):
    df = _series()
    validate_series(df.assign(y=to_dtype(df["y"])), freq="D", **COLS)


@pytest.mark.parametrize(
    "to_dtype",
    [
        lambda y: y.astype("Int64").where(y != 3),
        lambda y: y.astype("Float64").where(y != 3),
        lambda y: (y > 3).astype("boolean").where(y != 3),
        lambda y: y.astype("double[pyarrow]").where(y != 3),
    ],
    ids=["Int64", "Float64", "boolean", "double_pyarrow"],
)
def test__validate_series__nullable_target_with_missing__raises_nan(to_dtype):
    df = _series()
    with pytest.raises(InvalidSeriesError, match="has NaN or inf values"):
        validate_series(df.assign(y=to_dtype(df["y"])), freq="D", **COLS)


_OTHER_TARGETS = {
    "string_python": lambda y: y.astype(str).astype("string[python]"),
    "string_pyarrow": lambda y: y.astype(str).astype("string[pyarrow]"),
    "categorical": lambda y: y.astype("category"),
    "object_str": lambda y: y.astype(str).astype(object),
    "object_bool": lambda y: (y > 3).astype(object),
    "object_none": lambda y: pd.Series([None] * len(y), dtype=object),
    "object_mixed": lambda y: y.astype(object).where(y != 3, "x"),
    "datetime": lambda y: pd.to_datetime(y, unit="D"),
}


@pytest.mark.parametrize("to_dtype", _OTHER_TARGETS.values(), ids=_OTHER_TARGETS)
def test__validate_series__non_number_target_dtype__raises(to_dtype):
    df = _series()
    with pytest.raises(InvalidSeriesError, match="'y' has type .*, not a number type"):
        validate_series(df.assign(y=to_dtype(df["y"])), freq="D", **COLS)


def test__validate_series__bad_target_and_timestamps__raises_target_first():
    # The worker's order, so both name the same fault.
    df = _series().assign(y="text", timestamp="not a date")
    with pytest.raises(InvalidSeriesError, match="not a number type"):
        validate_series(df, freq="fortnight", **COLS)


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


@pytest.mark.parametrize("column", ["item_id", "timestamp", "y"])
def test__validate_series__duplicate_column_name__raises_naming_it(column):
    df = _series()
    df = pd.concat([df, df[[column]]], axis=1)
    with pytest.raises(
        InvalidSeriesError, match=rf"more than one column named \['{column}'\]"
    ):
        validate_series(df, freq="D", **COLS)


def test__validate_series__column_in_two_roles__raises():
    with pytest.raises(InvalidSeriesError, match="one role"):
        validate_series(
            _series(),
            freq="D",
            id_column="item_id",
            timestamp_column="timestamp",
            target_columns=["y", "item_id"],
        )


def test__validate_series__all_covariate_kinds__passes():
    validate_series(
        _covariate_series(),
        freq="D",
        **COLS,
        **COVARIATES,
        known_covariate_columns=["promo"],
    )


def test__validate_series__covariate_missing_from_table__raises():
    with pytest.raises(InvalidSeriesError, match="no column"):
        validate_series(_series(), freq="D", **COLS, past_covariate_columns=["price"])


def test__validate_series__covariate_also_target__raises():
    with pytest.raises(InvalidSeriesError, match="one role"):
        validate_series(
            _covariate_series(), freq="D", **COLS, past_covariate_columns=["y"]
        )


def test__validate_series__unnamed_covariate_column__raises():
    with pytest.raises(InvalidSeriesError, match="which the job does not name"):
        validate_series(_covariate_series(), freq="D", **COLS)


@pytest.mark.parametrize(
    "region",
    [
        lambda d: d["item_id"] + d["timestamp"].dt.day.astype(str),
        lambda d: d["item_id"].where(d["timestamp"].dt.day != 2),
    ],
    ids=["two_values", "value_and_nan"],
)
def test__validate_series__static_covariate_varies_in_item__raises(region):
    df = _covariate_series()
    with pytest.raises(InvalidSeriesError, match="Static covariate 'region'"):
        validate_series(
            df.assign(region=region(df)),
            freq="D",
            **COLS,
            past_covariate_columns=["price"],
            known_covariate_columns=["promo"],
            static_covariate_columns=["region"],
        )


def test__validate_series__static_covariate_all_nan_in_item__passes():
    df = _covariate_series()
    validate_series(
        df.assign(region=df["region"].where(df["item_id"] != "i0")),
        freq="D",
        **COLS,
        past_covariate_columns=["price"],
        known_covariate_columns=["promo"],
        static_covariate_columns=["region"],
    )


# --- validate_known_covariates --------------------------------------------------------


def _check_known(known: pd.DataFrame) -> None:
    validate_known_covariates(
        known,
        _series(),
        horizon=HORIZON,
        freq="D",
        id_column="item_id",
        timestamp_column="timestamp",
    )


def test__validate_known_covariates__valid_table__passes():
    _check_known(_known().sample(frac=1, random_state=0))


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda k: k.drop(columns="item_id"), "no column"),
        (lambda k: k.drop(columns="promo"), "no covariate column"),
        (lambda k: pd.concat([k, k[["promo"]]], axis=1), "more than one column"),
        (lambda k: k.assign(timestamp="not a date"), "parse"),
        (
            lambda k: k.assign(timestamp=k["timestamp"].where(k.index != 3)),
            "missing timestamps",
        ),
        (lambda k: k[k["item_id"] != "i1"], r"no rows for item\(s\) \['i1'\]"),
        (lambda k: pd.concat([k, k.assign(item_id="i9")]), "that the series table"),
        (lambda k: k.iloc[1:], "Item 'i0' has 2 rows in the known covariates"),
        (lambda k: pd.concat([k, k.iloc[[0]]]), "duplicate timestamps"),
        (
            lambda k: k.assign(timestamp=k["timestamp"] + pd.Timedelta(days=1)),
            "2024-01-08 00:00:00 in the known covariates table where 2024-01-07",
        ),
        (
            lambda k: k.assign(
                timestamp=k["timestamp"].where(
                    k.index % HORIZON != 1, k["timestamp"] + pd.Timedelta(days=2)
                )
            ),
            "where 2024-01-08 00:00:00 is due",
        ),
    ],
    ids=[
        "no_id",
        "no_covariate",
        "duplicate_column",
        "bad_ts",
        "missing_ts",
        "missing_item",
        "extra_item",
        "too_few_rows",
        "duplicate_row",
        "starts_late",
        "gap",
    ],
)
def test__validate_known_covariates__bad_table__raises(mutate, match):
    with pytest.raises(InvalidSeriesError, match=match):
        _check_known(mutate(_known()))


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


@pytest.mark.parametrize(
    ("kind", "error_type"),
    [
        ("parse", TimeSeriesInputParseError),
        ("validation", TimeSeriesInputValidationError),
        (None, TimeSeriesJobFailedError),
    ],
    ids=["parse", "validation", "internal"],
)
def test__forecast__job_fails_with_error_kind__raises_matching_type(
    server, kind, error_type
):
    server.router.post("/forecast/submit_job").respond(
        200, json={"job_id": str(JOB_ID)}
    )
    server.router.post("/forecast/get_job_result").respond(
        200,
        json={
            "status": "failed",
            "job_id": str(JOB_ID),
            "error": "the reason",
            "error_kind": kind,
        },
    )

    with pytest.raises(TimeSeriesJobFailedError, match="the reason") as exc:
        timeseries.forecast(_series(), horizon=3, freq="D", target_columns=["y"])

    assert type(exc.value) is error_type
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
            "known_covariates_info": {
                "signed_urls": [f"{server.base_url}/signed/known_covariates.parquet"],
                "expires_at": 0.0,
                "required_headers": {"X-Goog-Content-Length-Range": "0,20"},
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


@pytest.mark.parametrize("column", ["item_id", "timestamp", "y"])
@pytest.mark.parametrize(
    "new", ["target_column", "mean", "cutoff", "actual", "q_0.5", "q_0.9", "q_custom"]
)
def test__forecast__column_with_a_reserved_name__raises_before_any_request(
    server, column, new
):
    with pytest.raises(
        InvalidSeriesError, match=re.escape(f"column(s) ['{new}'] have reserved names")
    ):
        timeseries.forecast(
            horizon=3, freq="D", quantiles=[0.5], **_renamed(column, new)
        )


# --- covariates over HTTP -----------------------------------------------------------

_ROLES = (
    b'"columns":{"id":"item_id","timestamp":"timestamp","targets":["y"],'
    b'"past_covariates":["price"],"known_covariates":["promo"],'
    b'"static_covariates":["region"]}'
)


def _forecast_with_covariates(**kwargs) -> None:
    with pytest.raises(TimeSeriesJobFailedError):
        timeseries.forecast(
            _covariate_series(),
            horizon=HORIZON,
            freq="D",
            target_columns=["y"],
            known_covariates=_known(),
            **COVARIATES,
            **kwargs,
        )


def test__forecast__all_covariate_kinds_inline__sends_roles_and_both_tables(server):
    submit = server.router.post("/forecast/submit_job").respond(
        200, json={"job_id": str(JOB_ID)}
    )
    server.router.post("/forecast/get_job_result").respond(
        200, json={"status": "failed", "job_id": str(JOB_ID), "error": "stop here"}
    )

    with patch.object(
        ServiceClient, "get_settings", return_value=_settings(covariates=True)
    ):
        _forecast_with_covariates()

    body = submit.calls.last.request.content
    assert _ROLES in body
    assert b'name="series"; filename="series.parquet"' in body
    assert b'name="known_covariates"; filename="known_covariates.parquet"' in body


def test__forecast__all_covariate_kinds_uploaded__uploads_both_tables(server):
    prepare, put, submit = _mock_upload_then_fail(server)
    put_known = server.router.put("/signed/known_covariates.parquet").respond(200)
    sizes = [len(timeseries._parquet(t)) for t in (_covariate_series(), _known())]

    # Each table fits inline alone, the two together do not.
    with patch.object(
        ServiceClient,
        "get_settings",
        return_value=_settings(covariates=True, max_inline_table_bytes=max(sizes)),
    ):
        _forecast_with_covariates()

    request = json.loads(prepare.calls.last.request.content)
    assert request == {
        "series_info": {"format": "parquet", "size_bytes": sizes[0]},
        "known_covariates_info": {"format": "parquet", "size_bytes": sizes[1]},
    }
    assert put.calls.last.request.headers["X-Goog-Content-Length-Range"] == "0,10"
    assert put_known.calls.last.request.headers["X-Goog-Content-Length-Range"] == "0,20"
    form = parse_qs(submit.calls.last.request.content.decode())
    assert form["job_id"] == [str(JOB_ID)]
    assert "parquet" not in submit.calls.last.request.content.decode()
    assert _ROLES.decode().replace(" ", "") in form["spec"][0]


def test__forecast__covariates_not_supported__raises_before_any_request(server):
    # The default settings have `covariates=False`; the mock server has no routes.
    with pytest.raises(InvalidSeriesError, match="Covariates are not supported yet"):
        timeseries.forecast(
            _covariate_series(),
            horizon=HORIZON,
            freq="D",
            target_columns=["y"],
            **COVARIATES,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"past_covariate_columns": ["price"]},
        {"static_covariate_columns": ["region"]},
        {"known_covariates": "table"},
    ],
    ids=["past", "static", "known"],
)
def test__forecast__any_covariate_kind_not_supported__raises(server, kwargs):
    if kwargs.get("known_covariates") == "table":
        kwargs = {"known_covariates": _known()}
    with pytest.raises(InvalidSeriesError, match="Covariates are not supported yet"):
        timeseries.forecast(
            _covariate_series(),
            horizon=HORIZON,
            freq="D",
            target_columns=["y"],
            **kwargs,
        )


def test__forecast__more_covariates_than_the_limit__raises(server):
    with (
        patch.object(
            ServiceClient,
            "get_settings",
            return_value=_settings(covariates=True, max_covariates=2),
        ),
        pytest.raises(InvalidSeriesError, match="at most 2 covariates"),
    ):
        _forecast_with_covariates()


@pytest.mark.parametrize("kind", ["past", "static", "known"])
def test__forecast__covariate_with_a_reserved_name__raises_before_any_request(
    server, kind
):
    series = _covariate_series()
    column = {"past": "price", "static": "region", "known": "promo"}[kind]
    known = _known()
    if kind == "known":
        known = known.rename(columns={"promo": "mean"})
    with (
        patch.object(
            ServiceClient, "get_settings", return_value=_settings(covariates=True)
        ),
        pytest.raises(
            InvalidSeriesError, match=r"column\(s\) \['mean'\] have reserved"
        ),
    ):
        timeseries.forecast(
            series.rename(columns={column: "mean"}),
            horizon=HORIZON,
            freq="D",
            target_columns=["y"],
            known_covariates=known,
            past_covariate_columns=["mean" if kind == "past" else "price"],
            static_covariate_columns=["mean" if kind == "static" else "region"],
        )
