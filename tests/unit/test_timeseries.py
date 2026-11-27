#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Forecasts: the structure checks, and the request, poll and result over HTTP.

HTTP runs through the respx mock server, which fails a test on any request it
has no route for, so a check that raises first proves no request went out. The
poll loop runs on a fake clock, so nothing sleeps.
"""

from __future__ import annotations

import base64
import io
import json
import pickle
from collections.abc import Iterator
from email.parser import BytesParser
from email.policy import default as default_policy
from typing import Any
from unittest.mock import patch
from urllib.parse import parse_qs
from uuid import UUID

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from tabpfn_client import timeseries
from tabpfn_client.api_models import ForecastLimits, GetSettingsResponse
from tabpfn_client.client import ResolvedAsyncSettings, ServiceClient
from tabpfn_client.errors import (
    InvalidSeriesError,
    TimeSeriesInputParseError,
    TimeSeriesInputValidationError,
    TimeSeriesJobFailedError,
)
from tabpfn_client.timeseries import ForecastMetadata, ForecastResult
from tests.mock_tabpfn_server import MockTabPFNServer
from tests.unit.test_client import _api_settings_payload, _FakeTime

JOB_ID = UUID("00000000-0000-0000-0000-0000000000f1")
UPLOAD_ID = UUID("00000000-0000-0000-0000-0000000000e1")
HORIZON = 3
GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
METADATA = {"n_items": 2, "runtime_s": 1.5}
# Signed URLs, on the mock server's host so that its router answers them.
STORAGE = f"{MockTabPFNServer().base_url}/signed"


def _limits(**overrides: Any) -> ForecastLimits:
    return ForecastLimits(
        **{
            "max_horizon": 64,
            "max_inline_table_bytes": 1_000_000,
            "max_items": 100,
            "max_rows": 10_000,
            "quantiles": GRID,
            "max_targets": 2,
            "max_cells": 30_000,
            "max_output_rows": 10_000,
            "max_file_bytes": 1_000_000,
            "max_decoded_bytes": 4_000_000,
            "max_covariates": 3,
            **overrides,
        }
    )


def _settings(**limits: Any) -> GetSettingsResponse:
    return GetSettingsResponse(**_api_settings_payload(), forecast=_limits(**limits))


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
        price=df["y"] * 2,
        promo=df["y"] % 2,
        region=df["item_id"].map({"i0": "a", "i1": "b"}),
    )


def _known() -> pd.DataFrame:
    """The next `HORIZON` days of every item, with one known covariate."""
    steps = pd.date_range("2024-01-07", periods=HORIZON, freq="D")
    return pd.DataFrame(
        {
            "item_id": np.repeat(["i0", "i1"], HORIZON),
            "timestamp": np.tile(steps, 2),
            "promo": 1.0,
        }
    )


def _forecast(series: pd.DataFrame | None = None, **kwargs: Any) -> ForecastResult:
    args: dict[str, Any] = {"horizon": HORIZON, "freq": "D", "target_columns": ["y"]}
    return timeseries.forecast(
        _series() if series is None else series, **(args | kwargs)
    )


def _forecast_with_covariates(**kwargs: Any) -> ForecastResult:
    args: dict[str, Any] = {
        "past_covariate_columns": ["price"],
        "static_covariate_columns": ["region"],
        "known_covariates": _known(),
    }
    return _forecast(_covariate_series(), **(args | kwargs))


# -- What the server sends. --


def _result_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["i0", "i1"],
            "timestamp": pd.to_datetime(["2024-01-07", "2024-01-07"]),
            "target_column": ["y", "y"],
            "mean": np.array([1.0, 2.0], dtype=np.float32),
            "q_0.5": np.array([1.0, 2.0], dtype=np.float32),
        }
    )


def _result_parquet() -> bytes:
    buf = io.BytesIO()
    _result_frame().to_parquet(buf, index=False)
    return buf.getvalue()


def _pending(retry_in_secs: float = 2.0) -> dict[str, Any]:
    return {
        "status": "pending",
        "job_id": str(JOB_ID),
        "retry_in_secs": retry_in_secs,
    }


def _inline() -> dict[str, Any]:
    # URL-safe, as the server's pydantic encodes bytes in JSON.
    parquet = base64.urlsafe_b64encode(_result_parquet()).decode()
    return {
        "status": "succeeded",
        "job_id": str(JOB_ID),
        "metadata": METADATA,
        "result": {"kind": "inline", "parquet": parquet},
    }


def _stored() -> dict[str, Any]:
    return {
        "status": "succeeded",
        "job_id": str(JOB_ID),
        "metadata": METADATA,
        "result": {"kind": "stored", "url": f"{STORAGE}/result.parquet"},
    }


def _failed(error_kind: str | None) -> dict[str, Any]:
    return {
        "status": "failed",
        "job_id": str(JOB_ID),
        "error": "the reason",
        "error_kind": error_kind,
    }


def _error(status_code: int, error_code: str, *, job: bool = False) -> httpx.Response:
    """An error before the keepalive started, with the job's id if one exists."""
    return httpx.Response(
        status_code,
        json={"message": "the reason", "error_code": error_code, "trace_id": "t"},
        headers={"X-Job-Id": str(JOB_ID)} if job else {},
    )


def _streamed_error(status_code: int, error_code: str) -> httpx.Response:
    """An error after the keepalive started: in a 200 body, with the job's id."""
    envelope = {
        "message": "the reason",
        "error_code": error_code,
        "_streamed_error": True,
        "status_code": status_code,
        "detail": "the reason",
    }
    return httpx.Response(
        200,
        content=b"   " + json.dumps(envelope).encode(),
        headers={"content-type": "application/json", "X-Job-Id": str(JOB_ID)},
    )


def _upload_info(name: str, length_range: str) -> dict[str, Any]:
    return {
        "signed_urls": [f"{STORAGE}/{name}.parquet"],
        "expires_at": 0.0,
        "required_headers": {"X-Goog-Content-Length-Range": length_range},
    }


# -- What the client sends. --


def _form(request: httpx.Request) -> dict[str, bytes]:
    """The form fields of a request, multipart or URL-encoded."""
    content_type = request.headers["content-type"]
    if not content_type.startswith("multipart/form-data"):
        fields = parse_qs(request.content.decode())
        return {name: values[0].encode() for name, values in fields.items()}
    message = BytesParser(policy=default_policy).parsebytes(
        f"Content-Type: {content_type}\r\n\r\n".encode() + request.content
    )
    return {
        part.get_param("name", header="content-disposition"): part.get_payload(
            decode=True
        )
        for part in message.iter_parts()
    }


def _spec(route: respx.Route) -> dict[str, Any]:
    return json.loads(_form(route.calls.last.request)["spec"])


# -- Client state. --


@pytest.fixture(autouse=True)
def authorized_client() -> Iterator[None]:
    ServiceClient.reset_authorization()
    ServiceClient.authorize("dummy_token")
    with (
        patch.object(timeseries, "init"),
        patch.object(ServiceClient, "get_settings", return_value=_settings()),
    ):
        yield
    ServiceClient.reset_authorization()


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with MockTabPFNServer() as server:
        assert server.router is not None
        yield server.router


@pytest.fixture
def clock() -> Iterator[_FakeTime]:
    fake_time = _FakeTime()
    with (
        patch("tabpfn_client.client.time", fake_time),
        patch.object(
            ServiceClient,
            "_resolve_async_settings",
            return_value=ResolvedAsyncSettings(
                use_above_trainset_size_bytes=0, poll_timeout_secs=600.0
            ),
        ),
    ):
        yield fake_time


def _with_settings(**limits: Any):
    return patch.object(ServiceClient, "get_settings", return_value=_settings(**limits))


# -- Tests. --


class TestStructureChecks:
    """Each check raises before any request: the mock server has no routes."""

    @pytest.mark.parametrize(
        ("series", "kwargs", "match"),
        [
            (_series().drop(columns="y"), {}, r"no column\(s\) \['y'\]"),
            (_series().assign(extra=1), {}, r"column\(s\) \['extra'\], which the job"),
            (
                pd.concat([_series(), _series()[["y"]]], axis=1),
                {},
                r"more than one column named \['y'\]",
            ),
            (_series(), {"target_columns": []}, "at least one target"),
            (
                _series().assign(z=1.0, w=1.0),
                {"target_columns": ["y", "z", "w"]},
                "at most 2. Name fewer",
            ),
            (
                _series().assign(price=1.0),
                {"past_covariate_columns": ["price"], "target_columns": ["y", "price"]},
                r"\['price'\] have more than one role",
            ),
            (
                _covariate_series().assign(other=1.0),
                {
                    "past_covariate_columns": ["price", "promo"],
                    "static_covariate_columns": ["region", "other"],
                },
                "4 covariates, and a job can have at most 3",
            ),
            (_series(), {"horizon": 0}, "horizon is 0; it must be from 1 to 64"),
            (_series(), {"horizon": 65}, "horizon is 65; it must be from 1 to 64"),
            (_series(), {"quantiles": []}, "at least one quantile"),
            (_series(), {"quantiles": [0.5, 0.15]}, r"Quantiles \[0.15\] are not"),
        ],
        ids=[
            "missing_column",
            "extra_column",
            "duplicate_column",
            "no_target",
            "too_many_targets",
            "column_in_two_roles",
            "too_many_covariates",
            "horizon_zero",
            "horizon_too_long",
            "no_quantile",
            "off_grid_quantile",
        ],
    )
    def test_raises_before_any_request(self, router, series, kwargs, match):
        with pytest.raises(InvalidSeriesError, match=match):
            _forecast(series, **kwargs)

    @pytest.mark.parametrize(
        "name", ["target_column", "mean", "cutoff", "actual", "q_0.5", "q_custom"]
    )
    @pytest.mark.parametrize("role", ["id", "timestamp"])
    def test_refuses_a_reserved_id_or_timestamp_name(self, router, name, role):
        column = "item_id" if role == "id" else "timestamp"
        kwargs = {f"{role}_column": name}

        with pytest.raises(InvalidSeriesError, match=rf"\['{name}'\] has a reserved"):
            _forecast(_series().rename(columns={column: name}), **kwargs)

    @pytest.mark.parametrize(
        "name", ["target_column", "mean", "cutoff", "actual", "q_0.5", "q_custom"]
    )
    @pytest.mark.parametrize("role", ["target", "covariate"])
    def test_accepts_a_reserved_target_or_covariate_name(self, router, name, role):
        route = router.post("/forecast/forecast").respond(200, json=_inline())
        series = _series().assign(price=1.0)
        match role:
            case "target":
                series = series.rename(columns={"y": name})
                kwargs = {"target_columns": [name], "past_covariate_columns": ["price"]}
            case _:
                series = series.rename(columns={"price": name})
                kwargs = {"past_covariate_columns": [name]}

        _forecast(series, **kwargs)

        assert route.called

    @pytest.mark.parametrize(
        ("known", "match"),
        [
            (_known().drop(columns="timestamp"), r"no column\(s\) \['timestamp'\]"),
            (_known().drop(columns="promo"), "no covariate column"),
            (
                pd.concat([_known(), _known()[["promo"]]], axis=1),
                r"known covariates table has more than one column named \['promo'\]",
            ),
            (
                _known().rename(columns={"promo": "holiday"}),
                r"series table has no column\(s\) \['holiday'\]",
            ),
        ],
        ids=["no_timestamp", "no_covariate", "duplicate_column", "not_in_series"],
    )
    def test_refuses_a_known_covariates_table_without_the_columns_it_needs(
        self, router, known, match
    ):
        with pytest.raises(InvalidSeriesError, match=match):
            _forecast_with_covariates(known_covariates=known)

    def test_leaves_the_values_to_the_server(self, router):
        series = _series()
        # A NaN target, a gap and a duplicate timestamp: the worker refuses them.
        series.loc[0, "y"] = np.nan
        series = pd.concat([series.drop(index=2), series.iloc[[1]]])
        route = router.post("/forecast/forecast").respond(200, json=_inline())

        _forecast(series)

        assert route.called


class TestLimits:
    def test_raises_when_the_server_offers_no_forecasts(self, router):
        settings = GetSettingsResponse(**_api_settings_payload())
        with (
            patch.object(ServiceClient, "get_settings", return_value=settings),
            pytest.raises(RuntimeError, match="does not offer forecasts"),
        ):
            _forecast()

    def test_raises_when_the_settings_are_unavailable(self, router):
        with (
            patch.object(ServiceClient, "get_settings", return_value=None),
            pytest.raises(RuntimeError, match="Could not read the settings"),
        ):
            _forecast()

    def test_settings_carry_the_forecast_limits(self):
        payload = _api_settings_payload() | {"forecast": _limits().model_dump()}

        settings = GetSettingsResponse.model_validate(payload)

        assert settings.forecast == _limits()


class TestSyncForecast:
    def test_returns_an_inline_result_with_typed_metadata(self, router):
        route = router.post("/forecast/forecast").respond(200, json=_inline())

        result = _forecast(quantiles=[0.5])

        pd.testing.assert_frame_equal(result.forecast, _result_frame())
        assert result.metadata == ForecastMetadata(n_items=2, runtime_s=1.5)
        form = _form(route.calls.last.request)
        assert set(form) == {"spec", "series"}
        pd.testing.assert_frame_equal(
            pd.read_parquet(io.BytesIO(form["series"])), _series()
        )
        assert _spec(route) == {
            "freq": "D",
            "n_items": 2,
            "columns": {
                "id": "item_id",
                "timestamp": "timestamp",
                "targets": ["y"],
                "past_covariates": [],
                "known_covariates": [],
                "static_covariates": [],
            },
            "task": {"task": "forecast", "horizon": HORIZON, "quantiles": [0.5]},
        }

    def test__forecast__daily_series_without_freq__sends_d(self, router):
        route = router.post("/forecast/forecast").respond(200, json=_inline())

        _forecast(freq=None)

        assert _spec(route)["freq"] == "D"

    def test__forecast__hourly_series_without_freq__sends_h(self, router):
        route = router.post("/forecast/forecast").respond(200, json=_inline())
        series = _series()
        series["timestamp"] = pd.date_range("2024-01-01", periods=len(series), freq="h")

        _forecast(series, freq=None)

        assert _spec(route)["freq"] == "h"

    def test__forecast__items_disagree_on_freq__raises_without_request(self):
        hourly = _series(n_items=1).assign(item_id="i1")
        hourly["timestamp"] = pd.date_range("2024-01-01", periods=len(hourly), freq="h")
        series = pd.concat([_series(n_items=1), hourly], ignore_index=True)

        with pytest.raises(InvalidSeriesError, match="items disagree"):
            _forecast(series, freq=None)

    def test__forecast__irregular_timestamps_without_freq__raises(self):
        series = _series()
        series.loc[series.index[2], "timestamp"] += pd.Timedelta(hours=5)

        with pytest.raises(InvalidSeriesError, match="irregular"):
            _forecast(series, freq=None)

    def test__forecast__explicit_freq__sends_it_as_given(self, router):
        route = router.post("/forecast/forecast").respond(200, json=_inline())

        _forecast(freq="W")

        assert _spec(route)["freq"] == "W"

    def test_sends_the_count_of_distinct_ids_as_n_items(self, router):
        route = router.post("/forecast/forecast").respond(200, json=_inline())
        # Rows of one item apart from each other still count once.
        series = _series(n_items=3).sample(frac=1, random_state=0)

        _forecast(series)

        assert _spec(route)["n_items"] == 3

    def test_asks_for_the_whole_grid_without_quantiles(self, router):
        route = router.post("/forecast/forecast").respond(200, json=_inline())

        _forecast()

        assert _spec(route)["task"] == {"task": "forecast", "horizon": HORIZON}

    def test_sends_every_covariate_kind_and_both_tables(self, router):
        route = router.post("/forecast/forecast").respond(200, json=_inline())

        _forecast_with_covariates()

        assert _spec(route)["columns"] == {
            "id": "item_id",
            "timestamp": "timestamp",
            "targets": ["y"],
            "past_covariates": ["price"],
            "known_covariates": ["promo"],
            "static_covariates": ["region"],
        }
        form = _form(route.calls.last.request)
        pd.testing.assert_frame_equal(
            pd.read_parquet(io.BytesIO(form["known_covariates"])), _known()
        )

    def test_polls_a_pending_job_and_downloads_its_stored_result(self, router, clock):
        router.post("/forecast/forecast").respond(200, json=_pending())
        poll = router.post("/forecast/get_job_result")
        poll.side_effect = [
            httpx.Response(200, json=_pending(9.0)),
            httpx.Response(503),
            httpx.Response(200, json=_pending(0.5)),
            httpx.Response(200, json=_stored()),
        ]
        download = router.get(f"{STORAGE}/result.parquet").respond(
            200, content=_result_parquet()
        )

        result = _forecast()

        pd.testing.assert_frame_equal(result.forecast, _result_frame())
        assert result.metadata == ForecastMetadata(**METADATA)
        assert [json.loads(c.request.content) for c in poll.calls] == [
            {"job_id": str(JOB_ID)}
        ] * 4
        assert download.called
        # The first poll waits for the hint of the request's answer, and each
        # later one for the hint before it, at least 5 s after a failed poll.
        assert clock.sleeps == [2.0, 9.0, 9.0, 0.5]


class TestFailures:
    @pytest.mark.parametrize(
        ("error_kind", "error_type"),
        [
            ("parse", TimeSeriesInputParseError),
            ("validation", TimeSeriesInputValidationError),
            (None, TimeSeriesJobFailedError),
        ],
        ids=["parse", "validation", "service"],
    )
    def test_a_polled_job_raises_its_error_kind(
        self, router, clock, error_kind, error_type
    ):
        router.post("/forecast/forecast").respond(200, json=_pending())
        router.post("/forecast/get_job_result").respond(200, json=_failed(error_kind))

        with pytest.raises(TimeSeriesJobFailedError, match="the reason") as exc:
            _forecast()

        assert type(exc.value) is error_type
        assert exc.value.job_id == JOB_ID

    @pytest.mark.parametrize(
        ("response", "error_type"),
        [
            (_error(422, "INPUT_PARSE_ERROR", job=True), TimeSeriesInputParseError),
            (
                _error(422, "INPUT_VALIDATION_ERROR", job=True),
                TimeSeriesInputValidationError,
            ),
            (
                _streamed_error(422, "INPUT_PARSE_ERROR"),
                TimeSeriesInputParseError,
            ),
            (
                _streamed_error(422, "INPUT_VALIDATION_ERROR"),
                TimeSeriesInputValidationError,
            ),
            (_error(500, "INTERNAL_ERROR", job=True), TimeSeriesJobFailedError),
            (_streamed_error(504, "DEADLINE_EXCEEDED"), TimeSeriesJobFailedError),
            (_error(503, "BUSY", job=True), TimeSeriesJobFailedError),
        ],
        ids=[
            "parse",
            "validation",
            "streamed_parse",
            "streamed_validation",
            "internal",
            "streamed_deadline",
            "busy",
        ],
    )
    def test_a_job_that_fails_while_the_request_waits_raises_its_error(
        self, router, response, error_type
    ):
        router.post("/forecast/forecast").mock(return_value=response)

        with pytest.raises(TimeSeriesJobFailedError, match="the reason") as exc:
            _forecast()

        assert type(exc.value) is error_type
        assert exc.value.job_id == JOB_ID

    @pytest.mark.parametrize(
        "response",
        [_error(422, "USER_ERROR"), _error(413, "USER_ERROR")],
        ids=["refused", "too_large"],
    )
    def test_a_request_refused_before_any_job_raises_invalid_series(
        self, router, response
    ):
        router.post("/forecast/forecast").mock(return_value=response)

        with pytest.raises(InvalidSeriesError, match="the reason"):
            _forecast()

    def test_a_job_error_can_be_pickled_with_its_job_id(self):
        error = TimeSeriesInputParseError("the reason", job_id=JOB_ID)

        copy = pickle.loads(pickle.dumps(error))

        assert type(copy) is TimeSeriesInputParseError
        assert str(copy) == "the reason"
        assert copy.job_id == JOB_ID

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(
                429,
                json={
                    "message": "Monthly token quota exceeded.",
                    "error_code": "USAGE_LIMIT_EXCEEDED",
                },
            ),
            httpx.Response(
                429,
                json={
                    "detail": "Rate limit exceeded: at most 60 forecast requests "
                    "per minute are allowed. Retry in 12s."
                },
                headers={"Retry-After": "12"},
            ),
        ],
        ids=["quota", "rate_limit"],
    )
    def test_a_quota_or_rate_limit_raises_the_error_of_a_prediction(
        self, router, response
    ):
        router.post("/forecast/forecast").mock(return_value=response)

        with pytest.raises(
            RuntimeError, match=r"Fail to call forecast: \[HTTP 429\]"
        ) as exc:
            _forecast()

        assert type(exc.value) is RuntimeError
        assert not isinstance(exc.value, (InvalidSeriesError, TimeSeriesJobFailedError))

    def test_an_unknown_job_raises_at_once(self, router, clock):
        router.post("/forecast/forecast").respond(200, json=_pending())
        poll = router.post("/forecast/get_job_result").respond(
            404, json={"message": "Forecast job not found.", "error_code": "NOT_FOUND"}
        )

        with pytest.raises(RuntimeError, match="Forecast job not found"):
            _forecast()

        assert poll.call_count == 1


class TestUpload:
    def test_uploads_a_series_over_the_inline_limit(self, router):
        prepare = router.post("/forecast/prepare_upload").respond(
            200,
            json={
                "upload_id": str(UPLOAD_ID),
                "series_info": _upload_info("series", "0,10"),
            },
        )
        put = router.put(f"{STORAGE}/series.parquet").respond(200)
        route = router.post("/forecast/forecast").respond(200, json=_inline())
        size = len(timeseries._parquet(_series()))

        with _with_settings(max_inline_table_bytes=size - 1):
            _forecast()

        assert json.loads(prepare.calls.last.request.content) == {
            "series_info": {"format": "parquet", "size_bytes": size}
        }
        assert put.calls.last.request.headers["X-Goog-Content-Length-Range"] == "0,10"
        pd.testing.assert_frame_equal(
            pd.read_parquet(io.BytesIO(put.calls.last.request.content)), _series()
        )
        form = _form(route.calls.last.request)
        assert set(form) == {"spec", "upload_id"}
        assert form["upload_id"] == str(UPLOAD_ID).encode()

    def test_sends_a_series_at_the_inline_limit_inline(self, router):
        route = router.post("/forecast/forecast").respond(200, json=_inline())
        size = len(timeseries._parquet(_series()))

        with _with_settings(max_inline_table_bytes=size):
            _forecast()

        assert set(_form(route.calls.last.request)) == {"spec", "series"}

    def test_uploads_both_tables_when_together_they_are_over_the_limit(self, router):
        prepare = router.post("/forecast/prepare_upload").respond(
            200,
            json={
                "upload_id": str(UPLOAD_ID),
                "series_info": _upload_info("series", "0,10"),
                "known_covariates_info": _upload_info("known_covariates", "0,20"),
            },
        )
        put_series = router.put(f"{STORAGE}/series.parquet").respond(200)
        put_known = router.put(f"{STORAGE}/known_covariates.parquet").respond(200)
        route = router.post("/forecast/forecast").respond(200, json=_inline())
        sizes = [len(timeseries._parquet(t)) for t in (_covariate_series(), _known())]

        # Each table fits inline alone; the two together do not.
        with _with_settings(max_inline_table_bytes=max(sizes)):
            _forecast_with_covariates()

        assert json.loads(prepare.calls.last.request.content) == {
            "series_info": {"format": "parquet", "size_bytes": sizes[0]},
            "known_covariates_info": {"format": "parquet", "size_bytes": sizes[1]},
        }
        assert put_series.called
        assert (
            put_known.calls.last.request.headers["X-Goog-Content-Length-Range"]
            == "0,20"
        )
        pd.testing.assert_frame_equal(
            pd.read_parquet(io.BytesIO(put_known.calls.last.request.content)),
            _known(),
        )
        form = _form(route.calls.last.request)
        assert set(form) == {"spec", "upload_id"}
        assert _spec(route)["columns"]["known_covariates"] == ["promo"]
