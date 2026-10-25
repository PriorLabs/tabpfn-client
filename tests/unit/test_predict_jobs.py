#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Predictions as background jobs, at the transport layer: how `ServiceClient`
picks one request or a polled job, submits a job, polls for it, and maps what
the server reports, including a `/predict` the server stopped waiting for.

HTTP runs through the respx mock server, which fails a test on any request it
has no route for. The poll loop runs on a fake clock, so nothing sleeps.
"""

from __future__ import annotations

import copy
import json
import logging
import pickle
import time
from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID

import httpx
import numpy as np
import pytest
import respx

import tabpfn_client
from tabpfn_client import PredictFailedError, RefitRequiredError
from pydantic import ValidationError

from tabpfn_client.api_models import (
    ClassifierConfig,
    ClassifierTabPFNConfig,
    FitMode,
    FitResponse,
    GetSettingsResponse,
    PredictAsyncThresholds,
    PredictFailedErrorResponse,
)
from tabpfn_client.client import (
    _MIN_RETRY_INTERVAL_SECS,
    ResolvedAsyncSettings,
    ServiceClient,
)
from tabpfn_client.errors import (
    CappedRetryableServerError,
    PredictionGoneError,
    RetryableServerError,
    StreamedServerError,
)
from tabpfn_client.models import ApiMode, ClientOptions, PredictionResult
from tabpfn_client.options import get_opts
from tabpfn_client.service_wrapper import InferenceClient
from tests.mock_tabpfn_server import MockTabPFNServer
from tests.unit.test_client import _api_settings_payload, _FakeTime

FIT_ID = UUID("00000000-0000-0000-0000-000000000002")
UPLOAD_ID = "00000000-0000-0000-0000-000000000003"
PREDICTION_ID = UUID("00000000-0000-0000-0000-0000000000aa")
# Without dashes, as the server sends a trace id: a message must quote it in
# that form, which its logs and Sentry show.
TRACE_ID = "0000000000000000000000000000beef"

X_TEST = np.ones((3, 2))
PREDICTION = [1, 0, 1]
METADATA = {
    "task": "classification",
    "package_version": "0.0.0",
    "tabpfn_config": {},
    "test_set_num_rows": 3,
    "test_set_num_cols": 2,
}
TIMINGS = {
    "test_set_transform_queue_wait_s": 0.5,
    "test_set_transform_s": 1.0,
    "predict_queue_wait_s": 0.0,
    "predict_s": 2.0,
}
THRESHOLDS = {"train_rows": 50_000, "test_rows": 100_000}
THINKING_THRESHOLDS = {"train_rows": 25_000, "test_rows": 10_000}

# The server's own words.
NEEDS_REFITTING = (
    "This fitted model was created by an earlier version of the service and can "
    "no longer predict. Fit it again, then predict with the new "
    "fitted_train_set_id."
)
STALE_KV_CACHE = (
    "The KV cache of this fitted model is no longer valid because the server was "
    "upgraded. Please re-fit the model (call fit() again) before predicting."
)
MISSING_KV_CACHE = (
    "The cached state of this fitted model is no longer available. Please re-fit "
    "the model (call fit() again) before predicting."
)
EXPIRED = "This prediction's result is no longer stored. Predict again to get it."
# What a server without the endpoint answers, unlike the endpoints' own 404s:
# the error code is NOT_FOUND since tabpfn-server#1384, USER_ERROR before.
ROUTE_MISSING = pytest.mark.parametrize(
    "route_missing",
    [
        {"message": "Not Found", "error_code": "NOT_FOUND"},
        {"message": "Not Found", "error_code": "USER_ERROR"},
    ],
    ids=["route_missing", "route_missing_before_1384"],
)
NO_PREDICT_JOBS = (
    "This TabPFN server does not run predictions as background jobs; "
    "use predict() instead."
)
WAIT_BROKE_OFF = (
    f"Stopped waiting for prediction {PREDICTION_ID}. If it is still running, "
    "the server completes it and bills it even if nobody waits for it; its "
    "result stays on the server for about a day: call tabpfn_client.init(), "
    "then tabpfn_client.service_wrapper.InferenceClient.wait_for_predict("
    f'"{PREDICTION_ID}")'
)


# -- What the server sends. --


def _json(
    status_code: int,
    body: dict[str, Any],
    keepalive: bytes = b"",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """A JSON response, after the whitespace pings a slow answer starts with."""
    return httpx.Response(
        status_code,
        content=keepalive + json.dumps(body).encode(),
        headers={"content-type": "application/json", **(headers or {})},
    )


def _error(
    status_code: int,
    message: str,
    *,
    streamed: bool = False,
    error_code: str = "USER_ERROR",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """An error, sent at once or, after the keepalive started, in a 200 body."""
    if streamed:
        envelope = {"_streamed_error": True, "status_code": status_code}
        return _json(
            200, envelope | {"detail": message}, keepalive=b"   ", headers=headers
        )
    return _json(
        status_code,
        {"message": message, "error_code": error_code, "trace_id": TRACE_ID},
        headers=headers,
    )


def _prediction(*, with_id: bool = True) -> dict[str, Any]:
    body: dict[str, Any] = {
        "prediction": PREDICTION,
        "metadata": METADATA,
        "timings": TIMINGS,
    }
    if with_id:
        body["fitted_test_set_id"] = str(PREDICTION_ID)
    return body


def _predict_pending(retry_in_secs: float = 5.0) -> dict[str, Any]:
    """`/predict` once the server stopped waiting for the prediction."""
    return {
        "fitted_test_set_id": str(PREDICTION_ID),
        "status": "pending",
        "retry_in_secs": retry_in_secs,
    }


def _pending(retry_in_secs: float | None = 5.0) -> dict[str, Any]:
    """A poll of a running prediction, every field present as the server sends it."""
    return {
        "fitted_test_set_id": str(PREDICTION_ID),
        "status": "pending",
        "retry_in_secs": retry_in_secs,
        "error": None,
        "error_code": None,
        "prediction": None,
        "metadata": None,
        "timings": None,
    }


def _completed(**prediction: Any) -> dict[str, Any]:
    return {
        **_prediction(),
        **prediction,
        "status": "completed",
        "retry_in_secs": None,
        "error": None,
        "error_code": None,
    }


def _failed(error: str | None, error_code: str | None) -> dict[str, Any]:
    return {
        **_pending(),
        "status": "failed",
        "retry_in_secs": None,
        "error": error,
        "error_code": error_code,
    }


# -- Client state. --


def _settings_payload(*, thresholds: bool = True) -> dict[str, Any]:
    """`get_settings`, from a server with predict jobs or from an older one."""
    payload = _api_settings_payload()
    async_settings = payload["async_settings"]
    if thresholds:
        async_settings |= {
            "predict_use_above": THRESHOLDS,
            "thinking_predict_use_above": THINKING_THRESHOLDS,
        }
    else:
        del async_settings["predict_use_above"]
        del async_settings["thinking_predict_use_above"]
    return payload


def _server_settings() -> GetSettingsResponse:
    return GetSettingsResponse.model_validate(_settings_payload())


def _cache_settings() -> None:
    """Fill the settings cache, as `init()` does."""
    ServiceClient._api_settings = _server_settings()
    ServiceClient._api_settings_ts = time.monotonic()


def _async_settings(*, poll_timeout: float = 7200.0) -> ResolvedAsyncSettings:
    return ResolvedAsyncSettings(
        use_above_trainset_size_bytes=50 * 1024 * 1024,
        poll_timeout_secs=poll_timeout,
    )


@pytest.fixture(autouse=True)
def authorized_client() -> Iterator[None]:
    ServiceClient.reset_authorization()
    ServiceClient.authorize("dummy_token")
    try:
        yield
    finally:
        ServiceClient.reset_authorization()
        ServiceClient._api_settings = None
        ServiceClient._api_settings_ts = 0.0


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with MockTabPFNServer() as server:
        assert server.router is not None
        yield server.router


def _upload_route(router: respx.MockRouter) -> respx.Route:
    """Answer the test set upload as a duplicate, so nothing is uploaded."""
    return router.post("/tabpfn/prepare_test_set_upload").respond(
        409,
        json={
            "message": "duplicate",
            "error_code": "DUPLICATE_TEST_SET_UPLOAD",
            "test_set_upload_id": UPLOAD_ID,
        },
    )


class _TickingTime(_FakeTime):
    """A fake clock that moves on between any two reads, as a real one does,
    and a sleep that refuses a negative length, as `time.sleep` does."""

    def monotonic(self) -> float:
        self.now += 0.001
        return self.now

    def sleep(self, secs: float) -> None:
        if secs < 0:
            raise ValueError("sleep length must be non-negative")
        super().sleep(secs)


@contextmanager
def _fake_clock(fake_time: _FakeTime | None = None) -> Generator[_FakeTime]:
    if fake_time is None:
        fake_time = _FakeTime()
    with patch("tabpfn_client.client.time", fake_time):
        yield fake_time


@contextmanager
def _settings_cached_minutes_after_boot() -> Generator[_FakeTime]:
    """Cache the settings a minute ago, on a fake clock two minutes after boot.

    `time.monotonic()` counts from boot, so a cache expired by setting its
    timestamp to 0.0 would stay fresh here, as on a machine in its first hour
    of uptime.
    """
    _cache_settings()
    ServiceClient._api_settings_ts = 60.0
    with _fake_clock() as clock:
        clock.now = 120.0
        yield clock


@contextmanager
def _polling(
    router: respx.MockRouter,
    outcomes: list[Any],
    *,
    poll_timeout: float = 7200.0,
    fake_time: _FakeTime | None = None,
) -> Generator[tuple[_FakeTime, respx.Route]]:
    """Answer the polls with `outcomes` in turn, on a fake clock."""
    route = router.post("/tabpfn/get_predict_result")
    route.side_effect = outcomes
    with (
        _fake_clock(fake_time) as clock,
        patch.object(
            ServiceClient,
            "_resolve_async_settings",
            return_value=_async_settings(poll_timeout=poll_timeout),
        ),
    ):
        yield clock, route


def _predict(**kwargs: Any) -> PredictionResult:
    return ServiceClient.predict(FIT_ID, X_TEST, ClassifierConfig(), **kwargs)


def _wait(**kwargs: Any) -> PredictionResult:
    return ServiceClient.wait_for_predict(PREDICTION_ID, **kwargs)


def _sent(route: respx.Route) -> dict[str, Any]:
    return _sent_in(route.calls.last)


def _sent_in(call: Any) -> dict[str, Any]:
    return json.loads(call.request.content)


def _broke_off_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and "Stopped waiting" in r.getMessage()
    ]


class TestAsyncSettings:
    def test_settings_from_an_older_server_do_not_validate(self):
        with pytest.raises(ValidationError):
            GetSettingsResponse.model_validate(_settings_payload(thresholds=False))

    def test_an_older_servers_settings_are_unavailable(self, router):
        router.get("/tabpfn/get_settings").respond(
            200, json=_settings_payload(thresholds=False)
        )

        assert ServiceClient.get_settings() is None

    def test_settings_carry_the_predict_thresholds(self):
        settings = GetSettingsResponse.model_validate(_settings_payload())

        assert settings.async_settings.predict_use_above == PredictAsyncThresholds(
            train_rows=50_000, test_rows=100_000
        )
        assert (
            settings.async_settings.thinking_predict_use_above
            == PredictAsyncThresholds(train_rows=25_000, test_rows=10_000)
        )

    def test_dropped_settings_are_not_returned_when_a_refetch_fails(self, router):
        _cache_settings()
        settings_route = router.get("/tabpfn/get_settings").respond(503)

        ServiceClient._drop_settings()

        assert ServiceClient.get_settings() is None
        assert settings_route.call_count == 1


def _resolve(
    api_mode: ApiMode | str,
    *,
    server: bool = True,
    n_train_rows: int | None = None,
    n_test_rows: int = 1,
    thinking: bool = False,
    fit_with_cache: bool = False,
) -> ApiMode:
    """Resolve the mode with the settings of a server with predict jobs, or with
    none, as when the server is unreachable or predates predict jobs."""
    settings = _server_settings() if server else None
    with patch.object(ServiceClient, "get_settings", return_value=settings):
        return ServiceClient._resolve_predict_mode(
            cast(Any, api_mode),
            n_train_rows=n_train_rows,
            n_test_rows=n_test_rows,
            thinking=thinking,
            fit_with_cache=fit_with_cache,
        )


class TestResolvePredictMode:
    def test_auto_predicts_with_one_request_without_settings(self, caplog):
        with caplog.at_level(logging.DEBUG, logger="tabpfn_client.client"):
            mode = _resolve(
                ApiMode.AUTO, server=False, n_train_rows=10**6, n_test_rows=10**6
            )

        assert mode is ApiMode.SYNC
        assert "No server settings; predicting with one request." in caplog.text

    @pytest.mark.parametrize("api_mode", [ApiMode.SYNC, ApiMode.ASYNC])
    def test_sync_and_async_are_kept_without_settings(self, api_mode):
        # ASYNC tries the job endpoint, and `predict()` falls back to one
        # request if the server has none.
        mode = _resolve(api_mode, server=False, n_train_rows=1, n_test_rows=1)

        assert mode is api_mode

    @pytest.mark.parametrize(
        ("api_mode", "rows"), [(ApiMode.SYNC, 10**6), (ApiMode.ASYNC, 1)]
    )
    @pytest.mark.parametrize("fit_with_cache", [False, True])
    def test_sync_and_async_are_kept_whatever_the_size(
        self, api_mode, rows, fit_with_cache
    ):
        mode = _resolve(
            api_mode,
            n_train_rows=rows,
            n_test_rows=rows,
            thinking=True,
            fit_with_cache=fit_with_cache,
        )

        assert mode is api_mode

    @pytest.mark.parametrize(
        ("n_train_rows", "n_test_rows", "expected"),
        [
            (49_999, 99_999, ApiMode.SYNC),
            (50_000, 1, ApiMode.ASYNC),
            (50_001, 1, ApiMode.ASYNC),
            (1, 100_000, ApiMode.ASYNC),
            (None, 99_999, ApiMode.SYNC),
            (None, 100_000, ApiMode.ASYNC),
        ],
        ids=[
            "both_below",
            "train_rows_reached",
            "train_rows_above",
            "test_rows_reached",
            "unknown_train_rows_test_rows_below",
            "unknown_train_rows_test_rows_reached",
        ],
    )
    def test_auto_runs_a_job_once_either_count_reaches_its_threshold(
        self, n_train_rows, n_test_rows, expected
    ):
        mode = _resolve(
            ApiMode.AUTO,
            n_train_rows=n_train_rows,
            n_test_rows=n_test_rows,
        )

        assert mode is expected

    def test_auto_predicts_a_fit_with_a_kv_cache_with_one_request(self):
        mode = _resolve(
            ApiMode.AUTO,
            n_train_rows=10**6,
            n_test_rows=10**6,
            fit_with_cache=True,
        )

        assert mode is ApiMode.SYNC

    @pytest.mark.parametrize(
        ("n_train_rows", "n_test_rows"), [(25_000, 1), (None, 10_000)]
    )
    def test_auto_uses_the_thinking_thresholds_for_a_thinking_fit(
        self, n_train_rows, n_test_rows
    ):
        rows = {"n_train_rows": n_train_rows, "n_test_rows": n_test_rows}

        thinking = _resolve(ApiMode.AUTO, thinking=True, **rows)
        standard = _resolve(ApiMode.AUTO, thinking=False, **rows)

        assert thinking is ApiMode.ASYNC
        assert standard is ApiMode.SYNC

    @pytest.mark.parametrize(
        ("api_mode", "n_train_rows", "expected"),
        [
            ("sync", 50_000, ApiMode.SYNC),
            ("async", 1, ApiMode.ASYNC),
            ("auto", 50_000, ApiMode.ASYNC),
            ("auto", 1, ApiMode.SYNC),
        ],
    )
    def test_takes_the_mode_as_the_plain_string_load_model_restores(
        self, api_mode, n_train_rows, expected
    ):
        mode = _resolve(api_mode, n_train_rows=n_train_rows)

        assert mode is expected


class TestPredict:
    """`ServiceClient.predict` with one request, as SYNC and AUTO below the
    thresholds do, and what the server can answer it."""

    @pytest.fixture(autouse=True)
    def server_with_predict_jobs(self) -> None:
        _cache_settings()

    @pytest.mark.parametrize("with_id", [True, False], ids=["with_id", "older"])
    def test_returns_the_prediction_and_its_id(self, router, with_id):
        _upload_route(router)
        predict_route = router.post("/tabpfn/predict").respond(
            200, json=_prediction(with_id=with_id)
        )

        result = _predict()

        np.testing.assert_array_equal(result.y_pred, PREDICTION)
        assert result.metadata["task"] == "classification"
        assert result.timings == TIMINGS
        assert result.fitted_test_set_id == (PREDICTION_ID if with_id else None)
        assert _sent(predict_route) == {
            "test_set_upload_id": UPLOAD_ID,
            "fitted_train_set_id": str(FIT_ID),
            "task_config": {"task": "classification"},
        }

    def test_skips_the_keepalive_pings_before_the_prediction(self, router):
        _upload_route(router)
        router.post("/tabpfn/predict").mock(
            return_value=_json(200, _prediction(), keepalive=b"  \n  ")
        )

        result = _predict()

        np.testing.assert_array_equal(result.y_pred, PREDICTION)
        assert result.fitted_test_set_id == PREDICTION_ID

    def test_polls_for_a_prediction_the_server_stopped_waiting_for(
        self, router, caplog
    ):
        _upload_route(router)
        predict_route = router.post("/tabpfn/predict").mock(
            return_value=_json(200, _predict_pending(4.0), keepalive=b"   ")
        )
        outcomes = [_json(200, _pending(2.0)), _json(200, _completed())]

        with (
            _polling(router, outcomes) as (clock, poll_route),
            caplog.at_level(logging.INFO, logger="tabpfn_client.client"),
        ):
            result = _predict()

        # The first poll waits for the hint of the pending answer, the next
        # one for the hint of the first poll.
        assert clock.sleeps == [4.0, 2.0]
        assert predict_route.call_count == 1
        assert [json.loads(call.request.content) for call in poll_route.calls] == [
            {"fitted_test_set_id": str(PREDICTION_ID)}
        ] * 2
        np.testing.assert_array_equal(result.y_pred, PREDICTION)
        assert result.metadata["task"] == "classification"
        assert result.timings == TIMINGS
        assert result.fitted_test_set_id == PREDICTION_ID
        assert (
            f"Prediction {PREDICTION_ID} is still running on the server; polling "
            "for its result."
        ) in caplog.messages

    def test_the_wait_before_the_first_poll_counts_towards_the_poll_timeout(
        self, router
    ):
        _upload_route(router)
        router.post("/tabpfn/predict").respond(200, json=_predict_pending(4.0))

        # Timeout 6: the poll at t=4 leaves 2 s, too few for the next one.
        with _polling(
            router,
            [_json(200, _pending(5.0)), _json(200, _completed())],
            poll_timeout=6.0,
        ) as (_, poll_route):
            with pytest.raises(TimeoutError):
                _predict()

        assert poll_route.call_count == 1

    def test_the_first_poll_goes_out_by_the_deadline(self, router):
        _upload_route(router)
        router.post("/tabpfn/predict").respond(200, json=_predict_pending(5.0))

        with _polling(router, [_json(200, _completed())], poll_timeout=3.0) as (
            clock,
            poll_route,
        ):
            result = _predict()

        # Not after the 5 s hint, which is past the deadline; and a poll timeout
        # shorter than the hint still gets one answer.
        assert clock.sleeps == [3.0]
        assert poll_route.call_count == 1
        assert result.fitted_test_set_id == PREDICTION_ID

    def test_the_first_poll_goes_out_at_once_when_the_deadline_has_passed(self, router):
        _upload_route(router)
        router.post("/tabpfn/predict").respond(200, json=_predict_pending(5.0))

        # Timeout 0: a real clock has moved past the deadline by the time the
        # first poll's delay is clamped to it.
        with _polling(
            router,
            [_json(200, _completed())],
            poll_timeout=0.0,
            fake_time=_TickingTime(),
        ) as (clock, poll_route):
            result = _predict()

        assert sum(clock.sleeps) == 0.0
        assert poll_route.call_count == 1
        assert result.fitted_test_set_id == PREDICTION_ID

    def test_polls_a_pending_predict_with_the_request_timeout_and_headers(self, router):
        _upload_route(router)
        router.post("/tabpfn/predict").respond(200, json=_predict_pending(1.0))
        options = ClientOptions(timeout=42.0, headers={"sentry-trace": "abc"})

        with _polling(router, [_json(200, _completed())]) as (_, poll_route):
            _predict(client_options=options)

        request = poll_route.calls.last.request
        assert request.headers["sentry-trace"] == "abc"
        assert request.extensions["timeout"]["read"] == 42.0

    def test_a_pending_predict_caps_the_timeout_of_its_polls_alone(self, router):
        _upload_route(router)
        predict_route = router.post("/tabpfn/predict").respond(
            200, json=_predict_pending(1.0)
        )

        with _polling(router, [_json(200, _completed())]) as (_, poll_route):
            _predict(client_options=ClientOptions(timeout=900.0))

        # Only a poll is safe to cut short: the next poll gets its answer again.
        assert predict_route.calls.last.request.extensions["timeout"]["read"] == 900.0
        assert poll_route.calls.last.request.extensions["timeout"]["read"] == 120.0

    @pytest.mark.parametrize(
        "status", [None, "queued"], ids=["no_status", "unknown_status"]
    )
    def test_polls_a_pending_answer_without_a_status_this_client_knows(
        self, router, status
    ):
        _upload_route(router)
        pending = _predict_pending(1.0)
        if status is None:
            del pending["status"]
        else:
            pending["status"] = status
        router.post("/tabpfn/predict").respond(200, json=pending)

        with _polling(router, [_json(200, _completed())]) as (_, poll_route):
            result = _predict()

        assert poll_route.call_count == 1
        assert result.fitted_test_set_id == PREDICTION_ID

    @pytest.mark.parametrize(
        ("error", "error_code", "expected", "message"),
        [
            (
                "boom",
                "INTERNAL_ERROR",
                PredictFailedError,
                f"Prediction {PREDICTION_ID} failed: boom (error code INTERNAL_ERROR)",
            ),
            (
                None,
                None,
                PredictFailedError,
                f"Prediction {PREDICTION_ID} failed: no error message provided",
            ),
            (
                "X_test has 3 features, expected 4",
                "USER_ERROR",
                PredictFailedError,
                f"Prediction {PREDICTION_ID} failed: X_test has 3 features, "
                "expected 4 (error code USER_ERROR)",
            ),
            (
                NEEDS_REFITTING,
                "NEEDS_REFITTING",
                RefitRequiredError,
                f"Prediction {PREDICTION_ID} failed: {NEEDS_REFITTING} "
                "(error code NEEDS_REFITTING)",
            ),
            (
                None,
                "NEEDS_REFITTING",
                RefitRequiredError,
                f"Prediction {PREDICTION_ID} failed: no error message provided "
                "(error code NEEDS_REFITTING)",
            ),
            (
                STALE_KV_CACHE,
                "KV_CACHE_INVALID",
                RefitRequiredError,
                f"Prediction {PREDICTION_ID} failed: {STALE_KV_CACHE} "
                "(error code KV_CACHE_INVALID)",
            ),
            (
                MISSING_KV_CACHE,
                "KV_CACHE_INVALID",
                RefitRequiredError,
                f"Prediction {PREDICTION_ID} failed: {MISSING_KV_CACHE} "
                "(error code KV_CACHE_INVALID)",
            ),
            (
                None,
                "KV_CACHE_INVALID",
                RefitRequiredError,
                f"Prediction {PREDICTION_ID} failed: no error message provided "
                "(error code KV_CACHE_INVALID)",
            ),
            # A server without KV_CACHE_INVALID books a stale or missing cache
            # as a USER_ERROR: only the message tells.
            (
                STALE_KV_CACHE,
                "USER_ERROR",
                RefitRequiredError,
                f"Prediction {PREDICTION_ID} failed: {STALE_KV_CACHE} "
                "(error code USER_ERROR)",
            ),
            (
                MISSING_KV_CACHE,
                "USER_ERROR",
                RefitRequiredError,
                f"Prediction {PREDICTION_ID} failed: {MISSING_KV_CACHE} "
                "(error code USER_ERROR)",
            ),
            # The message counts only with a code a server without the refit
            # codes books for a refit.
            (
                STALE_KV_CACHE,
                "INTERNAL_ERROR",
                PredictFailedError,
                f"Prediction {PREDICTION_ID} failed: {STALE_KV_CACHE} "
                "(error code INTERNAL_ERROR)",
            ),
        ],
        ids=[
            "failed",
            "failed_without_a_message",
            "user_error",
            "needs_refitting",
            "needs_refitting_without_a_message",
            "stale_kv_cache",
            "missing_kv_cache",
            "kv_cache_invalid_without_a_message",
            "stale_kv_cache_before_its_code",
            "missing_kv_cache_before_its_code",
            "refit_text_under_another_code",
        ],
    )
    def test_a_polled_prediction_that_fails_raises_what_the_server_reports(
        self, router, caplog, error, error_code, expected, message
    ):
        _upload_route(router)
        router.post("/tabpfn/predict").respond(200, json=_predict_pending())

        with (
            _polling(
                router, [_json(200, _pending()), _json(200, _failed(error, error_code))]
            ),
            pytest.raises(PredictFailedError) as exc_info,
        ):
            _predict()

        assert type(exc_info.value) is expected
        assert str(exc_info.value) == message
        assert exc_info.value.fitted_test_set_id == PREDICTION_ID
        # The prediction has ended: there is nothing left to come back for.
        assert _broke_off_warnings(caplog) == []

    @pytest.mark.parametrize(
        ("status_code", "message", "error_code", "streamed"),
        [
            (409, NEEDS_REFITTING, "NEEDS_REFITTING", False),
            (409, NEEDS_REFITTING, "", True),
            (409, STALE_KV_CACHE, "USER_ERROR", False),
            (409, MISSING_KV_CACHE, "", True),
            (400, STALE_KV_CACHE, "USER_ERROR", False),
            (400, MISSING_KV_CACHE, "", True),
        ],
        ids=[
            "needs_refitting",
            "needs_refitting_streamed",
            "kv_cache",
            "kv_cache_streamed",
            "booked_kv_cache",
            "booked_kv_cache_streamed",
        ],
    )
    def test_a_model_that_can_no_longer_predict_raises_refit_required_error(
        self, router, status_code, message, error_code, streamed
    ):
        # The bodies of a server that predates `PredictFailedErrorResponse`
        # and KV_CACHE_INVALID: the status and the message tell.
        _upload_route(router)
        predict_route = router.post("/tabpfn/predict").mock(
            return_value=_error(
                status_code, message, streamed=streamed, error_code=error_code
            )
        )

        with pytest.raises(RefitRequiredError) as exc_info:
            _predict()

        # Only a fast error carries a trace id.
        assert str(exc_info.value) == (
            message if streamed else f"{message} Report trace ID: {TRACE_ID}."
        )
        # Such a server does not name the prediction.
        assert exc_info.value.fitted_test_set_id is None
        assert predict_route.call_count == 1

    @pytest.mark.parametrize(
        ("status_code", "message", "error_code"),
        [
            (409, NEEDS_REFITTING, "NEEDS_REFITTING"),
            (400, STALE_KV_CACHE, "USER_ERROR"),
        ],
        ids=["needs_refitting", "booked_kv_cache"],
    )
    def test_warns_of_a_deprecated_client_on_a_refit_answer(
        self, router, status_code, message, error_code
    ):
        _upload_route(router)
        router.post("/tabpfn/predict").respond(
            status_code,
            json={"message": message, "error_code": error_code},
            headers={"deprecation": "true"},
        )

        with (
            pytest.warns(DeprecationWarning, match="tabpfn-client is deprecated"),
            pytest.raises(RefitRequiredError),
        ):
            _predict()

    @pytest.mark.parametrize(
        ("status_code", "message", "streamed", "expected"),
        [
            (
                400,
                "Bad input",
                False,
                "Fail to call predict: [HTTP 400] Bad input. Report trace ID: "
                f"{TRACE_ID}.",
            ),
            (
                422,
                "Fitted train set is not ready",
                False,
                "Fail to call predict: [HTTP 422] Fitted train set is not ready. "
                f"Report trace ID: {TRACE_ID}.",
            ),
            (
                400,
                "Bad input",
                True,
                "Fail to call predict with error: streamed, Bad input",
            ),
            (
                500,
                "Internal server error.",
                True,
                "Fail to call predict with error: streamed, Internal server error.",
            ),
        ],
        ids=["fast_400", "fast_422", "streamed_400", "streamed_500"],
    )
    def test_other_errors_are_raised_as_before(
        self, router, status_code, message, streamed, expected
    ):
        _upload_route(router)
        predict_route = router.post("/tabpfn/predict").mock(
            return_value=_error(status_code, message, streamed=streamed)
        )

        with pytest.raises(RuntimeError) as exc_info:
            _predict()

        assert not isinstance(exc_info.value, PredictFailedError)
        assert str(exc_info.value) == expected
        assert predict_route.call_count == 1

    @pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout])
    def test_a_predict_that_never_reached_the_server_is_retried_once(
        self, router, error
    ):
        _upload_route(router)
        predict_route = router.post("/tabpfn/predict")
        predict_route.side_effect = [error("unreachable"), _json(200, _prediction())]

        result = _predict()

        assert predict_route.call_count == 2
        assert result.fitted_test_set_id == PREDICTION_ID

    @pytest.mark.parametrize(
        "outcome",
        [
            httpx.ReadTimeout("lost"),
            httpx.RemoteProtocolError("cut off"),
            _error(503, "Overloaded"),
        ],
        ids=["read_timeout", "cut_off", "503"],
    )
    def test_a_predict_the_server_may_have_run_is_not_retried(self, router, outcome):
        # Each /predict creates a job; even a 503 can come after it completed.
        _upload_route(router)
        predict_route = router.post("/tabpfn/predict")
        predict_route.side_effect = [outcome]

        with pytest.raises((httpx.HTTPError, RetryableServerError)) as exc_info:
            _predict()

        assert predict_route.call_count == 1
        if isinstance(outcome, httpx.Response):
            # `except RuntimeError` catches every failure the server reports.
            assert isinstance(exc_info.value, RuntimeError)

    def test_rejects_an_invalid_mode_before_any_request(self, router):
        with pytest.raises(ValueError):
            _predict(api_mode=cast(Any, "SYNC"))


def _job_failure(
    status_code: int,
    message: str,
    error_code: str,
    *,
    streamed: bool = False,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """The failure of the job a `/predict` or a submit created, which names it."""
    body = {
        "message": message,
        "error_code": error_code,
        "trace_id": TRACE_ID,
        "fitted_test_set_id": str(PREDICTION_ID),
    }
    if streamed:
        # As the server streams it: the body's fields, then the envelope's own
        # keys, with `detail` the message that older clients read.
        envelope = {"_streamed_error": True, "status_code": status_code}
        return _json(
            200,
            body | envelope | {"detail": message},
            keepalive=b"   ",
            headers=headers,
        )
    return _json(status_code, body, headers=headers)


STREAMED = pytest.mark.parametrize("streamed", [False, True], ids=["fast", "streamed"])


class TestAFailureReportedOnTheRequest:
    """A job that fails before the request that created it is answered raises
    what a poll for the job raises, with the job's id."""

    @STREAMED
    @pytest.mark.parametrize(
        ("status_code", "message", "error_code", "expected"),
        [
            (500, "boom", "INTERNAL_ERROR", PredictFailedError),
            (503, "Overloaded", "BUSY", PredictFailedError),
            (504, "Too slow", "DEADLINE_EXCEEDED", PredictFailedError),
            (
                422,
                "X_test has 3 features, expected 4",
                "REQUEST_VALIDATION_ERROR",
                PredictFailedError,
            ),
            (409, NEEDS_REFITTING, "NEEDS_REFITTING", RefitRequiredError),
            (409, STALE_KV_CACHE, "KV_CACHE_INVALID", RefitRequiredError),
            (409, MISSING_KV_CACHE, "KV_CACHE_INVALID", RefitRequiredError),
            # The code decides, not the status.
            (409, "Conflict", "USER_ERROR", PredictFailedError),
            # A server whose worker predates KV_CACHE_INVALID, as it answers
            # and as it books the failure: only the message tells.
            (409, STALE_KV_CACHE, "USER_ERROR", RefitRequiredError),
            (400, MISSING_KV_CACHE, "USER_ERROR", RefitRequiredError),
        ],
        ids=[
            "internal_error",
            "busy",
            "deadline_exceeded",
            "validation_error",
            "needs_refitting",
            "stale_kv_cache",
            "missing_kv_cache",
            "conflict_without_a_refit_code",
            "kv_cache_from_an_older_worker",
            "booked_kv_cache_from_an_older_worker",
        ],
    )
    def test_a_failed_predict_raises_what_a_poll_raises(
        self, router, caplog, status_code, message, error_code, expected, streamed
    ):
        _upload_route(router)
        # No poll route: a poll would fail the test.
        predict_route = router.post("/tabpfn/predict").mock(
            return_value=_job_failure(
                status_code, message, error_code, streamed=streamed
            )
        )

        with pytest.raises(PredictFailedError) as exc_info:
            _predict()

        assert type(exc_info.value) is expected
        assert str(exc_info.value) == (
            f"Prediction {PREDICTION_ID} failed: {message} (error code "
            f"{error_code}) Report trace ID: {TRACE_ID}."
        )
        assert exc_info.value.fitted_test_set_id == PREDICTION_ID
        assert predict_route.call_count == 1
        # The prediction has ended: there is nothing left to come back for.
        assert _broke_off_warnings(caplog) == []

    @STREAMED
    def test_a_body_without_the_id_is_recognized_as_before(self, router, streamed):
        # A server with KV_CACHE_INVALID that does not name the prediction.
        _upload_route(router)
        if streamed:
            envelope = {"_streamed_error": True, "status_code": 409}
            body = {"detail": STALE_KV_CACHE, "error_code": "KV_CACHE_INVALID"}
            response = _json(200, envelope | body, keepalive=b"   ")
        else:
            response = _error(409, STALE_KV_CACHE, error_code="KV_CACHE_INVALID")
        router.post("/tabpfn/predict").mock(return_value=response)

        with pytest.raises(RefitRequiredError) as exc_info:
            _predict()

        assert exc_info.value.fitted_test_set_id is None

    def test_warns_of_a_deprecated_client_on_a_failed_job(self, router):
        _upload_route(router)
        router.post("/tabpfn/predict").mock(
            return_value=_job_failure(
                503, "Overloaded", "BUSY", headers={"deprecation": "true"}
            )
        )

        with (
            pytest.warns(DeprecationWarning, match="tabpfn-client is deprecated"),
            pytest.raises(PredictFailedError),
        ):
            _predict()

    @pytest.mark.parametrize(
        ("status_code", "message", "error_code"),
        [
            (503, "The service is overloaded.", "BUSY"),
            (413, "The prediction request is too large.", "INVALID_REQUEST"),
        ],
        ids=["busy", "too_large"],
    )
    @pytest.mark.parametrize("call", ["predict", "submit_predict"])
    def test_a_submitted_job_that_failed_at_once_raises_what_a_poll_raises(
        self, router, caplog, status_code, message, error_code, call
    ):
        # The server created the job and could not queue it.
        _cache_settings()
        _upload_route(router)
        # No /predict and no poll route: either would fail the test.
        submit_route = router.post("/tabpfn/submit_predict_job").mock(
            return_value=_job_failure(status_code, message, error_code)
        )

        with pytest.raises(PredictFailedError) as exc_info:
            if call == "predict":
                _predict(api_mode=ApiMode.ASYNC)
            else:
                ServiceClient.submit_predict(FIT_ID, X_TEST, ClassifierConfig())

        assert type(exc_info.value) is PredictFailedError
        assert str(exc_info.value) == (
            f"Prediction {PREDICTION_ID} failed: {message} (error code "
            f"{error_code}) Report trace ID: {TRACE_ID}."
        )
        assert exc_info.value.fitted_test_set_id == PREDICTION_ID
        assert submit_route.call_count == 1
        assert _broke_off_warnings(caplog) == []

    def test_a_submit_answered_without_the_id_raises_as_before(self, router):
        # A server that predates the id, or failed before it created the job.
        _cache_settings()
        _upload_route(router)
        submit_route = router.post("/tabpfn/submit_predict_job").mock(
            return_value=_error(503, "Overloaded", error_code="INTERNAL_ERROR")
        )

        with pytest.raises(RetryableServerError) as exc_info:
            ServiceClient.submit_predict(FIT_ID, X_TEST, ClassifierConfig())

        assert str(exc_info.value) == (
            "Fail to call submit_predict_job: [HTTP 503] Overloaded. "
            f"Report trace ID: {TRACE_ID}."
        )
        assert submit_route.call_count == 1


class TestPredictionJobs:
    """`predict()` as a background job, `submit_predict()`, and servers that run
    no predictions as background jobs."""

    def test_async_submits_the_prediction_and_polls_for_it(self, router):
        _cache_settings()
        _upload_route(router)
        submit_route = router.post("/tabpfn/submit_predict_job").respond(
            200, json={"fitted_test_set_id": str(PREDICTION_ID)}
        )

        with _polling(
            router, [_json(200, _pending(0.5)), _json(200, _completed())]
        ) as (clock, poll_route):
            result = _predict(api_mode=ApiMode.ASYNC)

        assert _sent(submit_route) == {
            "test_set_upload_id": UPLOAD_ID,
            "fitted_train_set_id": str(FIT_ID),
            "task_config": {"task": "classification"},
        }
        assert submit_route.call_count == 1
        # The first poll goes out at once.
        assert clock.sleeps == [0.5]
        assert poll_route.call_count == 2
        np.testing.assert_array_equal(result.y_pred, PREDICTION)
        assert result.timings == TIMINGS
        assert result.fitted_test_set_id == PREDICTION_ID

    def test_a_job_is_submitted_and_polled_with_the_request_timeout_and_headers(
        self, router
    ):
        _cache_settings()
        _upload_route(router)
        submit_route = router.post("/tabpfn/submit_predict_job").respond(
            200, json={"fitted_test_set_id": str(PREDICTION_ID)}
        )
        options = ClientOptions(timeout=42.0, headers={"sentry-trace": "abc"})

        with _polling(router, [_json(200, _completed())]) as (_, poll_route):
            _predict(api_mode=ApiMode.ASYNC, client_options=options)

        for route in (submit_route, poll_route):
            request = route.calls.last.request
            assert request.headers["sentry-trace"] == "abc"
            assert request.extensions["timeout"]["read"] == 42.0

    def test_async_caps_the_timeout_of_the_polls_alone(self, router):
        _cache_settings()
        _upload_route(router)
        submit_route = router.post("/tabpfn/submit_predict_job").respond(
            200, json={"fitted_test_set_id": str(PREDICTION_ID)}
        )
        options = ClientOptions(timeout=900.0)

        with _polling(
            router, [_json(200, _pending(1.0)), _json(200, _completed())]
        ) as (_, poll_route):
            _predict(api_mode=ApiMode.ASYNC, client_options=options)

        # Only a poll is safe to cut short: the next poll gets its answer again.
        assert submit_route.calls.last.request.extensions["timeout"]["read"] == 900.0
        assert [
            call.request.extensions["timeout"]["read"] for call in poll_route.calls
        ] == [120.0] * 2

    def test_auto_submits_a_prediction_that_reaches_a_threshold(self, router):
        _cache_settings()
        _upload_route(router)
        router.post("/tabpfn/submit_predict_job").respond(
            200, json={"fitted_test_set_id": str(PREDICTION_ID)}
        )

        with _polling(router, [_json(200, _completed())]):
            result = _predict(api_mode=ApiMode.AUTO, n_train_rows=50_000)

        assert result.fitted_test_set_id == PREDICTION_ID

    def test_auto_predicts_a_prediction_below_the_thresholds_with_one_request(
        self, router
    ):
        _cache_settings()
        _upload_route(router)
        router.post("/tabpfn/predict").respond(200, json=_prediction())

        result = _predict(api_mode=ApiMode.AUTO, n_train_rows=49_999)

        assert result.fitted_test_set_id == PREDICTION_ID

    @pytest.mark.parametrize(
        ("fit_mode", "fit_with_cache"),
        [
            (None, False),
            (FitMode.FIT_PREPROCESSORS, False),
            (FitMode.FIT_WITH_CACHE, True),
        ],
    )
    def test_decides_on_the_fit_and_the_rows_to_predict(
        self, router, fit_mode, fit_with_cache
    ):
        _cache_settings()
        _upload_route(router)
        router.post("/tabpfn/predict").respond(200, json=_prediction())
        task_config = ClassifierConfig(
            tabpfn_config=ClassifierTabPFNConfig(fit_mode=fit_mode)
        )

        with patch.object(
            ServiceClient,
            "_resolve_predict_mode",
            wraps=ServiceClient._resolve_predict_mode,
        ) as resolve:
            ServiceClient.predict(
                FIT_ID,
                X_TEST,
                task_config,
                api_mode=ApiMode.AUTO,
                n_train_rows=7,
                thinking=True,
            )

        resolve.assert_called_once_with(
            ApiMode.AUTO,
            n_train_rows=7,
            n_test_rows=3,
            thinking=True,
            fit_with_cache=fit_with_cache,
        )

    @pytest.mark.parametrize("error", [httpx.ConnectError, httpx.ConnectTimeout])
    def test_a_submit_that_never_reached_the_server_is_retried_once(
        self, router, error
    ):
        _cache_settings()
        _upload_route(router)
        submit_route = router.post("/tabpfn/submit_predict_job")
        submit_route.side_effect = [
            error("unreachable"),
            httpx.Response(200, json={"fitted_test_set_id": str(PREDICTION_ID)}),
        ]

        fitted_test_set_id = ServiceClient.submit_predict(
            FIT_ID, X_TEST, ClassifierConfig()
        )

        assert fitted_test_set_id == PREDICTION_ID
        assert submit_route.call_count == 2

    @pytest.mark.parametrize("error", [httpx.ReadTimeout, httpx.RemoteProtocolError])
    def test_a_submit_the_server_may_have_accepted_is_not_retried(self, router, error):
        _cache_settings()
        _upload_route(router)
        submit_route = router.post("/tabpfn/submit_predict_job")
        submit_route.side_effect = [error("lost")]

        with pytest.raises(error):
            _predict(api_mode=ApiMode.ASYNC)

        assert submit_route.call_count == 1

    @ROUTE_MISSING
    def test_async_tries_a_job_on_a_server_whose_settings_are_unavailable(
        self, router, route_missing
    ):
        # A server that predates predict jobs: its settings don't validate.
        settings_route = router.get("/tabpfn/get_settings").respond(
            200, json=_settings_payload(thresholds=False)
        )
        _upload_route(router)
        submit_route = router.post("/tabpfn/submit_predict_job").respond(
            404, json=route_missing
        )
        predict_route = router.post("/tabpfn/predict").respond(
            200, json=_prediction(with_id=False)
        )

        result = _predict(api_mode=ApiMode.ASYNC)

        assert settings_route.called
        assert submit_route.call_count == 1
        assert predict_route.call_count == 1
        np.testing.assert_array_equal(result.y_pred, PREDICTION)

    @ROUTE_MISSING
    def test_predicts_with_one_request_when_the_server_has_no_submit_endpoint(
        self, router, caplog, route_missing
    ):
        upload_route = _upload_route(router)
        submit_route = router.post("/tabpfn/submit_predict_job").respond(
            404, json=route_missing
        )
        predict_route = router.post("/tabpfn/predict").respond(
            200, json=_prediction(with_id=False)
        )
        settings_route = router.get("/tabpfn/get_settings").respond(
            200, json=_settings_payload(thresholds=False)
        )

        # Settings cached before the server was rolled back.
        with _settings_cached_minutes_after_boot():
            with caplog.at_level(logging.WARNING, logger="tabpfn_client.client"):
                result = _predict(api_mode=ApiMode.ASYNC)
            assert settings_route.call_count == 0
            settings = ServiceClient.get_settings()

        np.testing.assert_array_equal(result.y_pred, PREDICTION)
        assert caplog.messages == [
            "This TabPFN server does not run predictions as background jobs; "
            "predicting with one request."
        ]
        # The same upload, once.
        assert upload_route.call_count == 1
        assert submit_route.call_count == 1
        assert _sent(predict_route) == _sent(submit_route)
        assert _sent(predict_route)["test_set_upload_id"] == UPLOAD_ID
        # The settings that sent the prediction to the job endpoint are
        # dropped, and the older server's own don't validate.
        assert settings_route.call_count == 1
        assert settings is None

    @pytest.mark.parametrize(
        ("message", "error_code"),
        [
            ("Fitted train set not found", "NOT_FOUND"),
            ("x_test blob not found", "USER_ERROR"),
        ],
        ids=["unknown_fit", "missing_test_set"],
    )
    def test_the_submit_endpoints_own_404_is_raised(self, router, message, error_code):
        _cache_settings()
        _upload_route(router)
        # No /predict route: falling back to one request would fail the test.
        router.post("/tabpfn/submit_predict_job").mock(
            return_value=_error(404, message, error_code=error_code)
        )

        with pytest.raises(RuntimeError) as exc_info:
            _predict(api_mode=ApiMode.ASYNC)

        assert str(exc_info.value) == (
            f"Fail to call submit_predict_job: [HTTP 404] {message}. "
            f"Report trace ID: {TRACE_ID}."
        )

    def test_submit_predict_returns_the_id_of_the_job(self, router, caplog):
        _cache_settings()
        _upload_route(router)
        # No poll route: submit_predict() must not wait.
        submit_route = router.post("/tabpfn/submit_predict_job").respond(
            200, json={"fitted_test_set_id": str(PREDICTION_ID)}
        )
        options = ClientOptions(timeout=42.0, headers={"sentry-trace": "abc"})

        with caplog.at_level(logging.DEBUG, logger="tabpfn_client.client"):
            fitted_test_set_id = ServiceClient.submit_predict(
                FIT_ID, X_TEST, ClassifierConfig(), options
            )

        assert fitted_test_set_id == PREDICTION_ID
        assert _sent(submit_route) == {
            "test_set_upload_id": UPLOAD_ID,
            "fitted_train_set_id": str(FIT_ID),
            "task_config": {"task": "classification"},
        }
        request = submit_route.calls.last.request
        assert request.headers["sentry-trace"] == "abc"
        assert request.extensions["timeout"]["read"] == 42.0
        assert str(PREDICTION_ID) in caplog.text

    def test_submit_predict_keeps_a_request_timeout_above_the_poll_cap(self, router):
        _cache_settings()
        _upload_route(router)
        submit_route = router.post("/tabpfn/submit_predict_job").respond(
            200, json={"fitted_test_set_id": str(PREDICTION_ID)}
        )

        ServiceClient.submit_predict(
            FIT_ID, X_TEST, ClassifierConfig(), ClientOptions(timeout=900.0)
        )

        # The 2-minute cap is for prediction polls alone.
        assert submit_route.calls.last.request.extensions["timeout"]["read"] == 900.0

    def test_submit_predict_submits_when_the_settings_are_unreachable(self, router):
        _upload_route(router)
        router.post("/tabpfn/submit_predict_job").respond(
            200, json={"fitted_test_set_id": str(PREDICTION_ID)}
        )

        with patch.object(ServiceClient, "get_settings", return_value=None):
            fitted_test_set_id = ServiceClient.submit_predict(
                FIT_ID, X_TEST, ClassifierConfig()
            )

        assert fitted_test_set_id == PREDICTION_ID

    @ROUTE_MISSING
    def test_submit_predict_raises_when_the_server_has_no_submit_endpoint(
        self, router, route_missing
    ):
        upload_route = _upload_route(router)
        submit_route = router.post("/tabpfn/submit_predict_job").respond(
            404, json=route_missing
        )
        settings_route = router.get("/tabpfn/get_settings").respond(
            200, json=_settings_payload(thresholds=False)
        )

        # Settings cached before the server was rolled back.
        with _settings_cached_minutes_after_boot():
            with pytest.raises(RuntimeError) as first:
                ServiceClient.submit_predict(FIT_ID, X_TEST, ClassifierConfig())
            with pytest.raises(RuntimeError) as second:
                ServiceClient.submit_predict(FIT_ID, X_TEST, ClassifierConfig())

        assert str(first.value) == str(second.value) == NO_PREDICT_JOBS
        assert upload_route.call_count == submit_route.call_count == 2
        # The first call dropped the settings, and the older server's own
        # don't validate.
        assert ServiceClient.get_settings() is None
        assert settings_route.called


def _capped_errors() -> list[tuple[Callable[[], Any], type[Exception]]]:
    return [
        (lambda: _error(429, "Rate limit exceeded"), CappedRetryableServerError),
        (lambda: _error(500, "Internal error"), CappedRetryableServerError),
        (lambda: httpx.Response(200, content=b" \n "), CappedRetryableServerError),
        (
            lambda: _error(500, "Internal server error.", streamed=True),
            CappedRetryableServerError,
        ),
        (
            lambda: _error(429, "Rate limit exceeded", streamed=True),
            CappedRetryableServerError,
        ),
        (
            lambda: httpx.RemoteProtocolError("incomplete chunked read"),
            httpx.RemoteProtocolError,
        ),
    ]


CAPPED_ERRORS = pytest.mark.parametrize(
    ("make_error", "cause"),
    _capped_errors(),
    ids=["429", "500", "empty", "streamed_500", "streamed_429", "cut_off"],
)

TRANSIENT_ERRORS = pytest.mark.parametrize(
    ("make_error", "cause"),
    [
        *[
            (
                lambda status=status, streamed=streamed: _error(
                    status, "Try again", streamed=streamed
                ),
                RetryableServerError,
            )
            for streamed in (False, True)
            for status in (408, 502, 503, 504)
        ],
        (lambda: httpx.ConnectError("refused"), httpx.ConnectError),
        (lambda: httpx.ReadError("reset"), httpx.ReadError),
        (lambda: httpx.ReadTimeout("slow"), httpx.ReadTimeout),
    ],
    ids=[
        *[
            f"{kind}_{status}"
            for kind in ("fast", "streamed")
            for status in (408, 502, 503, 504)
        ],
        "connect_error",
        "read_error",
        "read_timeout",
    ],
)


def _job_id(index: int) -> UUID:
    return UUID(int=0x100 + index)


def _chunk_result(index: int) -> PredictionResult:
    return PredictionResult(y_pred=np.array([index]), fitted_test_set_id=_job_id(index))


@contextmanager
def _chunk_server(
    *,
    sync: frozenset[int] = frozenset(),
    stop_start: dict[int, BaseException] | None = None,
    stop_wait: dict[int, BaseException] | None = None,
) -> Generator[list[str]]:
    """Serve `predict_chunks` for test sets whose cells hold their position.

    Each starts as a job, except those in `sync`, predicted with one request.
    Starting or waiting for the one at position i raises `stop_start[i]` or
    `stop_wait[i]`. Yields the starts and waits, in order.
    """
    events: list[str] = []

    def start(fitted_train_set_id: UUID, x_chunk: Any, *args: Any, **kwargs: Any):
        index = int(x_chunk[0][0])
        events.append(f"start {index}")
        if index in (stop_start or {}):
            raise (stop_start or {})[index]
        return _chunk_result(index) if index in sync else _job_id(index)

    def wait(fitted_test_set_id: UUID, *args: Any, **kwargs: Any) -> PredictionResult:
        index = fitted_test_set_id.int - 0x100
        events.append(f"wait {index}")
        if index in (stop_wait or {}):
            raise (stop_wait or {})[index]
        return _chunk_result(index)

    with (
        patch.object(ServiceClient, "_start_predict", autospec=True, side_effect=start),
        patch.object(
            ServiceClient, "_await_prediction", autospec=True, side_effect=wait
        ),
    ):
        yield events


def _predict_chunks(n_chunks: int) -> list[PredictionResult]:
    return ServiceClient.predict_chunks(
        FIT_ID,
        (np.full((1, 2), index) for index in range(n_chunks)),
        ClassifierConfig(),
        api_mode=ApiMode.ASYNC,
    )


class TestPredictChunks:
    """`ServiceClient.predict_chunks`, which a regressor's split full output
    goes through: its background predictions overlap, up to 4 at once."""

    def test_keeps_up_to_four_jobs_running_and_returns_them_in_order(self):
        with _chunk_server() as events:
            results = _predict_chunks(6)

        assert events == [
            *(f"start {i}" for i in range(4)),
            "wait 0",
            "start 4",
            "wait 1",
            "start 5",
            *(f"wait {i}" for i in range(2, 6)),
        ]
        assert [int(cast(np.ndarray, r.y_pred)[0]) for r in results] == list(range(6))

    def test_a_test_set_predicted_with_one_request_keeps_its_place(self):
        with _chunk_server(sync=frozenset({1})) as events:
            results = _predict_chunks(3)

        assert events == ["start 0", "start 1", "start 2", "wait 0", "wait 2"]
        assert [int(cast(np.ndarray, r.y_pred)[0]) for r in results] == [0, 1, 2]

    def test_passes_the_mode_and_the_fit_to_every_test_set(self):
        with (
            _chunk_server(),
            patch.object(
                ServiceClient,
                "_start_predict",
                side_effect=lambda *args, **kwargs: _chunk_result(0),
            ) as start,
        ):
            ServiceClient.predict_chunks(
                FIT_ID,
                [X_TEST, X_TEST],
                ClassifierConfig(),
                api_mode=ApiMode.AUTO,
                n_train_rows=7,
                thinking=True,
            )

        assert start.call_count == 2
        for call in start.call_args_list:
            assert call.args[0] == FIT_ID
            assert call.kwargs == {
                "api_mode": ApiMode.AUTO,
                "n_train_rows": 7,
                "thinking": True,
            }

    def test_a_failed_prediction_stops_the_rest_and_logs_the_running_jobs(self, caplog):
        with _chunk_server(stop_wait={0: PredictFailedError("boom")}) as events:
            with pytest.raises(PredictFailedError):
                _predict_chunks(6)

        # Nothing starts after the failure.
        assert events == [*(f"start {i}" for i in range(4)), "wait 0"]
        warnings = _broke_off_warnings(caplog)
        assert len(warnings) == 3
        for index, warning in zip([1, 2, 3], warnings):
            assert str(_job_id(index)) in warning

    def test_an_interrupted_start_logs_the_running_jobs(self, caplog):
        with _chunk_server(stop_start={2: KeyboardInterrupt()}) as events:
            with pytest.raises(KeyboardInterrupt):
                _predict_chunks(6)

        assert events == ["start 0", "start 1", "start 2"]
        warnings = _broke_off_warnings(caplog)
        assert len(warnings) == 2
        assert str(_job_id(0)) in warnings[0]
        assert str(_job_id(1)) in warnings[1]

    def test_runs_each_test_set_through_the_server_in_order(self, router):
        _cache_settings()
        _upload_route(router)
        job_ids = [_job_id(index) for index in range(5)]
        submit_route = router.post("/tabpfn/submit_predict_job")
        submit_route.side_effect = [
            httpx.Response(200, json={"fitted_test_set_id": str(job_id)})
            for job_id in job_ids
        ]

        def poll(request: httpx.Request) -> httpx.Response:
            job_id = json.loads(request.content)["fitted_test_set_id"]
            index = UUID(job_id).int - 0x100
            return _json(200, _completed(prediction=[index], fitted_test_set_id=job_id))

        poll_route = router.post("/tabpfn/get_predict_result")
        poll_route.side_effect = poll
        options = ClientOptions(timeout=42.0, headers={"sentry-trace": "abc"})

        with _fake_clock():
            results = ServiceClient.predict_chunks(
                FIT_ID,
                [X_TEST] * 5,
                ClassifierConfig(),
                options,
                api_mode=ApiMode.ASYNC,
            )

        assert [r.fitted_test_set_id for r in results] == job_ids
        assert [cast(np.ndarray, r.y_pred).tolist() for r in results] == [
            [index] for index in range(5)
        ]
        assert [_sent_in(call)["fitted_test_set_id"] for call in poll_route.calls] == [
            str(job_id) for job_id in job_ids
        ]
        for call in submit_route.calls:
            assert _sent_in(call) == {
                "test_set_upload_id": UPLOAD_ID,
                "fitted_train_set_id": str(FIT_ID),
                "task_config": {"task": "classification"},
            }
            assert call.request.headers["sentry-trace"] == "abc"
            assert call.request.extensions["timeout"]["read"] == 42.0
        for call in poll_route.calls:
            assert call.request.headers["sentry-trace"] == "abc"

    def test_rejects_an_invalid_mode_before_any_request(self):
        with _chunk_server() as events, pytest.raises(ValueError):
            ServiceClient.predict_chunks(
                FIT_ID, [X_TEST], ClassifierConfig(), api_mode=cast(Any, "ASYNC")
            )

        assert events == []


class TestWaitForPredict:
    """The poll loop: `ServiceClient.wait_for_predict` and what it does with
    each answer a poll can get."""

    def test_returns_a_completed_prediction_streamed_after_keepalive_pings(
        self, router
    ):
        with _polling(router, [_json(200, _completed(), keepalive=b" \n  ")]) as (
            clock,
            route,
        ):
            result = _wait()

        assert clock.sleeps == []
        assert route.call_count == 1
        np.testing.assert_array_equal(result.y_pred, PREDICTION)
        assert result.metadata["task"] == "classification"
        assert result.timings == TIMINGS
        assert result.fitted_test_set_id == PREDICTION_ID

    def test_converts_a_polled_prediction_as_an_inline_one(self, router):
        regression = {**METADATA, "task": "regression"}
        full = {"borders": [0.0, None, 2.0], "logits": [[1.0, None], [None, 4.0]]}

        with _polling(
            router, [_json(200, _completed(prediction=full, metadata=regression))]
        ):
            result = _wait()

        y_pred = cast("dict[str, np.ndarray]", result.y_pred)
        np.testing.assert_allclose(y_pred["borders"], [0.0, np.nan, 2.0])
        np.testing.assert_allclose(y_pred["logits"], [[1.0, np.nan], [np.nan, 4.0]])

    def test_a_malformed_poll_body_is_retried_as_a_capped_error(self, router, caplog):
        with (
            _polling(
                router,
                [_json(200, {"unexpected": "shape"}), _json(200, _completed())],
            ) as (clock, route),
            caplog.at_level(logging.WARNING, logger="tabpfn_client.client"),
        ):
            result = _wait()

        assert route.call_count == 2
        assert clock.sleeps == [5.0]
        assert result.fitted_test_set_id == PREDICTION_ID
        assert "Unexpected error while polling status of prediction" in caplog.text

    def test_the_fourth_malformed_poll_body_in_a_row_raises(self, router):
        with _polling(router, [_json(200, {"unexpected": "shape"})] * 4) as (_, route):
            with pytest.raises(RuntimeError) as exc_info:
                _wait()

        assert route.call_count == 4
        assert str(exc_info.value).startswith(
            f"Polling prediction {PREDICTION_ID} failed 4 times in a row:"
        )
        cause = exc_info.value.__cause__
        assert isinstance(cause, CappedRetryableServerError)
        assert isinstance(cause.__cause__, ValidationError)

    def test_a_status_this_client_does_not_know_counts_as_still_running(self, router):
        unknown = {**_pending(1.0), "status": "queued"}

        with _polling(router, [_json(200, unknown), _json(200, _completed())]) as (
            clock,
            route,
        ):
            _wait()

        assert clock.sleeps == [1.0]
        assert route.call_count == 2

    def test_floors_the_servers_retry_hint(self, router):
        with _polling(
            router,
            [
                _json(200, _pending(0.0)),
                _json(200, _pending(-3.0)),
                _json(200, _completed()),
            ],
        ) as (clock, _):
            _wait()

        assert clock.sleeps == [_MIN_RETRY_INTERVAL_SECS, _MIN_RETRY_INTERVAL_SECS]

    def test_polls_every_5_seconds_until_the_first_hint_then_keeps_the_last_one(
        self, router
    ):
        with _polling(
            router,
            [
                _error(503, "Overloaded"),
                _json(200, _pending(2.0)),
                _json(200, _pending(None)),
                _error(503, "Overloaded"),
                _json(200, _pending(None)),
                _json(200, _completed()),
            ],
        ) as (clock, _):
            _wait()

        # A failed poll waits at least the 5 s fallback; the hint outlasts it.
        assert clock.sleeps == [5.0, 2.0, 2.0, 5.0, 2.0]

    def test_a_failed_poll_waits_at_least_5_seconds_and_a_success_restores_the_hint(
        self, router
    ):
        with _polling(
            router,
            [
                _json(200, _pending(0.5)),
                _error(503, "Overloaded"),
                httpx.ConnectError("refused"),
                _json(200, _pending(None)),
                _json(200, _completed()),
            ],
        ) as (clock, _):
            _wait()

        assert clock.sleeps == [0.5, 5.0, 5.0, 0.5]

    @CAPPED_ERRORS
    def test_a_capped_error_waits_at_least_5_seconds_as_a_transient_one_does(
        self, router, make_error, cause
    ):
        with _polling(
            router,
            [
                _json(200, _pending(0.5)),
                make_error(),
                _json(200, _pending(None)),
                _json(200, _completed()),
            ],
        ) as (clock, _):
            _wait()

        assert clock.sleeps == [0.5, 5.0, 0.5]

    @CAPPED_ERRORS
    def test_a_run_of_capped_errors_waits_at_least_5_seconds_after_each(
        self, router, make_error, cause
    ):
        with _polling(
            router,
            [
                _json(200, _pending(0.5)),
                make_error(),
                make_error(),
                _json(200, _pending(None)),
                _json(200, _completed()),
            ],
        ) as (clock, _):
            _wait()

        assert clock.sleeps == [0.5, 5.0, 5.0, 0.5]

    def test_the_deadline_counts_the_5_seconds_a_failed_poll_waits(self, router):
        # Timeout 3: the poll at t=0.5 fails and leaves 2.5 s, too few for the
        # next one.
        with _polling(
            router,
            [
                _json(200, _pending(0.5)),
                _error(503, "Overloaded"),
                _json(200, _completed()),
            ],
            poll_timeout=3.0,
        ) as (clock, route):
            with pytest.raises(TimeoutError):
                _wait()

        assert clock.sleeps == [0.5]
        assert route.call_count == 2

    @ROUTE_MISSING
    def test_stops_at_once_on_a_server_without_the_endpoint(
        self, router, caplog, route_missing
    ):
        with _polling(router, [_json(404, route_missing)]) as (clock, route):
            with pytest.raises(RuntimeError) as exc_info:
                _wait()

        assert str(exc_info.value) == (
            f"Cannot wait for prediction {PREDICTION_ID}: this TabPFN server does "
            "not run predictions as background jobs right now. Wait for it again "
            "later: a server that runs them again may still complete it."
        )
        assert not isinstance(exc_info.value, PredictFailedError)
        assert clock.sleeps == []
        assert route.call_count == 1
        # The prediction may still be running, on the server it was sent to.
        assert _broke_off_warnings(caplog) == [WAIT_BROKE_OFF]

    @pytest.mark.parametrize(
        ("status_code", "message", "error_code"),
        [
            (404, "Prediction not found", "NOT_FOUND"),
            (410, EXPIRED, "USER_ERROR"),
            (403, "Forbidden", "USER_ERROR"),
            (422, "Validation error", "REQUEST_VALIDATION_ERROR"),
        ],
        ids=["unknown", "expired", "forbidden", "invalid"],
    )
    @pytest.mark.parametrize("streamed", [False, True], ids=["fast", "streamed"])
    def test_stops_at_once_on_an_error_no_retry_can_clear(
        self, router, status_code, message, error_code, streamed
    ):
        with _polling(
            router,
            [_error(status_code, message, streamed=streamed, error_code=error_code)],
        ) as (clock, route):
            with pytest.raises(RuntimeError) as exc_info:
                _wait()

        error = exc_info.value
        if status_code in (404, 410):
            # The server does not have the prediction.
            assert type(error) is PredictionGoneError
            assert str(error) == (
                f"The server does not have prediction {PREDICTION_ID}: "
                f"{error.__cause__}"
            )
            error = error.__cause__
        if streamed:
            assert isinstance(error, StreamedServerError)
            assert error.status_code == status_code
            assert str(error) == (
                f"Fail to call get_predict_result with error: streamed, {message}"
            )
        else:
            assert type(error) is RuntimeError
            assert f"[HTTP {status_code}] {message}" in str(error)
        assert clock.sleeps == []
        assert route.call_count == 1

    @CAPPED_ERRORS
    def test_the_fourth_capped_error_in_a_row_raises(self, router, make_error, cause):
        with _polling(router, [make_error() for _ in range(4)]) as (clock, route):
            with pytest.raises(RuntimeError) as exc_info:
                _wait()

        # A RuntimeError whatever the cause, which is chained.
        assert type(exc_info.value) is RuntimeError
        last_error = exc_info.value.__cause__
        assert isinstance(last_error, cause)
        assert str(exc_info.value) == (
            f"Polling prediction {PREDICTION_ID} failed 4 times in a row: {last_error}"
        )
        assert route.call_count == 4
        assert clock.sleeps == [5.0] * 3

    @CAPPED_ERRORS
    def test_a_successful_poll_resets_the_capped_count(self, router, make_error, cause):
        outcomes = [
            *[make_error() for _ in range(3)],
            _json(200, _pending(1.0)),
            *[make_error() for _ in range(3)],
            _json(200, _completed()),
        ]

        with _polling(router, outcomes) as (_, route):
            result = _wait()

        assert route.call_count == 8
        assert result.fitted_test_set_id == PREDICTION_ID

    def test_a_transient_error_ends_a_run_of_capped_errors(self, router):
        outcomes = [
            *[_error(429, "Rate limit exceeded") for _ in range(3)],
            _error(503, "Overloaded"),
            _error(429, "Rate limit exceeded"),
            _json(200, _completed()),
        ]

        with _polling(router, outcomes) as (_, route):
            result = _wait()

        assert route.call_count == 6
        assert result.fitted_test_set_id == PREDICTION_ID

    @TRANSIENT_ERRORS
    def test_a_transient_error_is_retried_until_the_deadline(
        self, router, make_error, cause
    ):
        # Timeout 30, interval 5: polls at t=0/5/10/15/20/25, more than a capped
        # error allows. After the t=25 poll, the 5s left cannot fit a sleep.
        with _polling(router, [make_error() for _ in range(6)], poll_timeout=30.0) as (
            clock,
            route,
        ):
            with pytest.raises(TimeoutError) as exc_info:
                _wait()

        assert route.call_count == 6
        assert clock.sleeps == [5.0] * 5
        last_error = exc_info.value.__cause__
        assert isinstance(last_error, cause)
        assert str(exc_info.value) == (
            f"Prediction {PREDICTION_ID} did not reach a terminal state within the "
            f"poll timeout of 30.0 seconds. The last status poll failed: {last_error}"
        )

    def test_the_deadline_is_a_hard_bound(self, router, caplog):
        # Timeout 12, interval 5: polls at t=0/5/10. After the t=10 poll only 2s
        # remain, so the loop gives up instead of polling once more past it.
        with _polling(
            router, [_json(200, _pending(5.0)) for _ in range(3)], poll_timeout=12.0
        ) as (clock, route):
            with pytest.raises(TimeoutError) as exc_info:
                _wait()

        assert str(exc_info.value) == (
            f"Prediction {PREDICTION_ID} did not reach a terminal state within the "
            "poll timeout of 12.0 seconds."
        )
        # The prediction was still running: no poll error to chain.
        assert exc_info.value.__cause__ is None
        assert clock.sleeps == [5.0, 5.0]
        assert route.call_count == 3
        assert _broke_off_warnings(caplog) == [WAIT_BROKE_OFF]

    def test_the_timeout_chains_a_capped_error_that_lasted_to_the_deadline(
        self, router
    ):
        # Timeout 12, interval 5: polls at t=0/5/10, all refused.
        with _polling(
            router,
            [_error(429, "Rate limit exceeded") for _ in range(3)],
            poll_timeout=12.0,
        ):
            with pytest.raises(TimeoutError) as exc_info:
                _wait()

        assert isinstance(exc_info.value.__cause__, CappedRetryableServerError)

    def test_the_timeout_chains_no_error_a_later_poll_cleared(self, router):
        with _polling(
            router,
            [
                _error(503, "Overloaded"),
                _json(200, _pending(5.0)),
                _json(200, _pending(5.0)),
            ],
            poll_timeout=12.0,
        ):
            with pytest.raises(TimeoutError) as exc_info:
                _wait()

        assert exc_info.value.__cause__ is None
        assert "The last status poll failed" not in str(exc_info.value)

    def test_logs_the_id_of_a_prediction_whose_wait_is_interrupted(self, caplog):
        with (
            patch.object(
                ServiceClient,
                "_resolve_async_settings",
                return_value=_async_settings(),
            ),
            patch.object(
                ServiceClient, "_get_predict_result", side_effect=KeyboardInterrupt
            ),
            pytest.raises(KeyboardInterrupt),
        ):
            _wait()

        assert _broke_off_warnings(caplog) == [WAIT_BROKE_OFF]

    def test_logs_the_id_of_a_prediction_the_client_may_not_poll_for(
        self, router, caplog
    ):
        with (
            _polling(router, [_error(401, "Requires authentication")]),
            pytest.raises(RuntimeError) as exc_info,
        ):
            _wait()

        # The prediction runs on while the client cannot poll for it.
        assert type(exc_info.value) is RuntimeError
        assert _broke_off_warnings(caplog) == [WAIT_BROKE_OFF]

    @pytest.mark.parametrize(
        ("make_answer", "shown"),
        [
            (
                lambda: httpx.Response(
                    404,
                    text="<html><body>Not Found</body></html>",
                    headers={"content-type": "text/html"},
                ),
                "Not Found",
            ),
            (lambda: _json(404, {"detail": "Not Found"}), "Not Found"),
            (
                lambda: _json(404, {"message": "no Route matched with those values"}),
                "no Route matched with those values",
            ),
        ],
        ids=["html", "starlette_default", "gateway"],
    )
    def test_logs_the_id_of_a_prediction_after_a_404_without_the_servers_code(
        self, router, caplog, make_answer, shown
    ):
        with _polling(router, [make_answer()]) as (clock, route):
            with pytest.raises(RuntimeError) as exc_info:
                _wait()

        # Only the server's NOT_FOUND code says that it does not have the
        # prediction; a 404 from a proxy, say, tells nothing about it.
        assert type(exc_info.value) is RuntimeError
        assert str(exc_info.value) == (
            f"Fail to call get_predict_result: [HTTP 404] {shown}."
        )
        assert clock.sleeps == []
        assert route.call_count == 1
        assert _broke_off_warnings(caplog) == [WAIT_BROKE_OFF]

    @pytest.mark.parametrize(
        ("error", "error_code"),
        [("boom", "INTERNAL_ERROR"), (NEEDS_REFITTING, "NEEDS_REFITTING")],
        ids=["failed", "needs_refitting"],
    )
    def test_a_failed_prediction_logs_nothing(self, router, caplog, error, error_code):
        with (
            _polling(router, [_json(200, _failed(error, error_code))]),
            pytest.raises(PredictFailedError),
        ):
            _wait()

        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    @pytest.mark.parametrize(
        ("status_code", "message", "error_code"),
        [(404, "Prediction not found", "NOT_FOUND"), (410, EXPIRED, "USER_ERROR")],
        ids=["unknown", "expired"],
    )
    @pytest.mark.parametrize("streamed", [False, True], ids=["fast", "streamed"])
    def test_a_prediction_the_server_does_not_have_logs_nothing(
        self, router, caplog, status_code, message, error_code, streamed
    ):
        answer = _error(status_code, message, streamed=streamed, error_code=error_code)

        with _polling(router, [answer]), pytest.raises(PredictionGoneError):
            _wait()

        assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []

    @pytest.mark.parametrize(
        ("status_code", "message", "error_code"),
        [(404, "Prediction not found", "NOT_FOUND"), (410, EXPIRED, "USER_ERROR")],
        ids=["unknown", "expired"],
    )
    @pytest.mark.parametrize("streamed", [False, True], ids=["fast", "streamed"])
    def test_warns_of_a_deprecated_client_on_a_prediction_the_server_does_not_have(
        self, router, status_code, message, error_code, streamed
    ):
        answer = _error(
            status_code,
            message,
            streamed=streamed,
            error_code=error_code,
            headers={"deprecation": "true"},
        )

        with (
            _polling(router, [answer]),
            pytest.warns(DeprecationWarning, match="tabpfn-client is deprecated"),
            pytest.raises(PredictionGoneError),
        ):
            _wait()

    def test_each_poll_uses_the_request_timeout_and_headers(self, router):
        options = ClientOptions(timeout=42.0, headers={"sentry-trace": "abc"})

        with _polling(
            router, [_json(200, _pending(1.0)), _json(200, _completed())]
        ) as (_, route):
            _wait(client_options=options)

        assert route.call_count == 2
        for call in route.calls:
            assert call.request.extensions["timeout"] == dict.fromkeys(
                ("connect", "read", "write", "pool"), 42.0
            )
            assert call.request.headers["sentry-trace"] == "abc"

    def test_a_poll_uses_the_client_timeout_when_the_request_has_none(self, router):
        # Below the cap on a poll's timeout.
        get_opts().TABPFN_CLIENT_TIMEOUT = 99.0
        options = ClientOptions(timeout=cast(Any, None))

        with _polling(router, [_json(200, _completed())]) as (_, route):
            _wait(client_options=options)

        assert route.calls.last.request.extensions["timeout"]["read"] == 99.0

    @pytest.mark.parametrize(
        "timeout", [900.0, None], ids=["request_timeout", "client_timeout"]
    )
    def test_caps_the_timeout_of_a_poll_at_2_minutes(self, router, timeout):
        # The default, whatever the environment sets.
        get_opts().TABPFN_CLIENT_TIMEOUT = 900.0
        options = ClientOptions(timeout=cast(Any, timeout))

        with _polling(router, [_json(200, _completed())]) as (_, route):
            _wait(client_options=options)

        assert route.calls.last.request.extensions["timeout"] == dict.fromkeys(
            ("connect", "read", "write", "pool"), 120.0
        )

    @pytest.mark.parametrize("missing", ["prediction", "metadata"])
    def test_a_completed_poll_without_the_prediction_fails_it(
        self, router, missing, caplog
    ):
        body = {**_completed(), missing: None}

        with _polling(router, [_json(200, body)]):
            with pytest.raises(
                PredictFailedError, match="sent no prediction"
            ) as exc_info:
                _wait()

        assert exc_info.value.fitted_test_set_id == PREDICTION_ID
        # A later poll would answer the same: nothing to come back for.
        assert _broke_off_warnings(caplog) == []

    def test_a_malformed_id_raises_before_any_poll(self, router, caplog):
        # No poll route: a poll would fail the test.
        with pytest.raises(ValueError):
            ServiceClient.wait_for_predict("not-a-prediction-id")

        # No wait started, so there is no prediction to come back for.
        assert _broke_off_warnings(caplog) == []

    def test_a_predict_that_breaks_off_while_polled_logs_the_prediction_id(
        self, router, caplog
    ):
        _cache_settings()
        _upload_route(router)
        router.post("/tabpfn/predict").respond(200, json=_predict_pending())

        with (
            _fake_clock(),
            patch.object(
                ServiceClient, "_get_predict_result", side_effect=KeyboardInterrupt
            ),
            pytest.raises(KeyboardInterrupt),
        ):
            _predict()

        assert _broke_off_warnings(caplog) == [WAIT_BROKE_OFF]


class TestInferenceClient:
    def test_predict_defaults_to_one_request(self):
        with patch.object(ServiceClient, "predict") as predict:
            InferenceClient.predict(
                X_TEST, fitted_train_set_id=FIT_ID, task_config=ClassifierConfig()
            )

        kwargs = predict.call_args.kwargs
        assert kwargs["api_mode"] is ApiMode.SYNC
        assert kwargs["n_train_rows"] is None
        assert kwargs["thinking"] is False

    def test_predict_passes_the_mode_and_what_auto_decides_on(self):
        options = ClientOptions()
        task_config = ClassifierConfig()

        with patch.object(ServiceClient, "predict") as predict:
            InferenceClient.predict(
                X_TEST,
                fitted_train_set_id=FIT_ID,
                task_config=task_config,
                client_options=options,
                api_mode=ApiMode.AUTO,
                n_train_rows=7,
                thinking=True,
            )

        kwargs = predict.call_args.kwargs
        assert kwargs["x_test"] is X_TEST
        assert kwargs["fitted_train_set_id"] == FIT_ID
        assert kwargs["task_config"] is task_config
        assert kwargs["client_options"] is options
        assert kwargs["api_mode"] is ApiMode.AUTO
        assert kwargs["n_train_rows"] == 7
        assert kwargs["thinking"] is True

    def test_predict_chunks_passes_everything_through(self):
        options = ClientOptions()
        task_config = ClassifierConfig()
        x_chunks = [X_TEST, X_TEST]

        with patch.object(ServiceClient, "predict_chunks", return_value=[]) as chunks:
            InferenceClient.predict_chunks(
                x_chunks,
                fitted_train_set_id=FIT_ID,
                task_config=task_config,
                client_options=options,
                api_mode=ApiMode.AUTO,
                n_train_rows=7,
                thinking=True,
            )

        kwargs = chunks.call_args.kwargs
        assert kwargs["x_chunks"] is x_chunks
        assert kwargs["fitted_train_set_id"] == FIT_ID
        assert kwargs["task_config"] is task_config
        assert kwargs["client_options"] is options
        assert kwargs["api_mode"] is ApiMode.AUTO
        assert kwargs["n_train_rows"] == 7
        assert kwargs["thinking"] is True

    def test_submit_predict_returns_the_id(self):
        options = ClientOptions()
        task_config = ClassifierConfig()

        with patch.object(
            ServiceClient, "submit_predict", return_value=PREDICTION_ID
        ) as submit:
            fitted_test_set_id = InferenceClient.submit_predict(
                X_TEST,
                fitted_train_set_id=FIT_ID,
                task_config=task_config,
                client_options=options,
            )

        assert fitted_test_set_id == PREDICTION_ID
        kwargs = submit.call_args.kwargs
        assert kwargs["x_test"] is X_TEST
        assert kwargs["fitted_train_set_id"] == FIT_ID
        assert kwargs["task_config"] is task_config
        assert kwargs["client_options"] is options

    def test_wait_for_predict_polls_with_the_whole_client_options(self, router):
        options = ClientOptions(timeout=42.0, headers={"sentry-trace": "abc"})

        with _polling(router, [_json(200, _completed())]) as (_, route):
            result = InferenceClient.wait_for_predict(
                PREDICTION_ID, client_options=options
            )

        request = route.calls.last.request
        assert request.extensions["timeout"]["read"] == 42.0
        assert request.headers["sentry-trace"] == "abc"
        assert result.fitted_test_set_id == PREDICTION_ID

    def test_wait_for_predict_takes_the_id_as_a_string(self, router):
        with _polling(router, [_json(200, _completed())]) as (_, route):
            result = InferenceClient.wait_for_predict(str(PREDICTION_ID))

        assert _sent(route) == {"fitted_test_set_id": str(PREDICTION_ID)}
        # A UUID, as for an id passed as one.
        assert result.fitted_test_set_id == PREDICTION_ID


def test_the_prediction_errors_are_public():
    assert {"PredictFailedError", "RefitRequiredError"} <= set(tabpfn_client.__all__)
    assert issubclass(
        tabpfn_client.RefitRequiredError, tabpfn_client.PredictFailedError
    )
    assert issubclass(tabpfn_client.PredictFailedError, RuntimeError)


@pytest.mark.parametrize(
    ("envelope", "status_code", "detail"),
    [
        ({"status_code": 503, "detail": "Overloaded"}, 503, "Overloaded"),
        ({"message": "Overloaded"}, None, "Overloaded"),
    ],
    ids=["with_status", "without_status"],
)
def test_a_streamed_error_keeps_the_status_it_reports(envelope, status_code, detail):
    response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://testserver/test"),
        content=b"  " + json.dumps({"_streamed_error": True, **envelope}).encode(),
    )

    with pytest.raises(StreamedServerError) as exc_info:
        ServiceClient._validate_response(response, "fit", success_model=FitResponse)

    assert exc_info.value.status_code == status_code
    assert exc_info.value.detail == detail
    assert str(exc_info.value) == f"Fail to call fit with error: streamed, {detail}"


def _streamed(envelope: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("POST", "http://testserver/test"),
        content=b"  " + json.dumps({"_streamed_error": True, **envelope}).encode(),
    )


def test_a_streamed_error_parses_into_the_error_model_of_its_status():
    response = _streamed(
        {
            "message": "Overloaded",
            "error_code": "BUSY",
            "trace_id": TRACE_ID,
            "fitted_test_set_id": str(PREDICTION_ID),
            "status_code": 503,
            "detail": "Overloaded",
        }
    )

    parsed = ServiceClient._validate_response(
        response,
        "predict",
        success_model=FitResponse,
        error_models={503: PredictFailedErrorResponse},
    )

    assert parsed == PredictFailedErrorResponse(
        message="Overloaded",
        error_code="BUSY",
        trace_id=UUID(TRACE_ID),
        fitted_test_set_id=PREDICTION_ID,
    )


@pytest.mark.parametrize(
    "envelope",
    [
        # From a server that streams no body fields.
        {"status_code": 503, "detail": "Overloaded"},
        # A status without an error model.
        {
            "message": "Overloaded",
            "fitted_test_set_id": str(PREDICTION_ID),
            "status_code": 502,
            "detail": "Overloaded",
        },
    ],
    ids=["without_the_fields", "another_status"],
)
def test_a_streamed_error_without_a_matching_model_still_raises(envelope):
    with pytest.raises(StreamedServerError) as exc_info:
        ServiceClient._validate_response(
            _streamed(envelope),
            "predict",
            success_model=FitResponse,
            error_models={503: PredictFailedErrorResponse},
        )

    assert exc_info.value.status_code == envelope["status_code"]
    assert exc_info.value.detail == "Overloaded"


@pytest.mark.parametrize(
    "error",
    [
        StreamedServerError(
            "Fail to call get_predict_result with error: streamed, Overloaded",
            status_code=503,
            detail="Overloaded",
        ),
        PredictFailedError(f"Prediction {PREDICTION_ID} failed: boom"),
        PredictFailedError(
            f"Prediction {PREDICTION_ID} failed: boom",
            fitted_test_set_id=PREDICTION_ID,
        ),
        RefitRequiredError(NEEDS_REFITTING),
        RefitRequiredError(NEEDS_REFITTING, fitted_test_set_id=PREDICTION_ID),
        PredictionGoneError(f"The server does not have prediction {PREDICTION_ID}"),
    ],
    ids=[
        "streamed",
        "failed",
        "failed_with_its_id",
        "refit_required",
        "refit_required_with_its_id",
        "gone",
    ],
)
@pytest.mark.parametrize(
    "duplicate",
    [lambda error: pickle.loads(pickle.dumps(error)), copy.copy, copy.deepcopy],
    ids=["pickle", "copy", "deepcopy"],
)
def test_an_error_survives_pickle_and_copy(error, duplicate):
    # A process pool hands the error of a worker to its parent as a pickle.
    duplicated = duplicate(error)

    assert type(duplicated) is type(error)
    assert str(duplicated) == str(error)
    # The status and detail of a streamed error, the id of a failed prediction.
    assert vars(duplicated) == vars(error)
