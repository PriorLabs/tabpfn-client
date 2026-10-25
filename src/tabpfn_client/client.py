#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Low-level HTTP client for the TabPFN API."""

from __future__ import annotations

from uuid import UUID
import base64
from collections import deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import PackageNotFoundError, version
import io
import json
import logging
from pathlib import Path
import numpy as np
import re
import struct
import time
import warnings
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing import Any, Callable, Iterable, cast, Mapping, NoReturn, TypeAlias
from typing_extensions import (
    Never,  # in `typing` only from Python 3.11
    TypeVar,  # supports default= on Python 3.10
    assert_never,  # in `typing` only from Python 3.11
)
from tabpfn_client.models import (
    ClientOptions,
    PredictionResult,
    ApiMode,
)
from tabpfn_client.errors import (
    RetryableServerError,
    CappedRetryableServerError,
    EmptyResponseError,
    FitFailedError,
    FittedModelNotFoundError,
    PredictFailedError,
    PredictionGoneError,
    RefitRequiredError,
    StreamedServerError,
)

import google_crc32c

import pandas as pd

import backoff
import httpx
from omegaconf import DictConfig, OmegaConf
from tabpfn_common_utils.utils import Singleton
from tabpfn_client.api_models import (
    GetSettingsResponse,
    EstimateCostRequest,
    EstimateCostResponse,
    TabPFNSystem,
    TextHandling,
    PrepareTrainSetUploadRequest,
    PrepareTrainSetUploadResponse,
    DuplicateTrainSetErrorResponse,
    PrepareTestSetUploadRequest,
    PrepareTestSetUploadResponse,
    DuplicateTestSetErrorResponse,
    NotFoundErrorResponse,
    FileUploadInfo,
    FileInfo,
    FitRequest,
    FitResponse,
    SubmitFitJobRequest,
    FitMode,
    FitStatus,
    GetFitStatusRequest,
    GetFitStatusResponse,
    PredictRequest,
    Metadata,
    PredictResponse,
    PredictFailedErrorResponse,
    PredictPendingResponse,
    PredictStatus,
    PredictTimings,
    ClassifierConfig,
    RegressorConfig,
    SubmitFitJobResponse,
    SubmitPredictJobRequest,
    SubmitPredictJobResponse,
    GetPredictResultRequest,
    GetPredictResultResponse,
    ThinkingConfig,
    FitTaskConfig,
    Prediction,
)
from tabpfn_client.options import get_opts
from tabpfn_client.utils import validate_text_handling

logger = logging.getLogger(__name__)


class PredictResponseWithDownloadURI(BaseModel):
    """A predict answered with a signed download URL for the prediction.

    Mirrors the server's model of the same name, which the generated API models
    leave out: the server answers so only when asked to, which this client
    never does, but the client follows the URL if it ever gets one.
    """

    prediction_uri: str
    prediction_uri_expires_in_secs: int
    metadata: Metadata
    timings: PredictTimings | None = None
    fitted_test_set_id: UUID | None = None


PredictResponseUnion: TypeAlias = (
    PredictResponse | PredictResponseWithDownloadURI | PredictPendingResponse
)

_PredictResponseUnion: TypeAdapter[PredictResponseUnion] = TypeAdapter(
    PredictResponseUnion
)

# Statuses from a signed-URL upload or download that a retry can clear.
_RETRYABLE_STORAGE_STATUSES = frozenset({408, 429, 502, 503, 504})

# avoid logging of httpx and httpcore on client side
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)


def _on_backoff(details: Mapping[str, Any]):
    """Callback function for retry attempts."""
    function_name = details["target"].__name__
    message = (
        f"Exception occurred during {function_name}, retrying in {details['wait']} seconds... "
        f"attempt {details['tries']}: {details['exception']}"
    )
    logger.warning(message)


def _on_giveup(details: Mapping[str, Any]):
    """Callback function when retries are exhausted."""
    function_name: str = details["target"].__name__.title()
    message = (
        f"{function_name} method failed after {details['tries']} attempts. "
        f"Giving up. Exception: {details['exception']}"
    )
    logger.error(message)


def _contains_none(value: Any) -> bool:
    """Return whether ``value`` is None or a (nested) list containing None."""
    if value is None:
        return True
    if isinstance(value, list):
        return any(_contains_none(item) for item in value)
    return False


httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

SERVER_CONFIG_FILE = Path(__file__).parent.resolve() / "server_config.yaml"
SERVER_CONFIG = cast(DictConfig, OmegaConf.load(SERVER_CONFIG_FILE))


def get_client_version() -> str:
    """Return the installed package version, or ``"5.5.5"`` if not installed."""
    try:
        return version("tabpfn_client")
    except PackageNotFoundError:
        # Package not found, should only happen during development. Execute 'pip install -e .' to use the actual
        # version number during development. Otherwise, simply return a version number that is large enough.
        return "5.5.5"


def _get_crc32c_hash(data: bytes) -> str:
    """Compute the CRC32C checksum of ``data`` as a base64-encoded string."""
    crc32c_value = google_crc32c.value(data)
    return base64.b64encode(struct.pack(">I", crc32c_value)).decode("ascii")


def _report_trace(trace_id: UUID | None) -> str:
    """Return the sentence that asks to report ``trace_id``, or "" without one.

    The server sends a trace id without dashes, as its logs and Sentry show it,
    and the message keeps that form so that the id can be searched as it is.
    ``str()`` of the parsed UUID would add the dashes.
    """
    return f" Report trace ID: {trace_id.hex}." if trace_id else ""


def _fitted_model_not_found(
    fitted_train_set_id: UUID, response: NotFoundErrorResponse
) -> FittedModelNotFoundError:
    """Build the error for a fit id the server does not know."""
    message = (
        f"Fitted model {fitted_train_set_id} was not found on the server. "
        "It may have been deleted (see UserDataClient), or it was fitted "
        "under a different account. Call fit() again to create a new one."
    )
    message += _report_trace(response.trace_id)
    return FittedModelNotFoundError(message)


_NO_PREDICT_JOBS = (
    "This TabPFN server does not run predictions as background jobs; "
    "use predict() instead."
)

# The error codes of a failed prediction that only a new fit can fix: a fit
# from an earlier version of the service, and a fit whose cache can no longer
# be loaded. The API models name them only in the description of
# `PredictFailedErrorResponse.error_code`.
_REFIT_ERROR_CODES = frozenset({"NEEDS_REFITTING", "KV_CACHE_INVALID"})

# A server that predates KV_CACHE_INVALID books a stale or missing KV cache as a
# USER_ERROR, and streams it, once its keepalive started, with no code at all:
# only the message tells that the model must be fitted again. The fallback
# applies to these codes alone, as a server with the refit codes sends one of
# them for every refit. Drop it once no such server is left.
_LEGACY_REFIT_CODES: frozenset[str | None] = frozenset({None, "USER_ERROR"})
_REFIT_TEXT = "call fit() again"


def _requires_refit(
    error_code: str | None,
    message: str | None,
    *,
    status_code: int | None = None,
) -> bool:
    """Return whether a failed prediction means the model must be fitted again.

    The error code decides. A failure with a code from a server that predates
    the refit codes falls back to the status and the message.

    Args:
        error_code: The server's error code, or None if it sent none.
        message: The server's reason, or None if it sent none.
        status_code: The HTTP status of a failed ``/tabpfn/predict`` whose
            body does not name the prediction, or None. Such a server predates
            ``PredictFailedErrorResponse``, and every 409 it answers asks for
            a new fit.

    Returns:
        True if the model must be fitted again.
    """
    if error_code in _REFIT_ERROR_CODES:
        return True
    if error_code not in _LEGACY_REFIT_CODES:
        return False
    return status_code == 409 or (message is not None and _REFIT_TEXT in message)


def _prediction_failed(
    fitted_test_set_id: UUID,
    error: str | None,
    error_code: str | None,
    *,
    trace_id: UUID | None = None,
) -> PredictFailedError:
    """Build the error for a prediction the server reports as failed.

    Args:
        fitted_test_set_id: The id of the prediction.
        error: The server's reason, or None if it sent none.
        error_code: The server's error code, or None if it sent none.
        trace_id: The trace id of the request the server answered with the
            failure, or None if it sent none, as a poll does not.

    Returns:
        A ``RefitRequiredError`` if the model must be fitted again, else a
        ``PredictFailedError``.
    """
    message = (
        f"Prediction {fitted_test_set_id} failed: "
        f"{error or 'no error message provided'}"
    )
    if error_code:
        message += f" (error code {error_code})"
    message += _report_trace(trace_id)
    if _requires_refit(error_code, error):
        return RefitRequiredError(message, fitted_test_set_id=fitted_test_set_id)
    return PredictFailedError(message, fitted_test_set_id=fitted_test_set_id)


# A prediction that failed once the server had created it is named in the error
# body, whatever the status. A body without the id fails validation and falls
# through to the generic error of its status.
_PREDICT_FAILED_ERRORS = dict.fromkeys(range(400, 600), PredictFailedErrorResponse)


def _prediction_gone(
    fitted_test_set_id: UUID, error: RuntimeError
) -> PredictionGoneError:
    """Build the error for a prediction the server does not have."""
    return PredictionGoneError(
        f"The server does not have prediction {fitted_test_set_id}: {error}"
    )


def _malformed_poll_body(
    method_name: str, error: ValidationError
) -> CappedRetryableServerError:
    """Build the error for a poll answered with a body that fails validation.

    Capped rather than fatal: such a body comes from a broken or foreign
    response, which a later poll may not repeat.
    """
    return CappedRetryableServerError(
        f"Fail to call {method_name}: the server sent a malformed body: {error}"
    )


def _resume_hint(fitted_test_set_id: UUID) -> str:
    """Say how any process gets the result of a prediction, for a log message."""
    return (
        "its result stays on the server for about a day: call "
        "tabpfn_client.init(), then tabpfn_client.service_wrapper."
        f'InferenceClient.wait_for_predict("{fitted_test_set_id}")'
    )


def _warn_stopped_waiting(fitted_test_set_id: UUID) -> None:
    """Log the id of a prediction that may run on after the client stopped waiting.

    The id is the only way back to a prediction that may run on, and be billed,
    after the wait breaks off; Python shows only WARNING and above by default.
    """
    logger.warning(
        f"Stopped waiting for prediction {fitted_test_set_id}. If it is "
        "still running, the server completes it and bills it even if "
        f"nobody waits for it; {_resume_hint(fitted_test_set_id)}"
    )


def _serialize_to_parquet(df: pd.DataFrame) -> tuple[bytes, str]:
    """Serialize ``df`` to zstd-compressed parquet.

    Returns:
        The parquet bytes and their base64-encoded CRC32C checksum.
    """
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    parquet_bytes = buf.getvalue()
    crc32c_b64 = _get_crc32c_hash(parquet_bytes)
    return parquet_bytes, crc32c_b64


SuccessT = TypeVar("SuccessT", bound=BaseModel)
ErrorT = TypeVar("ErrorT", bound=BaseModel, default=Never)
_PollResponseT = TypeVar(
    "_PollResponseT", GetFitStatusResponse, GetPredictResultResponse
)

# Floor for the server's `retry_in_secs` polling hint — defensive: a zero or
# negative value must neither busy-poll the endpoint nor crash `time.sleep`.
_MIN_RETRY_INTERVAL_SECS = 0.3

# Cap on the HTTP timeout of a prediction poll. httpx applies it to each read,
# not to the whole response, and the server sends a byte at least every 5 s
# while it works on a poll, then streams a stored result in chunks. So 2 minutes
# of silence means a lost response, which the next poll gets again; the default
# timeout of 15 minutes would stall the wait instead.
_MAX_POLL_TIMEOUT_SECS = 120.0

# HTTP timeout of a fit status poll, which the server answers at once.
_FIT_STATUS_POLL_TIMEOUT_SECS = 15.0

# Background predictions `predict_chunks` keeps running at once: enough to keep
# the server busy between polls, few enough not to flood its queue with the
# chunks of one call.
_MAX_PREDICT_JOBS_IN_FLIGHT = 4


@dataclass(frozen=True)
class ResolvedAsyncSettings:
    """The settings for background jobs that the client applies.

    Attributes:
        use_above_trainset_size_bytes: The in-memory size of a training set
            above which an AUTO fit runs as a background job.
        poll_timeout_secs: How long the client polls a background job before
            it gives up.
    """

    use_above_trainset_size_bytes: int
    poll_timeout_secs: float


class ServiceClient(Singleton):
    """Singleton class for handling communication with the server.

    It encapsulates all the API calls to the server.
    """

    server_config = SERVER_CONFIG
    server_endpoints = SERVER_CONFIG["endpoints"]
    base_url = (
        get_opts().TABPFN_CLIENT_API_URL
        or f"{server_config.protocol}://{server_config.host}:{server_config.port}"
    )
    # NOTE: HTTP/1.1 only. HTTP/2 used to be selectively enabled for the
    # /tabpfn/fit and /tabpfn/predict endpoints, but the long-running
    # thinking-mode fit kept the stream open for 5-15 min, which raced
    # against intermediate keepalive PINGs from Cloud Run's LB. The
    # `h2` state machine treats a PING received while the connection
    # is CLOSED as a protocol violation and surfaces it as
    # `httpx.LocalProtocolError("Invalid input ConnectionInputs.RECV_PING
    # in state ConnectionState.CLOSED")` — which is NOT in the SDK's
    # retry tuple, so the request fails hard instead of retrying.
    # HTTP/1.1 has no PING frames and no equivalent state machine, so
    # the race disappears. Unary POSTs against /fit and /predict don't
    # benefit from HTTP/2's multiplexing or HPACK in any measurable way
    # (one request per fit, dominated by the multipart body), so the
    # tradeoff is one-sided.
    httpx_client = httpx.Client(
        base_url=base_url,
        timeout=get_opts().TABPFN_CLIENT_TIMEOUT,
        headers={"Prior-Client-Version": get_client_version()},
        follow_redirects=True,
    )
    _access_token: str | None = None
    _api_settings: GetSettingsResponse | None = None
    _api_settings_ts: float = 0.0

    @classmethod
    def get_access_token(cls):
        """Return the access token set by ``authorize``, or None."""
        return cls._access_token

    @classmethod
    def get_settings(cls) -> GetSettingsResponse | None:
        """Fetch and cache dataset limits.

        The cache expires after 1 hour. If fetching fails, the stale cached value
        (or None) is returned. Not thread-safe, but concurrent calls are benign: duplicates fetch the
        same data and the reference assignment is atomic under the GIL.
        """
        ttl = 3600.0  # 1 hour
        if (
            cls._api_settings is not None
            and (time.monotonic() - cls._api_settings_ts) < ttl
        ):
            return cls._api_settings
        try:
            response = cls.httpx_client.get("/tabpfn/get_settings")
            response.raise_for_status()
            cls._api_settings = GetSettingsResponse.model_validate(response.json())
            cls._api_settings_ts = time.monotonic()
            return cls._api_settings
        except Exception:
            logger.debug("Failed to fetch API settings", exc_info=True)
            return cls._api_settings  # return stale value if available

    @classmethod
    def _drop_settings(cls) -> None:
        """Drop the cached settings, so that the next `get_settings()` asks again.

        For settings that describe a server version that is gone, as after a
        rollback: a `get_settings()` that then fails returns None, not them.
        """
        cls._api_settings = None

    @classmethod
    def _resolve_async_settings(cls) -> ResolvedAsyncSettings:
        """Resolve async settings: environment > server default > client default.

        An option explicitly set via the environment wins; otherwise the
        server-sent value applies; the client default is the fallback when the
        server's settings are unavailable.
        """
        opts = get_opts()
        api_settings = cls.get_settings()
        server = api_settings.async_settings if api_settings is not None else None

        def resolve(opt_name: str, server_value: Any) -> Any:
            if opt_name in opts.model_fields_set:
                return getattr(opts, opt_name)
            if server_value is not None:
                return server_value
            return getattr(opts, opt_name)

        return ResolvedAsyncSettings(
            use_above_trainset_size_bytes=resolve(
                "TABPFN_CLIENT_ASYNC_USE_ABOVE_TRAINSET_SIZE",
                server.use_above_trainset_size_bytes if server else None,
            ),
            poll_timeout_secs=resolve(
                "TABPFN_CLIENT_ASYNC_POLL_TIMEOUT",
                server.poll_timeout_secs if server else None,
            ),
        )

    @classmethod
    def _resolve_predict_mode(
        cls,
        api_mode: ApiMode,
        *,
        n_train_rows: int | None,
        n_test_rows: int,
        thinking: bool,
        fit_with_cache: bool,
    ) -> ApiMode:
        """Decide whether a prediction runs as one request or as a polled job.

        SYNC and ASYNC are kept. AUTO runs the prediction as a job once the
        fit's train rows or the test rows reach the server's thresholds: the
        thinking ones for a thinking fit. A fit with a KV cache predicts fast
        whatever its size, so AUTO predicts it with one request, as it does
        when the server's settings are unavailable.

        Args:
            api_mode: The requested mode; a plain string works too, as
                `load_model()` restores the estimator's `api_mode` as one.
            n_train_rows: The fit's train rows, or None if unknown, in which
                case the test rows decide alone.
            n_test_rows: The number of rows to predict.
            thinking: Whether the fit runs in thinking mode.
            fit_with_cache: Whether the fit keeps a KV cache.

        Returns:
            `ApiMode.SYNC` for one request, `ApiMode.ASYNC` for a polled job.
        """
        api_mode = ApiMode(api_mode)
        if api_mode != ApiMode.AUTO:
            return api_mode
        if fit_with_cache:
            return ApiMode.SYNC
        api_settings = cls.get_settings()
        if api_settings is None:
            # Unreachable, or a server that predates predictions as background
            # jobs, whose settings lack the thresholds and so don't validate.
            # One request works on every server.
            logger.debug("No server settings; predicting with one request.")
            return ApiMode.SYNC
        async_settings = api_settings.async_settings
        thresholds = (
            async_settings.thinking_predict_use_above
            if thinking
            else async_settings.predict_use_above
        )
        # `>=`: the server runs a prediction as a job once it reaches either
        # count. The fit rule compares bytes with `>` instead.
        reaches_thresholds = (
            n_train_rows is not None and n_train_rows >= thresholds.train_rows
        ) or n_test_rows >= thresholds.test_rows
        return ApiMode.ASYNC if reaches_thresholds else ApiMode.SYNC

    @classmethod
    def authorize(cls, access_token: str):
        """Store ``access_token`` and send it as bearer token on all requests."""
        get_opts().TABPFN_TOKEN = access_token
        cls._access_token = access_token
        cls.httpx_client.headers.update(
            {"Authorization": f"Bearer {cls.get_access_token()}"}
        )

    @classmethod
    def reset_authorization(cls):
        """Clear the stored access token and the Authorization header."""
        cls._access_token = None
        cls.httpx_client.headers.pop("Authorization", None)

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        # start: fit request
        task_config: FitTaskConfig,
        tabpfn_systems: list[TabPFNSystem],
        thinking_config: ThinkingConfig | None = None,
        # end: fit request
        api_mode: ApiMode = ApiMode.AUTO,
        client_options: ClientOptions | None = None,
        description: str | None = None,
        text_handling: TextHandling = "advanced",
    ) -> UUID:
        """Same as ``fit_with_result``, returning only the fitted train set ID."""
        return cls.fit_with_result(
            X,
            y,
            task_config=task_config,
            tabpfn_systems=tabpfn_systems,
            text_handling=text_handling,
            thinking_config=thinking_config,
            api_mode=api_mode,
            client_options=client_options,
            description=description,
        ).fitted_train_set_id

    @classmethod
    def fit_with_result(
        cls,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        # start: fit request
        task_config: FitTaskConfig,
        tabpfn_systems: list[TabPFNSystem],
        thinking_config: ThinkingConfig | None = None,
        # end: fit request
        api_mode: ApiMode = ApiMode.AUTO,
        client_options: ClientOptions | None = None,
        description: str | None = None,
        text_handling: TextHandling = "advanced",
        wait: bool = True,
    ) -> FitResponse:
        """Upload a train set to server and fit it.

        Args:
            X: The training input samples, of shape (n_samples, n_features).
            y: The target values, of shape (n_samples,) or (n_samples, n_outputs).
            task_config: The configuration for the fit task.
            tabpfn_systems: The systems to use for the fit method.
            thinking_config: The configuration for the thinking mode.
            api_mode: The API mode to use for the fit method: SYNC, ASYNC, or AUTO.
            client_options: Client specific options (e.g. timeout, headers).
            description: Description of the dataset and task for the server.
            text_handling: Text-processing preset, ``"advanced"`` or ``"simple"``.
            wait: Whether to poll the server until the fit completes. When
                False, return as soon as the server answers the fit request,
                which in ASYNC mode is right after it accepted the job; the
                result's ``status`` then says whether the fit is still running,
                and ``_wait_for_fit`` waits for it.

        Returns:
            The fit as the server reports it: the unique ID of the fitted train
            set, its status, COMPLETED once waited for, and its timings (None
            if the server reports none). An ASYNC fit not waited for is
            PENDING, without timings.

        Raises:
            ValueError: If ``text_handling`` is invalid or the in-memory size of
                ``X`` or ``y`` exceeds the server limit.
            FitFailedError: If the server reports that a fit waited for failed.
            TimeoutError: If a fit waited for has not completed within the poll
                timeout.
        """
        validate_text_handling(text_handling)
        api_mode = ApiMode(api_mode)
        client_options = client_options or ClientOptions()
        tabpfn_systems_: list[TabPFNSystem | str] = list(tabpfn_systems)

        df_X = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        df_y = cast(pd.DataFrame, y if isinstance(y, pd.DataFrame) else pd.DataFrame(y))

        total_mem_usage = 0
        api_settings = cls.get_settings()

        for name, df in [("x_train", df_X), ("y_train", df_y)]:
            mem_usage = df.memory_usage(deep=True).sum()
            total_mem_usage += mem_usage
            if (
                api_settings is not None
                and mem_usage > api_settings.dataset_max_size_bytes
            ):
                raise ValueError(
                    f"In-memory size of {name} ({mem_usage} bytes) exceeds "
                    f"the server limit of {api_settings.dataset_max_size_bytes} bytes."
                )

        match api_mode:
            case ApiMode.SYNC:
                use_async = False
            case ApiMode.ASYNC:
                use_async = True
            case ApiMode.AUTO:
                trainset_size_threshold = (
                    cls._resolve_async_settings().use_above_trainset_size_bytes
                )
                use_async = total_mem_usage > trainset_size_threshold

        x_bytes, x_crc32c_hash = _serialize_to_parquet(df_X)
        y_bytes, y_crc32c_hash = _serialize_to_parquet(df_y)

        if get_opts().TABPFN_CLIENT_DEDUP_DATASETS:
            x_dedup_hash = x_crc32c_hash
            y_dedup_hash = y_crc32c_hash
        else:
            x_dedup_hash = None
            y_dedup_hash = None

        prepare_req = PrepareTrainSetUploadRequest(
            x_train_info=FileInfo(
                format="parquet", hash=x_dedup_hash, size_bytes=len(x_bytes)
            ),
            y_train_info=FileInfo(
                format="parquet", hash=y_dedup_hash, size_bytes=len(y_bytes)
            ),
            description=description,
            force_reupload=get_opts().TABPFN_CLIENT_FORCE_REUPLOAD,
        )
        # NOTE: no retries here other than client built-in ones
        res = cls.httpx_client.post(
            url="/tabpfn/prepare_train_set_upload",
            json=prepare_req.model_dump(mode="json", exclude_none=True),
            headers=client_options.headers,
        )
        prepare_resp = cls._validate_response(
            res,
            "prepare_train_set_upload",
            success_model=PrepareTrainSetUploadResponse,
            error_models={409: DuplicateTrainSetErrorResponse},
        )

        if isinstance(prepare_resp, DuplicateTrainSetErrorResponse):
            logger.debug("Train set already exists, skipping upload.")
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        cls._upload_to_gcs,
                        "x_train",
                        x_bytes,
                        prepare_resp.x_train_info,
                    ),
                    pool.submit(
                        cls._upload_to_gcs,
                        "y_train",
                        y_bytes,
                        prepare_resp.y_train_info,
                    ),
                ]
                try:
                    for future in as_completed(futures):
                        future.result()
                except Exception:
                    for f in futures:
                        f.cancel()
                    raise

        if use_async:
            submit_resp = cls._submit_fit_job(
                req=SubmitFitJobRequest(
                    train_set_upload_id=prepare_resp.train_set_upload_id,
                    task_config=task_config,
                    tabpfn_systems=tabpfn_systems_,
                    text_handling=None
                    if text_handling == "advanced"
                    else text_handling,
                    thinking_config=thinking_config,
                ),
                timeout=client_options.timeout,
                headers=client_options.headers,
            )
            # The server answers a submitted job with its id alone: it has
            # accepted the fit, not run it.
            fit_resp = FitResponse(
                fitted_train_set_id=submit_resp.fitted_train_set_id,
                status=FitStatus.PENDING,
            )
        else:
            fit_resp = cls._fit(
                req=FitRequest(
                    train_set_upload_id=prepare_resp.train_set_upload_id,
                    task_config=task_config,
                    tabpfn_systems=tabpfn_systems_,
                    text_handling=None
                    if text_handling == "advanced"
                    else text_handling,
                    thinking_config=thinking_config,
                ),
                timeout=client_options.timeout,
                headers=client_options.headers,
            )

        # NOTE: The server currently only returns COMPLETED or PENDING
        # (failed requests raise), but `status` is forward-compatible
        # (`FitStatus | UnknownEnum`), so treat anything that is not
        # COMPLETED as still in progress instead of silently skipping
        # the wait when a newer server adds a status.
        if wait and fit_resp.status != FitStatus.COMPLETED:
            status_resp = cls._wait_for_fit(
                fit_resp.fitted_train_set_id,
                headers=client_options.headers,
            )
            fit_resp = FitResponse(
                fitted_train_set_id=fit_resp.fitted_train_set_id,
                status=status_resp.status,
                timings=status_resp.timings,
            )
        return fit_resp

    @classmethod
    def _wait_for_fit(
        cls,
        fitted_train_set_id: UUID,
        headers: dict[str, str] | None = None,
    ) -> GetFitStatusResponse:
        """Poll ``/tabpfn/get_fit_status`` until the fit reaches a terminal state.

        Polls as `_poll_until_done` describes, at once and then at the pace the
        server hints, which it derives from how long the fit has been pending,
        so a process resuming the wait for a fit another process submitted
        polls at the same cadence. A failed fit is reported by the server as
        HTTP 200 with ``status=failed`` (only an unknown/foreign id is a 404),
        so the status field — not the HTTP code — is what we inspect here.

        Network errors, cut-off responses and 408, 502, 503 and 504 are
        transient, logged and retried until the deadline; 429, 500, an empty
        body and a malformed one are retried up to 3 times in a row.

        Raises:
            FitFailedError: If the fit FAILED.
            FittedModelNotFoundError: If the server has no fit with this id.
            TimeoutError: If the fit reaches no terminal state within the
                resolved poll timeout.
            RuntimeError: On the 4th capped poll error in a row.
        """

        def is_done(status_resp: GetFitStatusResponse) -> bool:
            if status_resp.status == FitStatus.FAILED:
                raise FitFailedError(
                    f"Fit {fitted_train_set_id} failed: "
                    f"{status_resp.error or 'no error message provided'}"
                )
            return status_resp.status == FitStatus.COMPLETED

        req = GetFitStatusRequest(fitted_train_set_id=fitted_train_set_id)
        return cls._poll_until_done(
            lambda: cls._get_fit_status(
                req=req, timeout=_FIT_STATUS_POLL_TIMEOUT_SECS, headers=headers
            ),
            is_done,
            subject=f"fit {fitted_train_set_id}",
            transient_errors=(
                httpx.NetworkError,
                httpx.TimeoutException,
                httpx.RemoteProtocolError,
                RetryableServerError,
            ),
            capped_errors=(CappedRetryableServerError,),
        )

    @classmethod
    def _poll_until_done(
        cls,
        fetch: Callable[[], _PollResponseT],
        is_done: Callable[[_PollResponseT], bool],
        *,
        subject: str,
        transient_errors: tuple[type[Exception], ...],
        capped_errors: tuple[type[Exception], ...],
        first_poll_delay_secs: float = 0.0,
    ) -> _PollResponseT:
        """Poll until a response ends the wait, and return that response.

        The first poll happens after ``first_poll_delay_secs``, at once by
        default, and no later than the deadline; between polls we sleep for
        the server's per-response ``retry_in_secs`` hint (5s until the first
        hint arrives, and at least 5s after a failed poll), clamped to the
        ``_MIN_RETRY_INTERVAL_SECS`` floor so a zero/negative hint can neither
        busy-poll the endpoint nor crash ``time.sleep``.

        The resolved poll timeout, counted from the start of this call, is a
        hard bound: the loop gives up as soon as the next poll could no longer
        start before the deadline, instead of sleeping up to the deadline and
        polling once more past it. A single poll can still run past the
        deadline, for as long as its own request timeout allows.

        Args:
            fetch: Polls once.
            is_done: Whether a response ends the wait. It raises for one that
                ends the wait in failure.
            subject: What is polled, such as ``"fit <id>"``, for messages.
            transient_errors: Poll errors that are logged and retried until
                the deadline.
            capped_errors: Poll errors that are logged and retried up to 3
                times in a row.
            first_poll_delay_secs: Seconds to wait before the first poll.

        Returns:
            The response that ended the wait.

        Raises:
            TimeoutError: If no response ended the wait within the poll
                timeout. When polls were failing at the end, it chains the
                last poll error so the cause is not lost.
            RuntimeError: On the 4th capped poll error in a row, which it
                chains.
        """
        total_timeout = cls._resolve_async_settings().poll_timeout_secs
        deadline = time.monotonic() + total_timeout
        fallback_retry_in_secs = 5.0
        retry_in_secs = fallback_retry_in_secs  # until the server sends a hint
        # Most recent poll error; reset on every successful poll so the
        # TimeoutError only chains it if polls were failing at the end.
        last_error: Exception | None = None
        max_consecutive_capped_errors = 3
        consecutive_capped_errors = 0

        if first_poll_delay_secs > 0:
            # Clamped to the deadline, and the poll still goes out after it: a
            # poll timeout shorter than the hint still gets one answer.
            time.sleep(
                min(first_poll_delay_secs, max(0.0, deadline - time.monotonic()))
            )

        while True:
            try:
                response = fetch()
            except transient_errors as exc:
                last_error = exc
                consecutive_capped_errors = 0
                logger.warning(
                    f"Transient error while polling status of {subject}, "
                    f"retrying until the poll deadline: {exc}"
                )
            except capped_errors as exc:
                consecutive_capped_errors += 1
                if consecutive_capped_errors > max_consecutive_capped_errors:
                    # As a RuntimeError, like every other error that ends a
                    # poll, so that `except RuntimeError` catches it.
                    raise RuntimeError(
                        f"Polling {subject} failed {consecutive_capped_errors} "
                        f"times in a row: {exc}"
                    ) from exc
                last_error = exc
                logger.warning(
                    f"Unexpected error while polling status of {subject}, "
                    f"retrying up to {max_consecutive_capped_errors} "
                    f"consecutive errors: {exc}"
                )
            else:
                last_error = None
                consecutive_capped_errors = 0
                if is_done(response):
                    return response
                if response.retry_in_secs is not None:
                    retry_in_secs = max(
                        response.retry_in_secs, _MIN_RETRY_INTERVAL_SECS
                    )

            # After a failed poll, wait at least the fallback: the short hint of
            # a young job would otherwise poll, and warn, twice a second
            # through an outage. The hint applies again after the next success.
            sleep_secs = (
                retry_in_secs
                if last_error is None
                else max(retry_in_secs, fallback_retry_in_secs)
            )
            # Still running (or a poll error): sleep, then poll again — unless
            # the next poll could only start at/after the deadline.
            remaining = deadline - time.monotonic()
            if remaining <= sleep_secs:
                message = (
                    f"{subject[:1].upper()}{subject[1:]} did not reach a "
                    "terminal state within the poll timeout of "
                    f"{total_timeout} seconds."
                )
                if last_error is not None:
                    message += f" The last status poll failed: {last_error}"
                raise TimeoutError(message) from last_error
            time.sleep(sleep_secs)

    # NOTE: no backoff decorator here on purpose — the `_wait_for_fit` polling
    # loop already retries transient errors until its deadline, and a nested
    # retry layer would skew the poll interval accounting.
    @classmethod
    def _get_fit_status(
        cls,
        req: GetFitStatusRequest,
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> GetFitStatusResponse:
        """Fetch the status of a fit once.

        Raises:
            FittedModelNotFoundError: If the server has no fit with this ID.
            CappedRetryableServerError: On HTTP 429 or 500, or an empty or
                malformed success body (treated as a 500).
        """
        res = cls.httpx_client.post(
            url="/tabpfn/get_fit_status",
            json=req.model_dump(mode="json", exclude_none=True),
            timeout=timeout,
            headers=headers,
        )
        # We don't retry 429s and 500s (see _raise_http_error() when validating
        # the response). In this case because the error is decoupled from the
        # underlying job status, we allow a few retries up to a max of 3 consecutive
        # capped retryable errors.
        if res.status_code in (429, 500):
            raise CappedRetryableServerError(
                f"Fail to call get_fit_status: [HTTP {res.status_code}] {res.text}."
            )
        try:
            status_resp = cls._validate_response(
                res,
                "get_fit_status",
                success_model=GetFitStatusResponse,
                error_models={404: NotFoundErrorResponse},
            )
        except EmptyResponseError as exc:
            raise CappedRetryableServerError(str(exc)) from exc
        except ValidationError as exc:
            raise _malformed_poll_body("get_fit_status", exc) from exc
        if isinstance(status_resp, NotFoundErrorResponse):
            raise _fitted_model_not_found(req.fitted_train_set_id, status_resp)
        return status_resp

    @classmethod
    # Every fit call creates a new fit job, no dedup logic exists on the server.
    # We want to be careful with retries here, we then just retry on
    # errors that we can be confident prevented a job to be enqueued.
    @backoff.on_exception(
        backoff.constant,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
        ),
        max_tries=2,
        interval=0,
        logger=logger,
        on_backoff=_on_backoff,
        on_giveup=_on_giveup,
    )
    def _submit_fit_job(
        cls,
        req: SubmitFitJobRequest,
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> SubmitFitJobResponse:
        """Submit an asynchronous fit job for an uploaded train set."""
        res = cls.httpx_client.post(
            url="/tabpfn/submit_fit_job",
            json=req.model_dump(mode="json", exclude_none=True),
            timeout=timeout,
            headers=headers,
        )
        return cls._validate_response(
            res,
            "submit_fit_job",
            success_model=SubmitFitJobResponse,
        )

    @classmethod
    # Every fit call creates a new fit job, and the server has no dedup logic,
    # so retry only errors that prove the request never reached it: a retry
    # after a lost response or a 503 could fit, and bill, twice.
    @backoff.on_exception(
        backoff.constant,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
        ),
        max_tries=2,
        interval=0,
        logger=logger,
        on_backoff=_on_backoff,
        on_giveup=_on_giveup,
    )
    def _fit(
        cls,
        req: FitRequest,
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> FitResponse:
        """Fit an uploaded train set synchronously."""
        res = cls.httpx_client.post(
            url="/tabpfn/fit",
            json=req.model_dump(mode="json", exclude_none=True),
            timeout=timeout,
            headers=headers,
        )
        return cls._validate_response(
            res,
            "fit",
            success_model=FitResponse,
        )

    @classmethod
    def predict(
        cls,
        fitted_train_set_id: UUID,
        x_test: pd.DataFrame | np.ndarray,
        task_config: ClassifierConfig | RegressorConfig,
        client_options: ClientOptions | None = None,
        *,
        api_mode: ApiMode = ApiMode.SYNC,
        n_train_rows: int | None = None,
        thinking: bool = False,
    ) -> PredictionResult:
        """Upload a test set and predict it with a fitted train set.

        SYNC predicts with one request. When the server stops waiting for the
        prediction before it completes, it answers that the prediction is still
        running, and the client polls for the result. ASYNC runs the prediction
        as a background job the client polls from the start, or with one
        request on a server that does not run predictions as background jobs.
        AUTO picks ASYNC once the fit's train rows or the test rows reach the
        server's thresholds, and SYNC when the server's settings are
        unavailable.

        Args:
            fitted_train_set_id: The unique ID of the fitted train set in the server.
            x_test: The test input, of shape (n_samples, n_features).
            task_config: The classification or regression task configuration.
            client_options: Client specific options (e.g. timeout, headers).
            api_mode: The API mode to use for the prediction: SYNC, ASYNC, or AUTO.
            n_train_rows: The fit's train rows, for AUTO; None if unknown.
            thinking: Whether the fit runs in thinking mode, for AUTO.

        Returns:
            The result from the predict API call containing the prediction,
            metadata, server-reported timings and the prediction's id.

        Raises:
            ValueError: If the in-memory size of ``x_test`` exceeds the server
                limit, or ``api_mode`` is not a valid mode.
            FittedModelNotFoundError: If the fitted train set is not found.
            PredictFailedError: If the server reports that the prediction
                failed, in answer to the prediction request or to a poll for it.
            RefitRequiredError: If the fitted model can no longer predict and
                must be fitted again.
            TimeoutError: If a prediction the client polls for has not completed
                within the poll timeout.
        """
        client_options = client_options or ClientOptions()
        started = cls._start_predict(
            fitted_train_set_id,
            x_test,
            task_config,
            client_options,
            api_mode=api_mode,
            n_train_rows=n_train_rows,
            thinking=thinking,
        )
        if isinstance(started, UUID):
            return cls._await_prediction(started, client_options)
        return started

    @classmethod
    def predict_chunks(
        cls,
        fitted_train_set_id: UUID,
        x_chunks: Iterable[pd.DataFrame | np.ndarray],
        task_config: ClassifierConfig | RegressorConfig,
        client_options: ClientOptions | None = None,
        *,
        api_mode: ApiMode = ApiMode.SYNC,
        n_train_rows: int | None = None,
        thinking: bool = False,
    ) -> list[PredictionResult]:
        """Predict several test sets with one fitted train set, as `predict` does each.

        Each test set is predicted as `predict` would predict it. Those that run
        as background jobs overlap: up to 4 run at once, and with 4 running the
        client waits for the oldest before it starts the next. Those predicted
        with one request run one at a time. When a test set fails, or the wait
        breaks off, the client starts no more and logs the id of every job
        still running, which the server completes and bills regardless.

        Args:
            fitted_train_set_id: The unique ID of the fitted train set in the server.
            x_chunks: The test sets, each of shape (n_samples, n_features). An
                iterator is consumed one test set at a time.
            task_config: The classification or regression task configuration.
            client_options: Client specific options (e.g. timeout, headers).
            api_mode: The API mode for each test set: SYNC, ASYNC, or AUTO.
            n_train_rows: The fit's train rows, for AUTO; None if unknown.
            thinking: Whether the fit runs in thinking mode, for AUTO.

        Returns:
            One result per test set, in the order of ``x_chunks``.

        Raises:
            ValueError: If the in-memory size of a test set exceeds the server
                limit, or ``api_mode`` is not a valid mode.
            FittedModelNotFoundError: If the fitted train set is not found.
            PredictFailedError: If the server reports that a prediction
                failed, in answer to its request or to a poll for it.
            RefitRequiredError: If the fitted model can no longer predict and
                must be fitted again.
            TimeoutError: If a prediction the client polls for has not completed
                within the poll timeout.
        """
        api_mode = ApiMode(api_mode)
        client_options = client_options or ClientOptions()
        results: dict[int, PredictionResult] = {}
        # The jobs started and not waited for yet, oldest first, with the
        # position of their test set.
        in_flight: deque[tuple[int, UUID]] = deque()
        try:
            for index, x_chunk in enumerate(x_chunks):
                if len(in_flight) == _MAX_PREDICT_JOBS_IN_FLIGHT:
                    oldest, fitted_test_set_id = in_flight.popleft()
                    results[oldest] = cls._await_prediction(
                        fitted_test_set_id, client_options
                    )
                started = cls._start_predict(
                    fitted_train_set_id,
                    x_chunk,
                    task_config,
                    client_options,
                    api_mode=api_mode,
                    n_train_rows=n_train_rows,
                    thinking=thinking,
                )
                if isinstance(started, UUID):
                    in_flight.append((index, started))
                else:
                    results[index] = started
            while in_flight:
                oldest, fitted_test_set_id = in_flight.popleft()
                results[oldest] = cls._await_prediction(
                    fitted_test_set_id, client_options
                )
        except BaseException:
            # `_await_prediction` logs the one it was waiting for itself.
            for _, fitted_test_set_id in in_flight:
                _warn_stopped_waiting(fitted_test_set_id)
            raise
        return [results[index] for index in sorted(results)]

    @classmethod
    def _start_predict(
        cls,
        fitted_train_set_id: UUID,
        x_test: pd.DataFrame | np.ndarray,
        task_config: ClassifierConfig | RegressorConfig,
        client_options: ClientOptions,
        *,
        api_mode: ApiMode,
        n_train_rows: int | None,
        thinking: bool,
    ) -> UUID | PredictionResult:
        """Upload a test set, and predict it with one request or start it as a job.

        Returns:
            The prediction, if predicted with one request, or else the id of
            the background job to wait for.
        """
        api_mode = ApiMode(api_mode)
        test_set_upload_id = cls._upload_test_set(
            fitted_train_set_id, x_test, client_options
        )
        tabpfn_config = task_config.tabpfn_config
        predict_mode = cls._resolve_predict_mode(
            api_mode,
            n_train_rows=n_train_rows,
            n_test_rows=len(x_test),
            thinking=thinking,
            fit_with_cache=tabpfn_config is not None
            and tabpfn_config.fit_mode == FitMode.FIT_WITH_CACHE,
        )

        if predict_mode == ApiMode.ASYNC:
            submit_resp = cls._submit_predict_job(
                req=SubmitPredictJobRequest(
                    test_set_upload_id=test_set_upload_id,
                    fitted_train_set_id=fitted_train_set_id,
                    task_config=task_config,
                ),
                timeout=client_options.timeout,
                headers=client_options.headers,
            )
            if submit_resp is not None:
                return submit_resp.fitted_test_set_id
            # The cached settings promised background jobs that this server
            # (rolled back, say) does not run. It created no job, so one request
            # with the same upload predicts, and bills, once.
            logger.warning(
                "This TabPFN server does not run predictions as background jobs; "
                "predicting with one request."
            )
            cls._drop_settings()

        predict_resp = cls._predict(
            req=PredictRequest(
                test_set_upload_id=test_set_upload_id,
                fitted_train_set_id=fitted_train_set_id,
                task_config=task_config,
            ),
            timeout=client_options.timeout,
            headers=client_options.headers,
        )

        match predict_resp:
            case PredictResponse():
                return cls._prediction_result(
                    predict_resp.prediction,
                    predict_resp.metadata,
                    predict_resp.timings,
                    predict_resp.fitted_test_set_id,
                )
            case PredictResponseWithDownloadURI():
                prediction = cls._download_prediction(
                    predict_resp.prediction_uri,
                    timeout=client_options.timeout,
                )
                return cls._prediction_result(
                    prediction,
                    predict_resp.metadata,
                    predict_resp.timings,
                    predict_resp.fitted_test_set_id,
                )
            case PredictPendingResponse():
                logger.info(
                    f"Prediction {predict_resp.fitted_test_set_id} is still "
                    "running on the server; polling for its result."
                )
                return cls._await_prediction(
                    predict_resp.fitted_test_set_id,
                    client_options,
                    first_poll_delay_secs=predict_resp.retry_in_secs,
                )
            case _:
                assert_never(predict_resp)

    @classmethod
    def submit_predict(
        cls,
        fitted_train_set_id: UUID,
        x_test: pd.DataFrame | np.ndarray,
        task_config: ClassifierConfig | RegressorConfig,
        client_options: ClientOptions | None = None,
    ) -> UUID:
        """Upload a test set and start predicting it, without waiting for it.

        The server runs the prediction as a background job; `wait_for_predict`
        waits for it and returns it.

        Args:
            fitted_train_set_id: The unique ID of the fitted train set in the server.
            x_test: The test input, of shape (n_samples, n_features).
            task_config: The classification or regression task configuration.
            client_options: Client specific options (e.g. timeout, headers).

        Returns:
            The id of the prediction, which `wait_for_predict` takes.

        Raises:
            RuntimeError: If the server does not run predictions as background
                jobs.
            ValueError: If the in-memory size of ``x_test`` exceeds the server limit.
            FittedModelNotFoundError: If the fitted train set is not found.
            PredictFailedError: If the server reports that the prediction
                failed at once, for example because it could not queue it.
        """
        client_options = client_options or ClientOptions()
        test_set_upload_id = cls._upload_test_set(
            fitted_train_set_id, x_test, client_options
        )
        submit_resp = cls._submit_predict_job(
            req=SubmitPredictJobRequest(
                test_set_upload_id=test_set_upload_id,
                fitted_train_set_id=fitted_train_set_id,
                task_config=task_config,
            ),
            timeout=client_options.timeout,
            headers=client_options.headers,
        )
        if submit_resp is None:
            # The cached settings describe a server that ran background jobs.
            cls._drop_settings()
            raise RuntimeError(_NO_PREDICT_JOBS)
        logger.debug(f"Submitted prediction {submit_resp.fitted_test_set_id}.")
        return submit_resp.fitted_test_set_id

    @classmethod
    def wait_for_predict(
        cls,
        fitted_test_set_id: UUID | str,
        client_options: ClientOptions | None = None,
    ) -> PredictionResult:
        """Wait for a prediction the server accepted, and return it.

        Polls the server until the prediction has completed. The server keeps a
        prediction for about a day, so this also returns the result of a
        prediction whose wait broke off, given its id.

        Args:
            fitted_test_set_id: The id of the prediction, as a UUID or a string.
            client_options: Client specific options (e.g. timeout, headers).

        Returns:
            The prediction, as `predict` returns it.

        Raises:
            ValueError: If `fitted_test_set_id` is not a valid id.
            PredictFailedError: If the server reports that the prediction failed.
            RefitRequiredError: If the fitted model can no longer predict and
                must be fitted again.
            TimeoutError: If the prediction has not completed within
                `TABPFN_CLIENT_ASYNC_POLL_TIMEOUT` seconds.
            RuntimeError: If the server does not have the prediction, for
                instance because its result is no longer stored, or polling
                for it fails.
        """
        # Before any request, so that a malformed id fails at once, and the
        # result carries a UUID whatever the caller passed.
        fitted_test_set_id = UUID(str(fitted_test_set_id))
        return cls._await_prediction(
            fitted_test_set_id, client_options or ClientOptions()
        )

    @classmethod
    def _upload_test_set(
        cls,
        fitted_train_set_id: UUID,
        x_test: pd.DataFrame | np.ndarray,
        client_options: ClientOptions,
    ) -> UUID:
        """Upload a test set to predict with a fitted train set.

        Returns:
            The id of the uploaded test set, or of the same test set the server
            already holds.

        Raises:
            ValueError: If the in-memory size of ``x_test`` exceeds the server limit.
            FittedModelNotFoundError: If the fitted train set is not found.
        """
        df_x_test = x_test if isinstance(x_test, pd.DataFrame) else pd.DataFrame(x_test)

        api_settings = cls.get_settings()
        if api_settings is not None:
            mem_usage = df_x_test.memory_usage(deep=True).sum()
            if mem_usage > api_settings.dataset_max_size_bytes:
                raise ValueError(
                    f"In-memory size of x_test ({mem_usage} bytes) exceeds "
                    f"the server limit of {api_settings.dataset_max_size_bytes} bytes."
                )

        x_test_bytes, x_test_crc32c_hash = _serialize_to_parquet(df_x_test)

        if get_opts().TABPFN_CLIENT_DEDUP_DATASETS:
            x_test_dedup_hash = x_test_crc32c_hash
        else:
            x_test_dedup_hash = None

        prepare_req = PrepareTestSetUploadRequest(
            fitted_train_set_id=fitted_train_set_id,
            x_test_info=FileInfo(
                format="parquet",
                hash=x_test_dedup_hash,
                size_bytes=len(x_test_bytes),
            ),
            force_reupload=get_opts().TABPFN_CLIENT_FORCE_REUPLOAD,
        )
        # NOTE: no retries here other than client built-in ones
        res = cls.httpx_client.post(
            url="/tabpfn/prepare_test_set_upload",
            json=prepare_req.model_dump(mode="json", exclude_none=True),
            headers=client_options.headers,
        )
        prepare_resp = cls._validate_response(
            res,
            "prepare_test_set_upload",
            success_model=PrepareTestSetUploadResponse,
            error_models={
                404: NotFoundErrorResponse,
                409: DuplicateTestSetErrorResponse,
            },
        )

        if isinstance(prepare_resp, NotFoundErrorResponse):
            raise _fitted_model_not_found(fitted_train_set_id, prepare_resp)

        if isinstance(prepare_resp, DuplicateTestSetErrorResponse):
            logger.debug("Test set already exists, skipping upload.")
        else:
            cls._upload_to_gcs(
                "x_test",
                x_test_bytes,
                prepare_resp.x_test_info,
            )
        return prepare_resp.test_set_upload_id

    @staticmethod
    def _prediction_result(
        prediction: Prediction,
        metadata: Metadata,
        timings: PredictTimings | None,
        fitted_test_set_id: UUID | None,
    ) -> PredictionResult:
        """Convert a prediction as the server sends it into a `PredictionResult`."""
        if isinstance(prediction, dict):
            result = {}
            for k, v in prediction.items():
                if isinstance(v, list):  # type: ignore
                    dtype = float if _contains_none(v) else None
                    result[k] = np.array(v, dtype=dtype)
                else:
                    # The value should always be a list or list-of-lists,
                    # leaving this here as an extra precaution and future proofing.
                    result[k] = v
        else:
            result = np.array(prediction)

        return PredictionResult(
            y_pred=result,
            metadata=metadata.model_dump(mode="json", exclude_none=True),
            timings=timings.model_dump(mode="json") if timings is not None else None,
            fitted_test_set_id=fitted_test_set_id,
        )

    @classmethod
    def _await_prediction(
        cls,
        fitted_test_set_id: UUID,
        client_options: ClientOptions,
        *,
        first_poll_delay_secs: float = 0.0,
    ) -> PredictionResult:
        """Wait for a prediction the server accepted, and return it.

        A wait that breaks off for any other reason than a prediction that
        failed or that the server does not have (a timeout, an interrupt, an
        error) logs the prediction's id first: the prediction may run on, and
        `wait_for_predict` can still return it.

        Raises:
            PredictFailedError: If the server reports that the prediction
                failed, or that it completed but sends no prediction.
            PredictionGoneError: If the server does not have the prediction.
            TimeoutError: If the prediction has not completed within the poll
                timeout.
        """
        # A None timeout would leave a poll on a dead connection hanging
        # forever, past any deadline.
        timeout = cast("float | None", client_options.timeout)
        if timeout is None:
            timeout = get_opts().TABPFN_CLIENT_TIMEOUT
        try:
            result_resp = cls._wait_for_predict(
                fitted_test_set_id,
                timeout=min(timeout, _MAX_POLL_TIMEOUT_SECS),
                headers=client_options.headers,
                first_poll_delay_secs=first_poll_delay_secs,
            )
        except (PredictFailedError, PredictionGoneError):
            # The prediction has ended, or the server no longer has it: there is
            # nothing to come back for.
            raise
        except BaseException:
            _warn_stopped_waiting(fitted_test_set_id)
            raise
        if result_resp.prediction is None or result_resp.metadata is None:
            # A failure like any other: a later poll would answer the same.
            raise PredictFailedError(
                f"The server reported prediction {fitted_test_set_id} as "
                "completed, but sent no prediction.",
                fitted_test_set_id=fitted_test_set_id,
            )
        return cls._prediction_result(
            result_resp.prediction,
            result_resp.metadata,
            result_resp.timings,
            fitted_test_set_id,
        )

    @classmethod
    def _wait_for_predict(
        cls,
        fitted_test_set_id: UUID,
        *,
        timeout: float,
        headers: dict[str, str] | None = None,
        first_poll_delay_secs: float = 0.0,
    ) -> GetPredictResultResponse:
        """Poll ``/tabpfn/get_predict_result`` until the prediction reaches a terminal state.

        Polls as `_poll_until_done` describes, the first time after
        ``first_poll_delay_secs``. Each poll also runs the job's upkeep on the
        server, such as failing it past its deadline; nothing else does that.
        A single poll can run past the deadline by up to ``timeout`` when its
        response stalls, as httpx applies ``timeout`` to each read rather than
        to the whole response, and for as long as the server keeps sending
        bytes. A failed prediction is reported by the server as HTTP 200 with
        ``status=failed``, so the status field — not the HTTP code — is what we
        inspect here.

        Returns once the prediction is COMPLETED, with the whole prediction:
        every completed poll streams it again, so the loop stops at the first
        one.

        Poll errors are classified by their HTTP status, or by the status a
        streamed error envelope reports. 408, 502, 503, 504 and network errors
        are transient, logged and retried until the deadline; 429, 500, an
        empty or malformed body and a cut-off response are retried up to 3
        times in a row; anything else, such as an unknown (404) or expired
        (410) prediction, raises at once.

        Raises:
            PredictFailedError: If the prediction FAILED;
                ``RefitRequiredError`` if the model must be fitted again.
            TimeoutError: If the prediction reaches no terminal state within
                the resolved poll timeout.
            RuntimeError: On the 4th capped poll error in a row.
        """

        def is_done(result_resp: GetPredictResultResponse) -> bool:
            if result_resp.status == PredictStatus.FAILED:
                raise _prediction_failed(
                    fitted_test_set_id, result_resp.error, result_resp.error_code
                )
            # Anything else, a status this client does not know included,
            # counts as still running.
            return result_resp.status == PredictStatus.COMPLETED

        req = GetPredictResultRequest(fitted_test_set_id=fitted_test_set_id)
        return cls._poll_until_done(
            lambda: cls._get_predict_result(req=req, timeout=timeout, headers=headers),
            is_done,
            subject=f"prediction {fitted_test_set_id}",
            transient_errors=(
                httpx.NetworkError,
                httpx.TimeoutException,
                RetryableServerError,
            ),
            # A response cut off mid-body is capped rather than transient: the
            # server cuts it when it fails after starting it, as on a late 404.
            capped_errors=(CappedRetryableServerError, httpx.RemoteProtocolError),
            first_poll_delay_secs=first_poll_delay_secs,
        )

    # NOTE: no backoff decorator here on purpose — the `_wait_for_predict`
    # polling loop already retries transient errors until its deadline, and a
    # nested retry layer would skew the poll interval accounting.
    @classmethod
    def _get_predict_result(
        cls,
        req: GetPredictResultRequest,
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> GetPredictResultResponse:
        """Fetch the status of a prediction once, with the prediction once completed.

        Raises:
            PredictionGoneError: On HTTP 404 with the server's ``NOT_FOUND``
                code for an unknown prediction, or 410 for one whose result is
                no longer stored.
            RuntimeError: If the server has no such endpoint, as it runs no
                predictions as background jobs, or on another status no retry
                can clear.
            CappedRetryableServerError: On HTTP 429 or 500, or an empty or
                malformed success body (treated as a 500).
            RetryableServerError: On HTTP 408, 502, 503, or 504.
        """
        res = cls.httpx_client.post(
            url="/tabpfn/get_predict_result",
            json=req.model_dump(mode="json", exclude_none=True),
            timeout=timeout,
            headers=headers,
        )
        # As for fit statuses, these errors are decoupled from the prediction
        # itself, so the poll loop allows a few in a row.
        if res.status_code in (429, 500):
            raise CappedRetryableServerError(
                f"Fail to call get_predict_result: [HTTP {res.status_code}] {res.text}."
            )
        if cls._is_route_missing(res):
            raise RuntimeError(
                f"Cannot wait for prediction {req.fitted_test_set_id}: this TabPFN "
                "server does not run predictions as background jobs right now. "
                "Wait for it again later: a server that runs them again may "
                "still complete it."
            )
        try:
            return cls._validate_response(
                res, "get_predict_result", success_model=GetPredictResultResponse
            )
        except EmptyResponseError as exc:
            raise CappedRetryableServerError(str(exc)) from exc
        except ValidationError as exc:
            raise _malformed_poll_body("get_predict_result", exc) from exc
        except StreamedServerError as exc:
            # An error the server hit after its keepalive pings started is
            # classified by its status, exactly as a fast one.
            if exc.status_code in (429, 500):
                raise CappedRetryableServerError(str(exc)) from exc
            if exc.status_code in (408, 502, 503, 504):
                raise RetryableServerError(str(exc)) from exc
            if exc.status_code in (404, 410):
                raise _prediction_gone(req.fitted_test_set_id, exc) from exc
            raise
        except RuntimeError as exc:
            # Wrapped rather than checked up front, so that `_validate_response`
            # still checks the client version. A 404 counts only with the
            # server's own code: one from a proxy says nothing about the
            # prediction.
            if res.status_code == 410 or (
                res.status_code == 404
                and cls._read_json_body(res)[0].get("error_code") == "NOT_FOUND"
            ):
                raise _prediction_gone(req.fitted_test_set_id, exc) from exc
            raise

    @classmethod
    # Every submit creates a new predict job, no dedup logic exists on the
    # server. We want to be careful with retries here, we then just retry on
    # errors that we can be confident prevented a job to be enqueued.
    @backoff.on_exception(
        backoff.constant,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
        ),
        max_tries=2,
        interval=0,
        logger=logger,
        on_backoff=_on_backoff,
        on_giveup=_on_giveup,
    )
    def _submit_predict_job(
        cls,
        req: SubmitPredictJobRequest,
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> SubmitPredictJobResponse | None:
        """Submit a predict job for an uploaded test set.

        Returns:
            The id of the job, or None if the server has no such endpoint: it
            runs no predictions as background jobs, and created no job.

        Raises:
            PredictFailedError: If the server created the job and it failed
                at once, as when the server cannot queue it.
        """
        res = cls.httpx_client.post(
            url="/tabpfn/submit_predict_job",
            json=req.model_dump(mode="json", exclude_none=True),
            timeout=timeout,
            headers=headers,
        )
        if cls._is_route_missing(res):
            return None
        resp = cls._validate_response(
            res,
            "submit_predict_job",
            success_model=SubmitPredictJobResponse,
            error_models=_PREDICT_FAILED_ERRORS,
        )
        if isinstance(resp, PredictFailedErrorResponse):
            # What a poll for the job raises.
            raise _prediction_failed(
                resp.fitted_test_set_id,
                resp.message,
                resp.error_code,
                trace_id=resp.trace_id,
            )
        return resp

    @classmethod
    # Every predict creates a new predict job, as a submit does, so the same
    # caution applies: even a 503 can come after the job completed, when the
    # server cannot read the stored result.
    @backoff.on_exception(
        backoff.constant,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
        ),
        max_tries=2,
        interval=0,
        logger=logger,
        on_backoff=_on_backoff,
        on_giveup=_on_giveup,
    )
    def _predict(
        cls,
        req: PredictRequest,
        timeout: float,
        headers: dict[str, str] | None = None,
    ) -> PredictResponseUnion:
        """Request a prediction for an uploaded test set.

        Raises:
            PredictFailedError: If the server reports that the prediction it
                created for the request failed.
            RefitRequiredError: If the fitted model can no longer predict and
                must be fitted again.
        """
        res = cls.httpx_client.post(
            url="/tabpfn/predict",
            json=req.model_dump(mode="json", exclude_none=True),
            timeout=timeout,
            headers=headers,
        )
        try:
            # The body, not the request, decides which model applies: a
            # prediction, a download URL to follow, a prediction the server is
            # still running, or one that failed.
            resp = cls._validate_response(
                res,
                "predict",
                success_model=_PredictResponseUnion,
                error_models=_PREDICT_FAILED_ERRORS,
            )
        # Both handlers recognize a refit on a server that predates
        # `PredictFailedErrorResponse`: every 409 asks for a new fit, and so
        # does a 400 with the refit text, a stale KV cache as the server reports
        # it from the failure it booked rather than from the worker's reply.
        except StreamedServerError as exc:
            if exc.status_code in (400, 409) and _requires_refit(
                cls._read_json_body(res)[0].get("error_code"),
                exc.detail,
                status_code=exc.status_code,
            ):
                raise RefitRequiredError(exc.detail) from exc
            raise
        except RuntimeError as exc:
            if res.status_code in (400, 409):
                body, message = cls._read_json_body(res)
                if _requires_refit(
                    body.get("error_code"), message, status_code=res.status_code
                ):
                    if trace_id := body.get("trace_id"):
                        message += f" Report trace ID: {trace_id}."
                    raise RefitRequiredError(message) from exc
            raise
        if isinstance(resp, PredictFailedErrorResponse):
            # What a poll for the prediction raises.
            raise _prediction_failed(
                resp.fitted_test_set_id,
                resp.message,
                resp.error_code,
                trace_id=resp.trace_id,
            )
        return resp

    @classmethod
    @backoff.on_exception(
        backoff.constant,
        (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.ReadTimeout,
            httpx.RemoteProtocolError,
            RetryableServerError,
        ),
        max_tries=2,
        interval=1,
        logger=logger,
        on_backoff=_on_backoff,
        on_giveup=_on_giveup,
    )
    def _download_prediction(cls, uri: str, timeout: float) -> Prediction:
        """Download a prediction from a signed URL."""
        # The signed URL is the credential here, so the API bearer token on
        # `httpx_client` has no business reaching the object store; use a bare
        # request instead.
        resp = httpx.get(uri, timeout=timeout)
        if resp.status_code in _RETRYABLE_STORAGE_STATUSES:
            raise RetryableServerError(
                f"Prediction download failed: {resp.status_code} {resp.text}"
            )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Prediction download permanently failed: {resp.status_code} {resp.text}"
            )
        try:
            return resp.json()
        except ValueError as e:
            raise RuntimeError(
                "Prediction download returned a body that is not JSON: "
                f"{resp.text[:200]!r}"
            ) from e

    @classmethod
    def _upload_to_gcs(cls, dataset: str, data: bytes, info: FileUploadInfo) -> None:
        """Upload ``data`` to the signed URLs in ``info``, one chunk per URL."""
        num_chunks = len(info.signed_urls)
        chunk_size = len(data) // num_chunks
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < num_chunks - 1 else len(data)
            chunks.append(data[start:end])

        if num_chunks == 1:
            cls._upload_single_chunk(
                url=info.signed_urls[0],
                chunk=chunks[0],
                headers=info.required_headers,
                dataset=dataset,
            )
            return

        with ThreadPoolExecutor(
            max_workers=min(get_opts().TABPFN_CLIENT_MAX_THREAD_PER_UPLOAD, num_chunks)
        ) as pool:
            futures = {
                pool.submit(
                    cls._upload_single_chunk,
                    url=info.signed_urls[i],
                    chunk=chunks[i],
                    headers=info.required_headers,
                    dataset=dataset,
                    chunk_index=i,
                ): i
                for i in range(num_chunks)
            }
            for future in as_completed(futures):
                future.result()

    @classmethod
    @backoff.on_exception(
        backoff.constant,
        (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.RemoteProtocolError,
            RetryableServerError,
        ),
        max_tries=2,
        interval=1,
        logger=logger,
        on_backoff=_on_backoff,
        on_giveup=_on_giveup,
    )
    def _upload_single_chunk(
        cls,
        url: str,
        chunk: bytes,
        headers: dict[str, str],
        dataset: str,
        chunk_index: int = 0,
    ) -> None:
        """Upload one chunk to a signed URL."""
        resp = cls.httpx_client.put(
            url,
            content=chunk,
            headers=headers,
            timeout=get_opts().TABPFN_CLIENT_UPLOAD_TIMEOUT,
        )
        if resp.status_code == 200:
            return
        if resp.status_code in _RETRYABLE_STORAGE_STATUSES:
            raise RetryableServerError(
                f"GCS upload failed for dataset {dataset} at chunk {chunk_index}: "
                f"{resp.status_code} {resp.text}"
            )
        raise RuntimeError(
            f"GCS upload permanently failed for dataset {dataset} at chunk {chunk_index}: "
            f"{resp.status_code} {resp.text}"
        )

    @staticmethod
    def _warn_if_deprecated(response: httpx.Response) -> None:
        """Emit a ``DeprecationWarning`` if the response marks the client deprecated."""
        if response.headers.get("deprecation", "").lower() != "true":
            return

        msg = "This version of tabpfn-client is deprecated and will stop working in a future release."

        sunset = response.headers.get("sunset")
        if sunset:
            msg += f" Support ends on: {sunset}."

        link_header = response.headers.get("link", "")
        if link_header and 'rel="deprecation"' in link_header:
            match = re.search(r"<([^>]+)>", link_header)
            if match:
                msg += f" Please upgrade: {match.group(1)}"

        warnings.warn(msg, DeprecationWarning, stacklevel=4)

    @staticmethod
    def _read_json_body(response: httpx.Response) -> tuple[dict[str, Any], str]:
        """Read the response body and a display message.

        Returns:
            The body (``{}`` unless it is a JSON object) and a display message
            drawn from it, falling back to the HTTP reason phrase or raw text.
        """
        body: dict[str, Any] = {}
        try:
            # Streaming responses must be read explicitly to prevent
            # httpx.ResponseNotRead errors.
            if not response.is_closed:
                response.read()
            parsed = response.json()
            if isinstance(parsed, dict):
                body = parsed
        except json.JSONDecodeError:
            pass
        message = (
            body.get("message")
            or body.get("detail")  # defensive, starlette standard error format
            or response.reason_phrase
            or response.text
        )
        return body, message

    @staticmethod
    def _check_version(response: httpx.Response) -> None:
        """Warn if the endpoint is deprecated.

        Raises:
            RuntimeError: If the server requires a newer client version (HTTP 426).
        """
        ServiceClient._warn_if_deprecated(response)
        if response.status_code == 426:
            _, message = ServiceClient._read_json_body(response)
            raise RuntimeError(message)

    @staticmethod
    def _raise_http_error(
        response: httpx.Response, method_name: str, body: dict[str, Any], message: str
    ) -> NoReturn:
        """Raise the error for a failed response.

        Raises:
            RetryableServerError: On HTTP 408, 502, 503, or 504.
            RuntimeError: On any other status.
        """
        error_message = (
            f"Fail to call {method_name}: [HTTP {response.status_code}] {message}."
        )
        if trace_id := body.get("trace_id"):
            error_message += f" Report trace ID: {trace_id}."

        if response.status_code in {408, 502, 503, 504}:
            raise RetryableServerError(error_message)

        raise RuntimeError(error_message)

    @staticmethod
    def _raise_on_error(response: httpx.Response, method_name: str) -> None:
        """Raise for 4xx/5xx responses; no-op otherwise.

        For callers that expect no payload. The body is only read on errors,
        so streaming success responses stay unread. Callers that expect a
        payload use ``_validate_response`` instead.
        """
        ServiceClient._check_version(response)
        if not (400 <= response.status_code < 600):
            return
        body, message = ServiceClient._read_json_body(response)
        ServiceClient._raise_http_error(response, method_name, body, message)

    @staticmethod
    def _is_route_missing(response: httpx.Response) -> bool:
        """Return whether the server has no endpoint at the requested path.

        This is how a server that predates predictions as background jobs (or
        was rolled back to such a version) answers their endpoints: with the
        bare "Not Found" of an unmatched route, whose error code is
        ``NOT_FOUND`` since tabpfn-server#1384 and ``USER_ERROR`` before. The
        endpoints' own 404s say what is not found, such as "Prediction not
        found", and a proxy's carry no JSON message.
        """
        if response.status_code != 404:
            return False
        body, _ = ServiceClient._read_json_body(response)
        return body.get("message") == "Not Found"

    @staticmethod
    def _validate_response(
        response: httpx.Response,
        method_name: str,
        success_model: type[SuccessT] | TypeAdapter[SuccessT],
        error_models: dict[int, type[ErrorT]] | None = None,
    ) -> SuccessT | ErrorT:
        """Parse a response into its success or expected error model.

        Args:
            response: The HTTP response.
            method_name: Name of the called method, used in error messages.
            success_model: Model or adapter the body is validated against on success.
            error_models: Models to parse error bodies into, keyed by status
                code: the HTTP status, or the status a streamed error reports.

        Returns:
            The validated success model, or the matching error model instance,
            whether the error came as an HTTP status or in a streamed error.

        Raises:
            StreamedServerError: On a streamed error without a matching error
                model.
            RuntimeError: On an error status without a matching error model.
            RetryableServerError: On HTTP 408, 502, 503, or 504 without a
                matching error model.
            EmptyResponseError: On a success status whose body is empty
                (keepalive whitespace only).
        """
        error_models = error_models or {}

        ServiceClient._check_version(response)

        status_code = response.status_code
        is_success = not (400 <= status_code < 600)

        body, message = ServiceClient._read_json_body(response)

        # Streaming errors, should come before the success check.
        if body.get("_streamed_error"):
            streamed_status = body.get("status_code")
            # The envelope holds the fields of the body the error has when it
            # comes before the stream starts, beside its own keys, which the
            # error models ignore.
            if isinstance(streamed_status, int) and (
                error_model := error_models.get(streamed_status)
            ):
                try:
                    return error_model.model_validate(body)
                except ValidationError:
                    pass
            raise StreamedServerError(
                f"Fail to call {method_name} with error: streamed, {message}",
                status_code=streamed_status
                if isinstance(streamed_status, int)
                else None,
                detail=message,
            )

        # Success with expected schema.
        if is_success:
            # Long-running endpoints stream whitespace keepalive pings before
            # the final payload. A success body with nothing else means the
            # stream ended without a result, e.g. the server died mid-request
            # without emitting a `_streamed_error` envelope.
            if not response.text.strip():
                raise EmptyResponseError(
                    f"Fail to call {method_name}: [HTTP {status_code}] the "
                    "server ended the response without a final payload."
                )
            if isinstance(success_model, TypeAdapter):
                return success_model.validate_python(body)
            return success_model.model_validate(body)

        # Errors with expected schema.
        if error_model := error_models.get(status_code):
            try:
                return error_model.model_validate(body)
            except ValidationError:
                pass

        # Errors without expected schema.
        ServiceClient._raise_http_error(response, method_name, body, message)

    @classmethod
    def try_connection(cls) -> bool:
        """Check if server is reachable and accepts the connection."""
        found_valid_connection = False
        try:
            response = cls.httpx_client.get(cls.server_endpoints.root.path)
            cls._check_version(response)
            if response.status_code == 200:
                found_valid_connection = True

        except httpx.ConnectError:
            logger.debug("Failed to connect to the server", exc_info=True)
            found_valid_connection = False

        return found_valid_connection

    @classmethod
    def is_auth_token_outdated(cls, access_token: str) -> bool | None:
        """Check if the provided access token is valid.

        Args:
            access_token: The access token to check.

        Returns:
            True if the token is valid, None if the user is not verified,
            False otherwise.
        """
        is_authenticated = False
        response = cls.httpx_client.get(
            cls.server_endpoints.protected_root.path,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        cls._check_version(response)
        if response.status_code == 200:
            is_authenticated = True
        elif response.status_code == 403:
            # 403 means user is not verified
            is_authenticated = None
        return is_authenticated

    @classmethod
    def validate_email(cls, email: str) -> tuple[bool, str]:
        """Check with the server that an email is valid and not already in use.

        Returns:
            Whether the email is valid, and the message returned from the server
            (empty if valid).
        """
        response = cls.httpx_client.post(
            cls.server_endpoints.validate_email.path, params={"email": email}
        )

        cls._check_version(response)
        if response.status_code == 200:
            is_valid = True
            message = ""
        else:
            is_valid = False
            message = response.json().get("message", "")

        return is_valid, message

    @classmethod
    def register(
        cls,
        email: str,
        password: str,
        password_confirm: str,
        validation_link: str,
        additional_info: dict,
    ):
        """Register a new user with the provided credentials.

        Args:
            email: The user's email.
            password: The password.
            password_confirm: The password, repeated.
            validation_link: Link sent to the user to verify the email.
            additional_info: Extra fields merged into the request body.

        Returns:
            Whether the user was created, the message returned from the server,
            and the access token (None if not created).
        """
        response = cls.httpx_client.post(
            cls.server_endpoints.register.path,
            json={
                "email": email,
                "password": password,
                "password_confirm": password_confirm,
                "validation_link": validation_link,
                **additional_info,
            },
        )

        cls._check_version(response)
        if response.status_code == 200:
            is_created = True
            message = response.json().get("message", "")
        else:
            is_created = False
            message = response.json().get("message", "")

        access_token = response.json()["token"] if is_created else None
        return is_created, message, access_token

    @classmethod
    def verify_email(cls, token: str, access_token: str) -> tuple[bool, str]:
        """Verify the email with the provided token.

        Returns:
            Whether the email was verified, and the message returned from the
            server.
        """
        response = cls.httpx_client.get(
            cls.server_endpoints.verify_email.path,
            params={"token": token, "access_token": access_token},
        )
        cls._check_version(response)
        if response.status_code == 200:
            is_verified = True
            message = response.json().get("message", "")
        else:
            is_verified = False
            message = response.json().get("message", "")

        return is_verified, message

    @classmethod
    def get_password_policy(cls) -> list[str]:
        """Get the password policy requirements from the server."""
        response = cls.httpx_client.get(
            cls.server_endpoints.password_policy.path,
        )
        cls._check_version(response)

        return response.json()["requirements"]

    @classmethod
    def send_verification_email(cls, access_token: str) -> tuple[bool, str]:
        """Let the server send an email for verifying the email.

        Returns:
            Whether the email was sent, and the message returned from the server.
        """
        response = cls.httpx_client.post(
            cls.server_endpoints.send_verification_email.path,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if response.status_code == 200:
            sent = True
            message = response.json().get("message", "")
        else:
            sent = False
            message = response.json().get("message", "")
        return sent, message

    @classmethod
    def retrieve_greeting_messages(cls) -> list[str]:
        """Retrieve greeting messages that are new for the user."""
        response = cls.httpx_client.get(
            cls.server_endpoints.retrieve_greeting_messages.path
        )

        cls._check_version(response)
        if response.status_code != 200:
            return []

        greeting_messages = response.json()["messages"]
        return greeting_messages

    @classmethod
    def get_data_summary(cls) -> dict:
        """Get the data summary of the user from the server."""
        response = cls.httpx_client.get(
            cls.server_endpoints.get_data_summary.path,
        )
        cls._raise_on_error(response, "get_data_summary")

        return response.json()

    @classmethod
    def download_all_data(cls, save_dir: Path) -> Path | None:
        """Download all data uploaded by the user from the server.

        Args:
            save_dir: Directory to save the downloaded file to.

        Returns:
            The path to the downloaded file.
        """
        save_path = None

        full_url = cls.base_url + cls.server_endpoints.download_all_data.path
        with httpx.stream(
            "GET",
            full_url,
            headers={
                "Authorization": f"Bearer {cls.get_access_token()}",
                "Prior-Client-Version": get_client_version(),
            },
        ) as response:
            cls._raise_on_error(response, "download_all_data")

            filename = response.headers["Content-Disposition"].split("filename=")[1]
            save_path = Path(save_dir) / filename
            with open(save_path, "wb") as f:
                for data in response.iter_bytes():
                    f.write(data)

        return save_path

    @classmethod
    def delete_dataset(cls, dataset_uid: str) -> list[str]:
        """Delete the dataset with the provided UID from the server.

        Deleting a train set also deletes all associated test sets.

        Args:
            dataset_uid: The UID of the dataset to be deleted.

        Returns:
            The UIDs of all deleted datasets.
        """
        response = cls.httpx_client.delete(
            cls.server_endpoints.delete_dataset.path,
            params={"dataset_uid": dataset_uid},
        )

        cls._raise_on_error(response, "delete_dataset")

        return response.json()["deleted_dataset_uids"]

    @classmethod
    def delete_all_datasets(cls) -> list[str]:
        """Delete all datasets uploaded by the user from the server.

        Returns:
            The UIDs of the deleted datasets.
        """
        response = cls.httpx_client.delete(
            cls.server_endpoints.delete_all_datasets.path,
        )

        cls._raise_on_error(response, "delete_all_datasets")

        return response.json()["deleted_dataset_uids"]

    @classmethod
    def delete_user_account(cls) -> None:
        """Delete the user's account on the server."""
        response = cls.httpx_client.delete(
            cls.server_endpoints.delete_user_account.path,
        )

        cls._raise_on_error(response, "delete_user_account")

    @classmethod
    def estimate_cost(
        cls, req: EstimateCostRequest, *, access_token: str
    ) -> EstimateCostResponse:
        """Quote one operation using dimensions only; never upload or reserve quota."""
        response = cls.httpx_client.post(
            "/tabpfn/estimate_cost",
            json=req.model_dump(mode="json", exclude_none=True),
            headers={"Authorization": f"Bearer {access_token}"},
        )
        return cls._validate_response(response, "estimate_cost", EstimateCostResponse)

    @classmethod
    def get_api_usage(cls, access_token: str):
        """Retrieve current API usage data of the user from the server.

        Returns:
            The usage data as parsed from the JSON response.
        """
        response = cls.httpx_client.post(
            cls.server_endpoints.get_api_usage.path,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        cls._raise_on_error(response, "get_api_usage")

        response = response.json()

        return response
