#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations
from uuid import UUID
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
import io
import json
import logging
from pathlib import Path
import numpy as np
import re
import struct
import time
import traceback
import warnings
from pydantic import BaseModel, ValidationError
from typing import Any, Callable, Dict, Literal, Union, cast

import google_crc32c

import pandas as pd

import backoff
import httpx
from httpx._transports.default import HTTPTransport
from omegaconf import OmegaConf
from tabpfn_client.browser_auth import BrowserAuthHandler
from tabpfn_client.constants import (
    dedup_datasets_enabled,
    force_retransform_enabled,
    force_reupload_enabled,
    TABPFN_CLIENT_MAX_THREAD_PER_UPLOAD,
    TABPFN_CLIENT_TIMEOUT,
    TABPFN_CLIENT_UPLOAD_TIMEOUT,
    TABPFN_CLIENT_API_URL,
)
from tabpfn_common_utils import utils as common_utils
from tabpfn_common_utils.utils import Singleton
from tabpfn_client.api_models import (
    GetDatasetLimitsResponse,
    PrepareTrainSetUploadRequest,
    PrepareTrainSetUploadResponse,
    DuplicateTrainSetErrorResponse,
    PrepareTestSetUploadRequest,
    PrepareTestSetUploadResponse,
    DuplicateTestSetErrorResponse,
    FileUploadInfo,
    FileInfo,
    FitRequest,
    FitResponse,
    PredictRequest,
    PredictResponse,
    TaskConfig,
    ErrorResponse,
)

logger = logging.getLogger(__name__)

# avoid logging of httpx and httpcore on client side
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)


def _on_backoff(details: Dict[str, Any]):
    """Callback function for retry attempts."""
    function_name = details["target"].__name__
    message = (
        f"Exception occurred during {function_name}, retrying in {details['wait']} seconds... "
        f"attempt {details['tries']}: {details['exception']}"
    )
    logger.warning(message)


def _on_giveup(details: Dict[str, Any]):
    """Callback function when retries are exhausted."""
    function_name: str = details["target"].__name__.title()
    message = (
        f"{function_name} method failed after {details['tries']} attempts. "
        f"Giving up. Exception: {details['exception']}"
    )
    logger.error(message)


def _contains_none(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, list):
        return any(_contains_none(item) for item in value)
    return False


class GCPOverloaded(Exception):
    """
    Exception raised when the Google Cloud Platform service is overloaded or
    unavailable.
    """

    pass


class RetryableServerError(Exception):
    """
    Base exception for retryable server-side HTTP errors (typically 5xx).
    """

    pass


class SensitiveDataFilter(logging.Filter):
    def filter(self, record):
        if "password" in record.getMessage():
            original_query = str(record.args[1])
            filtered_query = re.sub(
                r"(password|password_confirm)=[^&]*", r"\1=[FILTERED]", original_query
            )
            record.args = (record.args[0], filtered_query, *record.args[2:])
        return True


# Apply the custom filter to the httpx logger
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
httpx_logger.addFilter(SensitiveDataFilter())

SERVER_CONFIG_FILE = Path(__file__).parent.resolve() / "server_config.yaml"
SERVER_CONFIG = OmegaConf.load(SERVER_CONFIG_FILE)


def get_client_version() -> str:
    try:
        return version("tabpfn_client")
    except PackageNotFoundError:
        # Package not found, should only happen during development. Execute 'pip install -e .' to use the actual
        # version number during development. Otherwise, simply return a version number that is large enough.
        return "5.5.5"


def _get_crc32c_hash(data: bytes) -> str:
    """Computes the CRC32C checksum and returns it as a base64 encoded string."""
    crc32c_value = google_crc32c.value(data)
    return base64.b64encode(struct.pack(">I", crc32c_value)).decode("ascii")


def _serialize_to_parquet(data) -> tuple[bytes, str]:
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, compression="zstd")
    dataset_bytes = buf.getvalue()
    crc32c_b64 = _get_crc32c_hash(dataset_bytes)
    return dataset_bytes, crc32c_b64


class NeedsRefittingError(Exception):
    """
    Exception raised when the server is not able to predict given the current state.
    """

    pass


@dataclass
class ClientOptions:
    """
    Options for the client.
    Can be used to override default client behavior for a single request.

    Parameters
    ----------
    timeout : float, optional
        Timeout for the request in seconds.
    headers : dict[str, str], optional
        Headers for the request overriding the default headers.
    """

    # Note: timeout=None does not fallback to the client default, rather it disables
    # the timeout altogether.
    timeout: float = TABPFN_CLIENT_TIMEOUT
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionResult:
    y_pred: Union[np.ndarray, list[np.ndarray], dict[str, np.ndarray]]
    metadata: dict[str, Any] = field(default_factory=dict)


class SelectiveHTTP2Transport(HTTPTransport):
    def __init__(self, http2_paths=None, *args, **kwargs):
        self.http2_paths = http2_paths or []
        self.http1 = HTTPTransport(http2=False, *args, **kwargs)
        self.http2 = HTTPTransport(http2=True, *args, **kwargs)

    def handle_request(self, request):
        if request.url.path in self.http2_paths:
            return self.http2.handle_request(request)
        return self.http1.handle_request(request)

    def close(self) -> None:
        self.http1.close()
        self.http2.close()


class ServiceClient(Singleton):
    """
    Singleton class for handling communication with the server.
    It encapsulates all the API calls to the server.
    """

    server_config = SERVER_CONFIG
    server_endpoints = SERVER_CONFIG["endpoints"]
    base_url = (
        TABPFN_CLIENT_API_URL
        or f"{server_config.protocol}://{server_config.host}:{server_config.port}"
    )
    fit_path = SERVER_CONFIG["endpoints"]["fit"]["path"]
    httpx_client = httpx.Client(
        base_url=base_url,
        timeout=TABPFN_CLIENT_TIMEOUT,
        headers={"Prior-Client-Version": get_client_version()},
        transport=SelectiveHTTP2Transport(http2_paths=[fit_path]),
        follow_redirects=True,
    )
    _access_token = None
    _dataset_limits: GetDatasetLimitsResponse | None = None
    _dataset_limits_ts: float = 0.0

    @classmethod
    def get_access_token(cls):
        return cls._access_token

    @classmethod
    def get_dataset_limits(cls) -> GetDatasetLimitsResponse | None:
        """Fetch and cache dataset limits. The cache expires after 30 minutes.

        Not thread-safe, but concurrent calls are benign: duplicates fetch the
        same data and the reference assignment is atomic under the GIL."""
        ttl = 1800.0  # 30 minutes
        if (
            cls._dataset_limits is not None
            and (time.monotonic() - cls._dataset_limits_ts) < ttl
        ):
            return cls._dataset_limits
        try:
            response = cls.httpx_client.get("/tabpfn/get_dataset_limits")
            response.raise_for_status()
            cls._dataset_limits = GetDatasetLimitsResponse.model_validate(
                response.json()
            )
            cls._dataset_limits_ts = time.monotonic()
            return cls._dataset_limits
        except Exception:
            logger.debug("Failed to fetch dataset limits", exc_info=True)
            return cls._dataset_limits  # return stale value if available

    @classmethod
    def authorize(cls, access_token: str):
        cls._access_token = access_token
        cls.httpx_client.headers.update(
            {"Authorization": f"Bearer {cls.get_access_token()}"}
        )

    @classmethod
    def reset_authorization(cls):
        cls._access_token = None
        cls.httpx_client.headers.pop("Authorization", None)

    @classmethod
    def fit(
        cls,
        X,
        y,
        task: Literal["classification", "regression"],
        tabpfn_config: Union[dict, None] = None,
        description: str | None = None,
        client_options: ClientOptions | None = None,
        is_refitting: bool = False,
    ) -> UUID:
        """
        Upload a train set to server and return the train set UID if successful.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.
        task: str
            Task type: "classification" or "regression"
        tabpfn_config : dict, optional
            Configuration for the fit method. Supported keys currently include
            `paper_version`.
        description: str, optional
            Description of the dataset and task for the server.
        client_options : ClientOptions, optional
            Per-request options (e.g. timeout, headers) for the fitting API call
            only. Does not apply to file uploads. Because uploads can run before fitting,
            this method may return later than the timeout specified.

        Returns
        -------
        fitted_train_set_id: UUID
            The unique ID of the fitted train set in the server.
        """
        if task not in {"classification", "regression"}:
            raise ValueError("task must be either 'classification' or 'regression'.")

        client_options = client_options or ClientOptions()

        x_bytes, x_crc32c_hash = _serialize_to_parquet(X)
        y_bytes, y_crc32c_hash = _serialize_to_parquet(y)

        limits = cls.get_dataset_limits()
        if limits is not None:
            for name, data in [("x_train", x_bytes), ("y_train", y_bytes)]:
                if len(data) > limits.dataset_max_size_bytes:
                    raise ValueError(
                        f"Compressed size of {name} ({len(data)} bytes) exceeds "
                        f"the server limit of {limits.dataset_max_size_bytes} bytes."
                    )

        if dedup_datasets_enabled():
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
            force_reupload=force_reupload_enabled(),
        )
        res = cls.httpx_client.post(
            url="/tabpfn/prepare_train_set_upload",
            json=prepare_req.model_dump(mode="json"),
            headers=client_options.headers,
        )
        prepare_resp = cast(
            Union[
                PrepareTrainSetUploadResponse,
                DuplicateTrainSetErrorResponse,
            ],
            cls._validate_response(
                res,
                "prepare_train_set_upload",
                response_models={
                    200: PrepareTrainSetUploadResponse,
                    409: DuplicateTrainSetErrorResponse,
                },
            ),
        )

        if isinstance(prepare_resp, DuplicateTrainSetErrorResponse):
            logger.warning(prepare_resp.message)
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

        tabpfn_systems = ["preprocessing", "text"]
        if tabpfn_config and tabpfn_config.get("paper_version") is True:
            tabpfn_systems = []

        res = cls._fit(
            req=FitRequest(
                train_set_upload_id=prepare_resp.train_set_upload_id,
                task=task,
                tabpfn_systems=tabpfn_systems,
                force_retransform=is_refitting or force_retransform_enabled(),
            ),
            timeout=client_options.timeout,
            headers=client_options.headers,
        )
        fit_resp = cast(
            FitResponse,
            cls._validate_response(
                res,
                "fit",
                response_models={200: FitResponse},
            ),
        )

        return fit_resp.fitted_train_set_id

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
    ) -> httpx.Response:
        return cls.httpx_client.post(
            url="/tabpfn/fit",
            json=req.model_dump(mode="json"),
            headers=headers,
            timeout=timeout,
        )

    @classmethod
    def predict(
        cls,
        fitted_train_set_id: UUID,
        x_test,
        task: Literal["classification", "regression"],
        tabpfn_config: Union[dict, None] = None,
        predict_params: Union[dict, None] = None,
        client_options: ClientOptions | None = None,
    ) -> PredictionResult:
        """
        Predict the class labels for the provided data (test set).

        Parameters
        ----------
        fitted_train_set_id : UUID
            The unique ID of the fitted train set in the server.
        x_test : array-like of shape (n_samples, n_features)
            The test input.
        task: str, optional
            Task type: "classification" or "regression"
        tabpfn_config : dict, optional
            Configuration used to initialize the the TabPFN model.
        predict_params: dict, optional
            Parameters for the predict method.
        client_options : ClientOptions, optional
            Per-request options (e.g. timeout, headers) for the fitting API call
            only. Does not apply to file uploads. Because uploads can run before fitting,
            this method may return later than the timeout specified.

        Returns
        -------
        prediction_result : PredictionResult
            The result from the predict API call containing the prediction and metadata.
        """
        client_options = client_options or ClientOptions()

        x_test_bytes, x_test_crc32c_hash = _serialize_to_parquet(x_test)

        limits = cls.get_dataset_limits()
        if limits is not None:
            if len(x_test_bytes) > limits.dataset_max_size_bytes:
                raise ValueError(
                    f"Compressed size of x_test ({len(x_test_bytes)} bytes) exceeds "
                    f"the server limit of {limits.dataset_max_size_bytes} bytes."
                )

        if dedup_datasets_enabled():
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
            force_reupload=force_reupload_enabled(),
        )
        res = cls.httpx_client.post(
            url="/tabpfn/prepare_test_set_upload",
            json=prepare_req.model_dump(mode="json"),
            headers=client_options.headers,
        )
        prepare_resp = cast(
            Union[
                PrepareTestSetUploadResponse,
                DuplicateTestSetErrorResponse,
            ],
            cls._validate_response(
                res,
                "prepare_test_set_upload",
                response_models={
                    200: PrepareTestSetUploadResponse,
                    404: ErrorResponse,
                    409: DuplicateTestSetErrorResponse,
                },
                handlers={404: cls._raise_not_found_error},
            ),
        )

        if isinstance(prepare_resp, DuplicateTestSetErrorResponse):
            logger.warning(prepare_resp.message)
        else:
            cls._upload_to_gcs(
                "x_test",
                x_test_bytes,
                prepare_resp.x_test_info,
            )

        # Strip client-only keys that the server does not expect.
        if tabpfn_config is not None:
            tabpfn_config = {
                k: v for k, v in tabpfn_config.items() if k not in {"paper_version"}
            }

        res = cls._predict(
            req=PredictRequest(
                test_set_upload_id=prepare_resp.test_set_upload_id,
                fitted_train_set_id=fitted_train_set_id,
                task_config=TaskConfig(
                    task=task,
                    tabpfn_config=tabpfn_config,
                    predict_params=predict_params,
                ),
                force_retransform=force_retransform_enabled(),
            ),
            timeout=client_options.timeout,
            headers=client_options.headers,
        )
        predict_resp = cast(
            PredictResponse,
            cls._validate_response(
                res,
                "predict",
                response_models={200: PredictResponse, 404: ErrorResponse},
                handlers={404: cls._raise_not_found_error},
            ),
        )

        prediction = predict_resp.prediction

        if isinstance(prediction, dict):
            result = {}
            for k, v in prediction.items():
                if isinstance(v, list):
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
            metadata=predict_resp.metadata.model_dump(mode="json"),
        )

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
    ) -> httpx.Response:
        return cls.httpx_client.post(
            url="/tabpfn/predict",
            json=req.model_dump(mode="json"),
            headers=headers,
            timeout=timeout,
        )

    @classmethod
    def _upload_to_gcs(cls, dataset: str, data: bytes, info: FileUploadInfo) -> None:
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
            max_workers=min(TABPFN_CLIENT_MAX_THREAD_PER_UPLOAD, num_chunks)
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
        interval=0,
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
        resp = cls.httpx_client.put(
            url,
            content=chunk,
            headers=headers,
            timeout=TABPFN_CLIENT_UPLOAD_TIMEOUT,
        )
        if resp.status_code == 200:
            return
        if resp.status_code in {502, 503, 504}:
            raise RetryableServerError(
                f"GCS upload failed for dataset {dataset} at chunk {chunk_index}: "
                f"{resp.status_code} {resp.text}"
            )
        raise RuntimeError(
            f"GCS upload permanently failed for dataset {dataset} at chunk {chunk_index}: "
            f"{resp.status_code} {resp.text}"
        )

    @staticmethod
    def _raise_not_found_error(error_response: ErrorResponse) -> None:
        if error_response.error_code == "NOT_FOUND":
            raise NeedsRefittingError(error_response.message)
        raise RuntimeError(error_response.message)

    @staticmethod
    def _warn_if_deprecated(response: httpx.Response) -> None:
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
    def _validate_response(
        response: httpx.Response,
        method_name: str,
        only_version_check: bool = False,
        response_models: dict[int, type[BaseModel]] | None = None,
        handlers: dict[int, Callable[[BaseModel], Any]] | None = None,
    ) -> BaseModel | None:
        ServiceClient._warn_if_deprecated(response)

        if response.status_code == 200 and response_models is None:
            return

        # Read response.
        load = {}
        try:
            # This if clause is necessary for streaming responses (e.g. download) to
            # prevent httpx.ResponseNotRead error.
            if not response.is_closed:
                response.read()
            load = response.json()
        except json.JSONDecodeError as e:
            logging.info(f"Failed to parse JSON from response in {method_name}: {e}")

        # Check if the server requires a newer client version.
        if response.status_code == 426:
            logger.info(
                f"Fail to call {method_name}, response status: {response.status_code}"
            )
            raise RuntimeError(load.get("message") or load.get("detail", ""))

        # Inform about user errors, unless the caller only wants a version
        # check or a response_model is registered for this status code.
        if (
            400 <= response.status_code < 500
            and not only_version_check
            and not (response_models and response.status_code in response_models)
        ):
            logger.info(
                f"Fail to call {method_name}, response status: {response.status_code}"
            )
            message = (
                load.get("message") or load.get("detail", "") or response.reason_phrase
            )
            status_code = response.status_code
            raise RuntimeError(
                f"Fail to call {method_name} with error: {status_code}, {message}"
            )

        if response_models is not None and response.status_code == 200:
            if 200 not in response_models:
                raise RuntimeError(
                    f"Fail to call {method_name} with no response model configured for status 200"
                )

        if response_models is not None and response.status_code in response_models:
            if load is None and response.status_code == 200:
                raise RuntimeError(
                    f"Fail to call {method_name} with invalid JSON response"
                )

            if load is not None:
                try:
                    parsed_response = response_models[
                        response.status_code
                    ].model_validate(load)
                except ValidationError as e:
                    if response.status_code == 200:
                        raise RuntimeError(
                            f"Fail to call {method_name} with invalid response schema: {e}"
                        ) from e
                    response.raise_for_status()
                else:
                    if handlers is not None and response.status_code in handlers:
                        handlers[response.status_code](parsed_response)
                    return parsed_response

        if response.status_code == 200:
            return

        # If we not only want to check the version compatibility, also raise other errors.
        if not only_version_check:
            # Treat selected errors as retryable.
            if response.status_code in {408, 502, 503, 504}:
                error_msg = (
                    f"Fail to call {method_name} with error: {response.status_code}, reason: "
                    f"{response.reason_phrase} and text: {response.text}"
                )
                raise RetryableServerError(error_msg)
            if load is not None:
                message = (
                    load.get("message")
                    or load.get("detail", "")
                    or response.reason_phrase
                )
                trace_id = load.get("trace_id")
                status_code = response.status_code
                raise RuntimeError(
                    f"Fail to call {method_name} with error: {status_code}, {message} {trace_id=}"
                )
            logger.error(
                f"Fail to call {method_name}, response status: {response.status_code}"
            )
            try:
                if (
                    len(
                        reponse_split_up := response.text.split(
                            "The following exception has occurred:"
                        )
                    )
                    > 1
                ):
                    relevant_reponse_text = reponse_split_up[1].split(
                        "debug_error_string"
                    )[0]
                    if "ValueError" in relevant_reponse_text:
                        # Extract the ValueError message
                        value_error_msg = relevant_reponse_text.split(
                            "ValueError. Arguments: ("
                        )[1].split(",)")[0]
                        # Remove extra quotes and spaces
                        value_error_msg = value_error_msg.strip("'")
                        # Raise the ValueError with the extracted message
                        raise ValueError(value_error_msg)
                    raise RuntimeError(relevant_reponse_text)
            except Exception as e:
                if isinstance(e, (ValueError, RuntimeError)):
                    raise e
            raise RuntimeError(
                f"Fail to call {method_name} with error: {response.status_code}, reason: "
                f"{response.reason_phrase} and text: {response.text}"
            )

    @classmethod
    def try_connection(cls) -> bool:
        """
        Check if server is reachable and accepts the connection.
        """
        found_valid_connection = False
        try:
            response = cls.httpx_client.get(cls.server_endpoints.root.path)
            cls._validate_response(response, "try_connection", only_version_check=True)
            if response.status_code == 200:
                found_valid_connection = True

        except httpx.ConnectError as e:
            logger.error(f"Failed to connect to the server with error: {e}")
            traceback.print_exc()
            found_valid_connection = False

        return found_valid_connection

    @classmethod
    def is_auth_token_outdated(cls, access_token) -> Union[bool, None]:
        """
        Check if the provided access token is valid and return True if successful.
        """
        is_authenticated = False
        response = cls.httpx_client.get(
            cls.server_endpoints.protected_root.path,
            headers={"Authorization": f"Bearer {access_token}"},
        )

        cls._validate_response(
            response, "is_auth_token_outdated", only_version_check=True
        )
        if response.status_code == 200:
            is_authenticated = True
        elif response.status_code == 403:
            # 403 means user is not verified
            is_authenticated = None
        return is_authenticated

    @classmethod
    def validate_email(cls, email: str) -> tuple[bool, str]:
        """
        Send entered email to server that checks if it is valid and not already in use.

        Parameters
        ----------
        email : str

        Returns
        -------
        is_valid : bool
            True if the email is valid.
        message : str
            The message returned from the server.
        """
        response = cls.httpx_client.post(
            cls.server_endpoints.validate_email.path, params={"email": email}
        )

        cls._validate_response(response, "validate_email", only_version_check=True)
        if response.status_code == 200:
            is_valid = True
            message = ""
        else:
            is_valid = False
            message = response.json()["detail"]

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
        """
        Register a new user with the provided credentials.

        Parameters
        ----------
        email : str
        password : str
        password_confirm : str
        validation_link: str
        additional_info : dict

        Returns
        -------
        is_created : bool
            True if the user is created successfully.
        message : str
            The message returned from the server.
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

        cls._validate_response(response, "register", only_version_check=True)
        if response.status_code == 200:
            is_created = True
            message = response.json()["message"]
        else:
            is_created = False
            message = response.json()["detail"]

        access_token = response.json()["token"] if is_created else None
        return is_created, message, access_token

    @classmethod
    def verify_email(cls, token: str, access_token: str) -> tuple[bool, str]:
        """
        Verify the email with the provided token.

        Parameters
        ----------
        token : str
        access_token : str

        Returns
        -------
        is_verified : bool
            True if the email is verified successfully.
        message : str
            The message returned from the server.
        """

        response = cls.httpx_client.get(
            cls.server_endpoints.verify_email.path,
            params={"token": token, "access_token": access_token},
        )
        cls._validate_response(response, "verify_email", only_version_check=True)
        if response.status_code == 200:
            is_verified = True
            message = response.json()["message"]
        else:
            is_verified = False
            message = response.json()["detail"]

        return is_verified, message

    @classmethod
    def login(
        cls, email: str, password: str
    ) -> tuple[str | None, str, int, str | None]:
        """
        Login with the provided credentials and return the access token if successful.

        Parameters
        ----------
        email : str
        password : str

        Returns
        -------
        access_token : str | None
            The access token returned from the server. Return None if login fails.
        message : str
            The message returned from the server.
        status_code : int
            The status code returned from the server.
        session_id : str | None
            The session id returned from the server.
        """

        access_token = None
        session_id = None
        response = cls.httpx_client.post(
            cls.server_endpoints.login.path,
            data=common_utils.to_oauth_request_form(email, password),
        )

        cls._validate_response(response, "login", only_version_check=True)
        if response.status_code == 200:
            access_token = response.json()["access_token"]
            session_id = response.cookies.get("session_id")
            message = ""
        elif response.status_code == 403:
            access_token = response.headers.get("access_token")
            message = "Email not verified"
        else:
            try:
                message = response.json()["detail"]
            except (json.JSONDecodeError, KeyError):
                message = (
                    response.text
                    or f"Login failed with status code {response.status_code}"
                )
        # status code signifies the success of the login, issues with password, and email verification
        # 200 : success, 401 : wrong password, 403 : email not verified yet
        return access_token, message, response.status_code, session_id

    @classmethod
    def get_password_policy(cls) -> dict:
        """
        Get the password policy from the server.

        Returns
        -------
        password_policy : {}
            The password policy returned from the server.
        """

        response = cls.httpx_client.get(
            cls.server_endpoints.password_policy.path,
        )
        cls._validate_response(response, "get_password_policy", only_version_check=True)

        return response.json()["requirements"]

    @classmethod
    def send_reset_password_email(cls, email: str) -> tuple[bool, str]:
        """
        Let the server send an email for resetting the password.
        """
        response = cls.httpx_client.post(
            cls.server_endpoints.send_reset_password_email.path,
            params={"email": email},
        )
        if response.status_code == 200:
            sent = True
            message = response.json()["message"]
        else:
            sent = False
            message = response.json()["detail"]
        return sent, message

    @classmethod
    def send_verification_email(cls, access_token: str) -> tuple[bool, str]:
        """
        Let the server send an email for verifying the email.
        """
        response = cls.httpx_client.post(
            cls.server_endpoints.send_verification_email.path,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if response.status_code == 200:
            sent = True
            message = response.json()["message"]
        else:
            sent = False
            message = response.json()["detail"]
        return sent, message

    @classmethod
    def retrieve_greeting_messages(cls) -> list[str]:
        """
        Retrieve greeting messages that are new for the user.
        """
        response = cls.httpx_client.get(
            cls.server_endpoints.retrieve_greeting_messages.path
        )

        cls._validate_response(
            response, "retrieve_greeting_messages", only_version_check=True
        )
        if response.status_code != 200:
            return []

        greeting_messages = response.json()["messages"]
        return greeting_messages

    @classmethod
    def get_data_summary(cls) -> dict:
        """
        Get the data summary of the user from the server.

        Returns
        -------
        data_summary : dict
            The data summary returned from the server.
        """
        response = cls.httpx_client.get(
            cls.server_endpoints.get_data_summary.path,
        )
        cls._validate_response(response, "get_data_summary")

        return response.json()

    @classmethod
    def download_all_data(cls, save_dir: Path) -> Union[Path, None]:
        """
        Download all data uploaded by the user from the server.

        Returns
        -------
        save_path : Path | None
            The path to the downloaded file. Return None if download fails.

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
            cls._validate_response(response, "download_all_data")

            filename = response.headers["Content-Disposition"].split("filename=")[1]
            save_path = Path(save_dir) / filename
            with open(save_path, "wb") as f:
                for data in response.iter_bytes():
                    f.write(data)

        return save_path

    @classmethod
    def delete_dataset(cls, dataset_uid: str) -> list[str]:
        """
        Delete the dataset with the provided UID from the server.
        Note that deleting a train set with lead to deleting all associated test sets.

        Parameters
        ----------
        dataset_uid : str
            The UID of the dataset to be deleted.

        Returns
        -------
        deleted_dataset_uids : [str]
            The list of deleted dataset UIDs.

        """
        response = cls.httpx_client.delete(
            cls.server_endpoints.delete_dataset.path,
            params={"dataset_uid": dataset_uid},
        )

        cls._validate_response(response, "delete_dataset")

        return response.json()["deleted_dataset_uids"]

    @classmethod
    def delete_all_datasets(cls) -> [str]:
        """
        Delete all datasets uploaded by the user from the server.

        Returns
        -------
        deleted_dataset_uids : [str]
            The list of deleted dataset UIDs.
        """
        response = cls.httpx_client.delete(
            cls.server_endpoints.delete_all_datasets.path,
        )

        cls._validate_response(response, "delete_all_datasets")

        return response.json()["deleted_dataset_uids"]

    @classmethod
    def delete_user_account(cls, confirm_pass: str) -> None:
        response = cls.httpx_client.delete(
            cls.server_endpoints.delete_user_account.path,
            params={"confirm_password": confirm_pass},
        )

        cls._validate_response(response, "delete_user_account")

    @classmethod
    def try_browser_login(cls) -> tuple[bool, str]:
        """
        Attempts browser-based login flow
        Returns (success: bool, message: str)
        """
        browser_auth = BrowserAuthHandler(cls.server_config.gui_url)
        success, token = browser_auth.try_browser_login()

        if success and token:
            # Don't authorize directly, let UserAuthenticationClient handle it
            return True, token

        return False, "Browser login failed or was cancelled"

    @classmethod
    def get_api_usage(cls, access_token: str):
        """
        Retrieve current API usage data of the user from the server.
        Returns summary: str
        """
        response = cls.httpx_client.post(
            cls.server_endpoints.get_api_usage.path,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        cls._validate_response(response, "get_api_usage")

        response = response.json()

        return response
