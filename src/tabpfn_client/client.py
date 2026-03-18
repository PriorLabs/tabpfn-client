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
from pydantic import ValidationError
from typing import Any, Dict, Literal, Optional, Union

import google_crc32c

import pandas as pd

import backoff
import httpx
from httpx._transports.default import HTTPTransport
from omegaconf import OmegaConf
from tabpfn_client.browser_auth import BrowserAuthHandler
from tabpfn_client.constants import dedup_datasets_enabled, force_reupload_enabled
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

# Chunks are usually 32 MB in size, at maximum 16 threads we achieve full parallelization
# up to 256 MB.
_DEFAULT_MAX_UPLOAD_PARALLELISM = 8
_DEFAULT_HTTPX_TIMEOUT = 900.0  # 15 minutes


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

    timeout: float = _DEFAULT_HTTPX_TIMEOUT
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
    base_url = f"{server_config.protocol}://{server_config.host}:{server_config.port}"
    fit_path = SERVER_CONFIG["endpoints"]["fit"]["path"]
    httpx_client = httpx.Client(
        base_url=base_url,
        timeout=_DEFAULT_HTTPX_TIMEOUT,
        headers={"client-version": get_client_version()},
        transport=SelectiveHTTP2Transport(http2_paths=[fit_path]),
        follow_redirects=True,
    )
    _access_token = None
    _dataset_limits: GetDatasetLimitsResponse | None = None
    _dataset_limits_ts: float = 0.0

    @staticmethod
    def _process_tabpfn_config(
        tabpfn_config: Union[dict, None],
    ) -> tuple[bool, list[str], Union[dict, None], Optional[dict]]:
        """
        Process tabpfn_config to extract paper_version, tabpfn_systems and tabpfnr_params.

        Parameters
        ----------
        tabpfn_config : dict or None
            Configuration dict that may contain a 'paper_version' key.

        Returns
        -------
        paper_version : bool
            Whether to use paper version (affects tabpfn_systems).
        tabpfn_systems : list[str]
            List of systems to use (empty if paper_version, otherwise ["preprocessing", "text"]).
        processed_config : dict or None
            The config dict with paper_version removed (if present), or None if input was None.
        tabpfnr_params:
        """
        tabpfnr_params: Optional[dict[str, any]]
        if tabpfn_config is None:
            paper_version = False
            processed_config = None
            tabpfnr_params = None
        else:
            # Make a copy to avoid modifying the original
            processed_config = tabpfn_config.copy()
            paper_version = processed_config.pop("paper_version", False)

            # Thinking params are only used during fit and are not accepted by the underlying model.
            processed_config.pop("thinking", None)
            tabpfnr_params = processed_config.pop("thinking_params", None)

        tabpfn_systems = [] if paper_version else ["preprocessing", "text"]
        return paper_version, tabpfn_systems, processed_config, tabpfnr_params

    @staticmethod
    def _build_tabpfn_params(tabpfn_config: Union[dict, None]) -> dict:
        """
        Build parameters dict for tabpfn_config and tabpfn_systems.

        This is a unified helper for both fit and predict methods to ensure
        consistent parameter handling.

        Parameters
        ----------
        tabpfn_config : dict or None
            Configuration dict that may contain a 'paper_version' key.

        Returns
        -------
        params : dict
            Dictionary containing 'tabpfn_systems' and optionally 'tabpfn_config'.
        """
        _, tabpfn_systems, processed_config, _ = ServiceClient._process_tabpfn_config(
            tabpfn_config
        )

        params = {"tabpfn_systems": json.dumps(tabpfn_systems)}

        if processed_config is not None:
            params["tabpfn_config"] = json.dumps(
                processed_config, default=lambda x: x.to_dict()
            )

        return params

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
            response = cls.httpx_client.get("/tabpfn/get_dataset_limits/")
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
        task: Optional[Literal["classification", "regression"]] = None,
        tabpfn_config: Union[dict, None] = None,
        description: str | None = None,
        client_options: ClientOptions | None = None,
    ) -> UUID:
        """
        Upload a train set to server and return the train set UID if successful.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.
        task: str, optional
            Task type: "classification" or "regression"
        tabpfn_config : dict, optional
            Configuration for the fit method. Includes tabpfn_systems, paper_version and thinking params.
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
        client_options = client_options or ClientOptions()

        x_bytes, x_crc32c_hash = _serialize_to_parquet(X)
        y_bytes, y_crc32c_hash = _serialize_to_parquet(y)

        limits = cls.get_dataset_limits()
        if limits is not None:
            for name, data in [("x_train", x_bytes), ("y_train", y_bytes)]:
                if len(data) > limits.max_size_bytes:
                    raise ValueError(
                        f"Compressed size of {name} ({len(data)} bytes) exceeds "
                        f"the server limit of {limits.max_size_bytes} bytes."
                    )

        if dedup_datasets_enabled():
            x_dedup_hash = x_crc32c_hash
            y_dedup_hash = y_crc32c_hash
        else:
            x_dedup_hash = None
            y_dedup_hash = None

        res = cls.httpx_client.post(
            url="/tabpfn/prepare_train_set_upload/",
            json=PrepareTrainSetUploadRequest(
                x_train_info=FileInfo(
                    format="parquet", hash=x_dedup_hash, size_bytes=len(x_bytes)
                ),
                y_train_info=FileInfo(
                    format="parquet", hash=y_dedup_hash, size_bytes=len(y_bytes)
                ),
                description=description,
                force_reupload=force_reupload_enabled(),
            ).model_dump(),
            headers=client_options.headers,
        )
        try:
            if res.status_code == 409:
                return DuplicateTrainSetErrorResponse.model_validate(res.json())
        except ValidationError:
            res.raise_for_status()

        cls._validate_response(res, "prepare_train_set_upload")  # TODO needed?
        prepare_resp = PrepareTrainSetUploadResponse.model_validate(res.json())

        if isinstance(prepare_resp, DuplicateTrainSetErrorResponse):
            logger.warning(prepare_resp.message)
        elif isinstance(prepare_resp, PrepareTrainSetUploadResponse):
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
            ),
            timeout=client_options.timeout,
            headers=client_options.headers,
        )

        cls._validate_response(res, "fit")  # TODO needed?
        fit_resp = FitResponse.model_validate(res.json())

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
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return cls.httpx_client.post(
            url="/tabpfn/fit/",
            json=req.model_dump(),
            headers=headers,
            timeout=timeout,
        )

    def predict(
        cls,
        fitted_train_set_id: UUID,
        x_test,
        task: Literal["classification", "regression"],
        tabpfn_config: Union[dict, None] = None,
        predict_params: Union[dict, None] = None,
        client_options: ClientOptions | None = None,
        X_train=None,
        y_train=None,
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
        X_train: array-like of shape (n_samples, n_features), optional
            The training input samples. Needed in case of refitting.
        y_train: array-like of shape (n_samples,), optional
            The target values. Needed in case of refitting.

        Returns
        -------
        prediction_result : PredictionResult
            The result from the predict API call containing the prediction and metadata.
        """
        client_options = client_options or ClientOptions()

        x_test_bytes, x_test_crc32c_hash = _serialize_to_parquet(x_test)

        limits = cls.get_dataset_limits()
        if limits is not None:
            if len(x_test_bytes) > limits.max_size_bytes:
                raise ValueError(
                    f"Compressed size of x_test ({len(x_test_bytes)} bytes) exceeds "
                    f"the server limit of {limits.max_size_bytes} bytes."
                )

        if dedup_datasets_enabled():
            x_test_dedup_hash = x_test_crc32c_hash
        else:
            x_test_dedup_hash = None

        res = cls.httpx_client.post(
            url="/tabpfn/prepare_test_set_upload/",
            json=PrepareTestSetUploadRequest(
                fitted_train_set_id=fitted_train_set_id,
                x_test_info=FileInfo(
                    format="parquet",
                    hash=x_test_dedup_hash,
                    size_bytes=len(x_test_bytes),
                ),
                force_reupload=force_reupload_enabled(),
            ).model_dump(),
            headers=client_options.headers,
        )
        try:
            if res.status_code == 404:
                err_resp = ErrorResponse.model_validate(res.json())
                if err_resp.error_code == "NOT_FOUND":
                    raise NeedsRefittingError(err_resp.message)
                raise RuntimeError(err_resp.message)
            if res.status_code == 409:
                return DuplicateTestSetErrorResponse.model_validate(res.json())
        except ValidationError:
            res.raise_for_status()

        cls._validate_response(res, "prepare_test_set_upload")  # TODO needed?
        prepare_resp = PrepareTestSetUploadResponse.model_validate(res.json())

        if isinstance(prepare_resp, DuplicateTestSetErrorResponse):
            logger.warning(prepare_resp.message)
        elif isinstance(prepare_resp, PrepareTestSetUploadResponse):
            cls._upload_to_gcs(
                "x_test",
                x_test_bytes,
                prepare_resp.x_test_info,
            )

        res = cls._predict(
            req=PredictRequest(
                test_set_upload_id=prepare_resp.test_set_upload_id,
                fitted_train_set_id=fitted_train_set_id,
                task_config=TaskConfig(
                    task=task,
                    tabpfn_config=tabpfn_config,
                    predict_params=predict_params,
                ),
            ),
            timeout=client_options.timeout,
            headers=client_options.headers,
        )
        try:
            if res.status_code == 404:
                err_resp = ErrorResponse.model_validate(res.json())
                if err_resp.error_code == "NOT_FOUND":
                    raise NeedsRefittingError(err_resp.message)
                raise RuntimeError(err_resp.message)
        except ValidationError:
            res.raise_for_status()

        cls._validate_response(res, "predict")  # TODO needed?
        predict_resp = PredictResponse.model_validate(res.json())

        prediction = predict_resp.prediction

        if isinstance(prediction, dict):
            result = {}
            for k, v in prediction.items():
                if isinstance(v, list):
                    result[k] = np.array(v)
                else:
                    # The value should always be a list or list-of-lists,
                    # leaving this here as an extra precaution and future proofing.
                    result[k] = v
        else:
            result = np.array(prediction)

        return PredictionResult(
            y_pred=result,
            metadata=predict_resp.metadata.model_dump(),
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
        timeout: float | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        return cls.httpx_client.post(
            url="/tabpfn/predict/",
            json=req.model_dump(),
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
            max_workers=min(_DEFAULT_MAX_UPLOAD_PARALLELISM, num_chunks)
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
            timeout=600,
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
    def _validate_response(
        response: httpx.Response, method_name, only_version_check=False
    ):
        # If status code is 200, no errors occurred on the server side.
        if response.status_code == 200:
            return

        # Read response.
        load = None
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
            raise RuntimeError(load.get("detail"))

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
                raise RuntimeError(f"Fail to call {method_name} with error: {load}")
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
                "client-version": get_client_version(),
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
