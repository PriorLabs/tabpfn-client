#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations
import base64
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from importlib.metadata import PackageNotFoundError, version
import io
import json
import logging
import os
from pathlib import Path
import re
import struct
import threading
import time
import traceback
from typing import Annotated, Any, Dict, Literal, Optional, Union

import google_crc32c

import pandas as pd
from uuid import UUID

from tqdm import tqdm

import backoff
import httpx
from httpx._transports.default import HTTPTransport
import numpy as np
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
import sseclient
from tabpfn_client.browser_auth import BrowserAuthHandler
from tabpfn_client.constants import CACHE_DIR
from tabpfn_common_utils import utils as common_utils
from tabpfn_common_utils.utils import Singleton
import xxhash

logger = logging.getLogger(__name__)

# avoid logging of httpx and httpcore on client side
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpcore.http11").setLevel(logging.WARNING)

_DEFAULT_CHUNK_UPLOAD_PARALLELISM = 16
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


def _caching_disabled() -> bool:
    val = os.getenv("DISABLE_DS_CACHING", "")
    disabled = str(val).lower() in {"1", "true", "yes", "on"}
    if disabled:
        logger.warning("Dataset caching is disabled.")
    return disabled


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


class DatasetUIDCacheManager:
    """
    Manages a cache of the last 50 uploaded datasets, tracking dataset hashes and their UIDs.

    Can be disabled using the environment variable 'DISABLE_DS_CACHING'.
    """

    def __init__(self):
        self.file_path = CACHE_DIR / "dataset_cache"
        self.cache_limit = 50
        self.disable_ds_caching = _caching_disabled()

        self.cache = self.load_cache() if not self.disable_ds_caching else OrderedDict()

    def load_cache(self):
        """
        Loads the cache from disk if it exists, otherwise initializes an empty cache.
        """
        if os.path.exists(self.file_path):
            with open(self.file_path, "r") as file:
                data = json.load(file)
                return OrderedDict(data)
        return OrderedDict()

    def _compute_hash(self, *args):
        combined_bytes = b"".join(
            item if isinstance(item, bytes) else str.encode(item) for item in args
        )
        return xxhash.xxh64(combined_bytes).hexdigest()

    def get_dataset_uid(self, *args):
        """
        Generates hash by all received arguments and returns cached dataset uid if in cache, otherwise None.
        """
        dataset_hash = self._compute_hash(*args)
        if str(dataset_hash) in self.cache and not self.disable_ds_caching:
            self.cache.move_to_end(dataset_hash)
            return self.cache[dataset_hash], dataset_hash
        else:
            return None, dataset_hash

    def add_dataset_uid(self, hash: str, dataset_uid: str):
        """
        Adds a new dataset to the cache, removing the oldest item if the cache exceeds 50 entries.
        Assumes the dataset is not already in the cache.
        """
        self.cache[hash] = dataset_uid
        # Move to end for the case that hash already was stored in the cache
        self.cache.move_to_end(hash)
        if len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)

        if not self.disable_ds_caching:
            self.save_cache()

    def save_cache(self):
        """
        Saves the current cache to disk.
        """
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        with open(self.file_path, "w") as file:
            json.dump(self.cache, file)

    def delete_uid(self, dataset_uid: str) -> Optional[str]:
        """
        Deletes an entry from the cache based on the dataset UID.
        """
        hash_to_delete = None
        for hash, uid in self.cache.items():
            if uid == dataset_uid:
                hash_to_delete = hash
                break

        if hash_to_delete:
            del self.cache[hash_to_delete]
            self.save_cache()
            return hash_to_delete
        return None


# Apply the custom filter to the httpx logger
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)
httpx_logger.addFilter(SensitiveDataFilter())

SERVER_CONFIG_FILE = Path(__file__).parent.resolve() / "server_config.yaml"
SERVER_CONFIG = OmegaConf.load(SERVER_CONFIG_FILE)


class ModelType(Enum):
    TABPFN = auto()
    TABPFN_R = auto()


def get_client_version() -> str:
    try:
        return version("tabpfn_client")
    except PackageNotFoundError:
        # Package not found, should only happen during development. Execute 'pip install -e .' to use the actual
        # version number during development. Otherwise, simply return a version number that is large enough.
        return "5.5.5"


class FitProgressEvent(BaseModel):
    """
    Update to the progress of fitting the model.
    """

    status: Literal["in_progress"] = "in_progress"
    estimated_time_remaining_sec: float
    total_steps: int
    current_step: int
    message: str


class FitCompleteEvent(BaseModel):
    status: Literal["complete"] = "complete"
    # Identifies the dataset and thus the model
    train_set_uid: str


class FitErrorEvent(BaseModel):
    """
    Error event emitted by the server during fitting.
    """

    status: Literal["error"] = "error"
    message: str


###############################################################
## Gapi models                                               ##
###############################################################


# From: apps/gapi/routers/tabpfn/prepare_train_set_upload.py
class FileInfo(BaseModel):
    format: Literal["csv", "parquet"]
    hash: Optional[str] = None
    size_bytes: int
    use_chunks: bool = True


class FileUploadInfo(BaseModel):
    signed_urls: list[str]
    expires_at: float
    required_headers: dict[str, str]


class PrepareTrainSetUploadRequest(BaseModel):
    x_train: FileInfo
    y_train: FileInfo


class PrepareTrainSetUploadResponse(BaseModel):
    upload_id: UUID
    x_train: FileUploadInfo
    y_train: FileUploadInfo


class DuplicateFilesUploadedResponse(BaseModel):
    message: str
    upload_id: UUID


# From: apps/gapi/routers/tabpfn/get_dataset_limits.py
class GetDatasetLimitsResponse(BaseModel):
    max_size_bytes: int
    max_cells: int
    max_cols: int
    max_classes: int


###############################################################

ResponseEvents = Annotated[
    Union[FitCompleteEvent, FitProgressEvent, FitErrorEvent],
    Field(discriminator="status"),
]
ResponseEventAdapter = TypeAdapter(ResponseEvents)


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


@dataclass
class ClientOptions:
    """
    Options for the client.
    Can be used to override default client behavior for a single request.

    Parameters
    ----------
    timeout : float, optional
        Timeout for the request in seconds.
    extra_headers : dict[str, str], optional
        Extra headers for the request.
    """

    timeout: float = _DEFAULT_HTTPX_TIMEOUT
    extra_headers: dict[str, str] | None = field(default_factory=dict)


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
    dataset_uid_cache_manager = DatasetUIDCacheManager()

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
        model_type: ModelType,
        tabpfn_config: Union[dict, None] = None,
        task: Optional[Literal["classification", "regression"]] = None,
        description: str = "",
        client_options: ClientOptions | None = None,
        dedup_files: bool = True,
    ) -> str:
        """
        Upload a train set to server and return the train set UID if successful.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.
        model_type : ModelType
            Whether to fit a TabPFN or TabPFN-R model.
        tabpfn_config : dict, optional
            Configuration for the fit method. Includes tabpfn_systems, paper_version and thinking params.
        task: str, optional
            Task type: "classification" or "regression"
        description: str, optional
            Description of the dataset and task for the server.
        client_options : ClientOptions, optional
            Per-request options (e.g. timeout, extra_headers) for the fitting API call
            only. Does not apply to file uploads. Because uploads can run before fitting,
            this method may return later than the timeout specified.
        dedup_files: bool, optional
            If True, the client will check if the same files (identified by their content hash)
            have already been uploaded and return the corresponding upload_id. Default is True.
            This is only used if DISABLE_DS_CACHING is False.

        Returns
        -------
        train_set_uid : str
            The unique ID of the train set in the server.

        """
        client_options = client_options or ClientOptions()

        if model_type == ModelType.TABPFN_R:
            raise NotImplementedError("TABPFN-R is no longer supported")

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

        if dedup_files and not _caching_disabled():
            x_dedup_hash = x_crc32c_hash
            y_dedup_hash = y_crc32c_hash
        else:
            x_dedup_hash = None
            y_dedup_hash = None

        prepare_resp = cls._prepare_train_set_upload(
            req=PrepareTrainSetUploadRequest(
                x_train=FileInfo(
                    format="parquet", hash=x_dedup_hash, size_bytes=len(x_bytes)
                ),
                y_train=FileInfo(
                    format="parquet", hash=y_dedup_hash, size_bytes=len(y_bytes)
                ),
            ),
            extra_headers=client_options.extra_headers,
        )

        if isinstance(prepare_resp, DuplicateFilesUploadedResponse):
            logger.warning(prepare_resp.message)

        if isinstance(prepare_resp, PrepareTrainSetUploadResponse):
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(
                        cls._upload_to_gcs,
                        "x_train",
                        x_bytes,
                        x_dedup_hash,
                        prepare_resp.x_train,
                    ),
                    pool.submit(
                        cls._upload_to_gcs,
                        "y_train",
                        y_bytes,
                        y_dedup_hash,
                        prepare_resp.y_train,
                    ),
                ]
                try:
                    for future in as_completed(futures):
                        future.result()
                except Exception:
                    for f in futures:
                        f.cancel()
                    raise

        form_data = {"desc": description}
        query_params = cls._build_tabpfn_params(tabpfn_config)
        query_params["task"] = task
        query_params["upload_id"] = str(prepare_resp.upload_id)

        train_set_uid = cls._fit(
            form_data=form_data,
            query_params=query_params,
            timeout=client_options.timeout,
            extra_headers=client_options.extra_headers,
        )

        # Note: at the moment we do prevent re-uploading but we do not prevent re-fitting
        # as the TrainSet.status is not implemented yet.
        return train_set_uid

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
        form_data: dict[str, Any],
        query_params: dict[str, Any],
        timeout: float | None,
        extra_headers: dict[str, str] | None,
    ) -> str:
        with cls.httpx_client.stream(
            "POST",
            url="/fit/",
            data=form_data,
            params=query_params,
            timeout=timeout,
            headers=extra_headers,
        ) as response:
            cls._validate_response(response, "fit")
            train_set_uid = None
            for line in response.iter_lines():
                # Handle responses that might be missing the status field
                try:
                    event = ResponseEventAdapter.validate_json(line)
                except (ValidationError, ValueError, json.JSONDecodeError) as e:
                    # If validation fails, try to infer the event type and add missing status field
                    raw_data = json.loads(line)
                    if "train_set_uid" in raw_data and "status" not in raw_data:
                        # This looks like a FitCompleteEvent missing the status field
                        raw_data["status"] = "complete"
                        event = FitCompleteEvent(**raw_data)
                    else:
                        # Re-raise the original error if we can't handle it
                        raise e
                if isinstance(event, FitCompleteEvent):
                    train_set_uid = event.train_set_uid
                elif isinstance(event, FitErrorEvent):
                    # Handle structured error events
                    raise RuntimeError(f"Error from server: {event.message}")

        if not train_set_uid:
            raise RuntimeError("Error during fit. No valid model received.")
        return train_set_uid

    @classmethod
    def _prepare_train_set_upload(
        cls,
        req: PrepareTrainSetUploadRequest,
        extra_headers: dict[str, str] | None = None,
    ) -> PrepareTrainSetUploadResponse | DuplicateFilesUploadedResponse:
        resp = cls.httpx_client.post(
            url="/tabpfn/prepare_train_set_upload/",
            json=req.model_dump(),
            headers=extra_headers,
        )
        if resp.status_code == 409:
            return DuplicateFilesUploadedResponse.model_validate(resp.json()["detail"])
        cls._validate_response(resp, "prepare_train_set_upload")
        return PrepareTrainSetUploadResponse.model_validate(resp.json())

    @classmethod
    def _upload_to_gcs(
        cls, dataset: str, data: bytes, crc32c_hash: str, info: FileUploadInfo
    ) -> None:
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
                crc32c_hash=crc32c_hash,
            )
            return

        # CRC32C hashes are computed to verify the integrity of the uploaded data.
        # When uploading multiple chunks we want to compute per-chunk CRC32C hashes.
        # We compute them once so retries don't recalculate.
        chunk_hashes = [_get_crc32c_hash(c) for c in chunks]

        with ThreadPoolExecutor(
            max_workers=min(_DEFAULT_CHUNK_UPLOAD_PARALLELISM, num_chunks)
        ) as pool:
            futures = {
                pool.submit(
                    cls._upload_single_chunk,
                    url=info.signed_urls[i],
                    chunk=chunks[i],
                    headers=info.required_headers,
                    dataset=dataset,
                    crc32c_hash=chunk_hashes[i],
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
        crc32c_hash: str,
        chunk_index: int = 0,
    ) -> None:
        resp = cls.httpx_client.put(
            url,
            content=chunk,
            headers={**headers, "x-goog-hash": f"crc32c={crc32c_hash}"},
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

    @classmethod
    @backoff.on_exception(
        backoff.expo,
        (
            httpx.ConnectError,
            httpx.TimeoutException,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.RemoteProtocolError,
            GCPOverloaded,
            RetryableServerError,
        ),
        max_tries=6,
        base=2,
        max_value=120,
        logger=logger,
        on_backoff=_on_backoff,
        on_giveup=_on_giveup,
    )
    def predict(
        cls,
        train_set_uid: str,
        x_test,
        model_type: ModelType,
        task: Literal["classification", "regression"],
        predict_params: Union[dict, None] = None,
        tabpfn_config: Union[dict, None] = None,
        X_train=None,
        y_train=None,
        client_options: ClientOptions | None = None,
    ) -> PredictionResult:
        """
        Predict the class labels for the provided data (test set).

        Parameters
        ----------
        train_set_uid : str
            The unique ID of the train set in the server.
        x_test : array-like of shape (n_samples, n_features)
            The test input.

        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            The predicted class labels.
        """
        client_options = client_options or ClientOptions()

        tabpfn_config = deepcopy(tabpfn_config)

        x_test_serialized = common_utils.serialize_to_csv_formatted_bytes(x_test)

        # Build unified params for tabpfn_config and tabpfn_systems
        params = cls._build_tabpfn_params(tabpfn_config)
        params.update(
            {
                "train_set_uid": train_set_uid,
                "task": task,
                "predict_params": json.dumps(predict_params),
            }
        )

        # Extract tabpfn_systems for hashing
        paper_version, tabpfn_systems, _, _ = cls._process_tabpfn_config(tabpfn_config)

        # In the arguments for hashing, include train_set_uid for the case that the same test set was previously used
        # with different train set. Include access token for the case that a user uses different accounts.
        (
            cached_test_set_uid,
            dataset_hash,
        ) = cls.dataset_uid_cache_manager.get_dataset_uid(
            x_test_serialized,
            train_set_uid,
            # Note: This may not be strictly necessary for predict, but keep it for safety.
            model_type.name,
            cls._access_token,
            "_".join(tabpfn_systems),
        )

        # Send prediction request. Loop two times, such that if anything cached is not correct
        # anymore, there is a second iteration where the datasets are uploaded.
        results = None

        # Store metadata about the prediction including TabPFN model version
        metadata = {}

        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                with cls._make_prediction_request(
                    cached_test_set_uid,
                    x_test_serialized,
                    params,
                    model_type=model_type,
                    timeout=client_options.timeout,
                    extra_headers=client_options.extra_headers,
                ) as response:
                    cls._validate_response(response, "predict")
                    # Handle updates from server
                    client = sseclient.SSEClient(response.iter_bytes())

                    progress_bar = None

                    def run_progress():
                        nonlocal progress_bar
                        progress_bar = tqdm(
                            range(int(duration * 10)),
                            desc="Processing",
                            total=int(duration * 10),
                            bar_format="{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]",
                        )
                        for _ in progress_bar:
                            time.sleep(0.1)

                    for event in client.events():
                        data = json.loads(event.data)
                        if data["event"] == "message":
                            print(data["data"])
                        elif data["event"] == "estimated_time_to_answer":
                            duration = float(data["data"])
                            progress_thread = threading.Thread(target=run_progress)
                            progress_thread.daemon = True
                            progress_thread.start()
                        elif data["event"] == "result":
                            results = data["data"]
                            # Extract metadata from the response
                            metadata = data.get("metadata", {})
                            if progress_bar:
                                progress_bar.n = progress_bar.total
                                progress_bar.refresh()
                                progress_bar.close()
                        elif data["event"] == "error":
                            if data["error_class"] == "GCPOverloaded":
                                raise GCPOverloaded(data["detail"])
                            elif data["error_class"] == "ValueError":
                                raise ValueError(data["detail"])
                            else:
                                raise RuntimeError(
                                    data["error_class"] + ": " + data["detail"]
                                )
                break
            except RuntimeError as e:
                error_message = str(e)
                if (
                    "Invalid train or test set uid" in error_message
                    and attempt < max_attempts - 1
                ):
                    # Retry by re-uploading the train set
                    cls.dataset_uid_cache_manager.delete_uid(train_set_uid)
                    if X_train is None or y_train is None:
                        raise RuntimeError(
                            "Train set data is required to re-upload but was not provided."
                        )
                    # I don't think this check is necessary anymore, but I want
                    # to understand why the non-thinking models needed this
                    # re-upload feature in the first place and test that
                    # thinking actually works before removing it.
                    if model_type == ModelType.TABPFN_R:
                        raise NotImplementedError(
                            "Automatically re-uploading the train set is not supported for thinking mode. Please call fit()."
                        )
                    train_set_uid = cls.fit(
                        X_train,
                        y_train,
                        model_type=model_type,
                        tabpfn_config=tabpfn_config,
                        task=task,
                    )
                    params["train_set_uid"] = train_set_uid
                    cached_test_set_uid = None
                else:
                    raise  # Re-raise the exception if it's not recoverable

        else:
            raise RuntimeError(
                f"Failed to get prediction after {max_attempts} attempts."
            )

        # The response from the predict API always returns a dictionary with the task as the key.
        # This is just s.t. we do not confuse the tasks, as they both use the same API endpoint.
        # That is why below we use the task as the key to access the response.
        result = results[task]
        test_set_uid = results["test_set_uid"]

        if cached_test_set_uid is None:
            cls.dataset_uid_cache_manager.add_dataset_uid(dataset_hash, test_set_uid)

        if not isinstance(result, dict):
            result = np.array(result)
        else:
            for k in result:
                if isinstance(result[k], list):
                    result[k] = np.array(result[k])

        return PredictionResult(result, metadata)

    @classmethod
    def _make_prediction_request(
        cls,
        test_set_uid,
        x_test_serialized,
        params,
        model_type: ModelType,
        timeout: float | None,
        extra_headers: dict[str, str],
    ):
        """
        Helper function to make the prediction request to the server.
        """
        if model_type == ModelType.TABPFN:
            url = cls.server_endpoints.predict.path
        elif model_type == ModelType.TABPFN_R:
            url = cls.server_endpoints.predict_reasoning.path
        else:
            raise RuntimeError(f"Unknown model type {model_type}")

        if test_set_uid:
            params = params.copy()
            params["test_set_uid"] = test_set_uid
            response = cls.httpx_client.stream(
                method="post",
                url=url,
                params=params,
                headers=extra_headers,
                timeout=timeout,
            )
        else:
            response = cls.httpx_client.stream(
                method="post",
                url=url,
                params=params,
                files=common_utils.to_httpx_post_file_format(
                    [("x_file", "x_test_filename", x_test_serialized)]
                ),
                headers=extra_headers,
                timeout=timeout,
            )
        return response

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
