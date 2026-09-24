"""`ClientOptions.with_download_uri` fetches the prediction from a signed URL."""

from __future__ import annotations

import json
import time
from typing import Any, Iterator
from unittest.mock import patch
from uuid import UUID

import httpx
import numpy as np
import pytest

from tabpfn_client.api_models import (
    ClassifierConfig,
    GetSettingsResponse,
    RegressorConfig,
    RegressorOutputType,
    RegressorPredictParams,
)
from tabpfn_client.client import ServiceClient
from tabpfn_client.models import ClientOptions, PredictionResult
from tests.mock_tabpfn_server import MockTabPFNServer
from tests.unit.test_client import _api_settings_payload

FIT_ID = UUID("00000000-0000-0000-0000-000000000002")
DOWNLOAD_URL = "https://storage.example/predictions/result.json?X-Goog-Signature=abc"
N_ROWS = 20
X_TEST = np.random.RandomState(0).rand(N_ROWS, 3)
OPT_IN = ClientOptions(with_download_uri=True)


def _metadata(task: str) -> dict[str, Any]:
    return {
        "task": task,
        "package_version": "0.0.0",
        "tabpfn_config": {},
        "test_set_num_rows": N_ROWS,
        "test_set_num_cols": 3,
    }


def _uri_body(task: str) -> dict[str, Any]:
    return {
        "prediction_uri": DOWNLOAD_URL,
        "prediction_uri_expires_in_secs": 1800,
        "metadata": _metadata(task),
    }


@pytest.fixture
def authorized_client() -> Iterator[None]:
    ServiceClient.reset_authorization()
    ServiceClient._api_settings = GetSettingsResponse(**_api_settings_payload())
    ServiceClient._api_settings_ts = time.monotonic()
    ServiceClient.authorize("dummy_token")
    try:
        yield
    finally:
        ServiceClient.reset_authorization()
        ServiceClient._api_settings = None
        ServiceClient._api_settings_ts = 0.0


@pytest.fixture
def mock_server(authorized_client: None) -> Iterator[MockTabPFNServer]:
    with MockTabPFNServer() as server:
        assert server.router is not None
        server.router.post("/tabpfn/prepare_test_set_upload").respond(
            200,
            json={
                "test_set_upload_id": "00000000-0000-0000-0000-000000000003",
                "x_test_info": {
                    "signed_urls": ["https://upload.example/x_test"],
                    "expires_at": 1_700_000_000.0,
                    "required_headers": {},
                },
            },
        )
        yield server


def _predict(
    task_config, client_options: ClientOptions | None = None
) -> PredictionResult:
    with patch.object(ServiceClient, "_upload_to_gcs"):
        return ServiceClient.predict(
            fitted_train_set_id=FIT_ID,
            x_test=X_TEST,
            task_config=task_config,
            client_options=client_options,
        )


def test_choice_is_left_to_the_server_by_default(mock_server):
    predict_route = mock_server.router.post("/tabpfn/predict").respond(
        200, json={"prediction": [0] * N_ROWS, "metadata": _metadata("classification")}
    )

    result = _predict(ClassifierConfig())

    sent = json.loads(predict_route.calls.last.request.content)
    assert "with_download_uri" not in sent
    np.testing.assert_array_equal(result.y_pred, np.zeros(N_ROWS, dtype=int))


def test_opting_out_sends_false(mock_server):
    predict_route = mock_server.router.post("/tabpfn/predict").respond(
        200, json={"prediction": [0] * N_ROWS, "metadata": _metadata("classification")}
    )

    _predict(ClassifierConfig(), ClientOptions(with_download_uri=False))

    sent = json.loads(predict_route.calls.last.request.content)
    assert sent["with_download_uri"] is False


def test_inline_that_does_not_fit_raises_the_server_message(mock_server):
    detail = (
        "The prediction is too large to return inline. "
        "Request it with `with_download_uri=True` to receive a signed download URL instead."
    )
    mock_server.router.post("/tabpfn/predict").respond(
        422, json={"message": detail, "error_code": "VALIDATION_ERROR"}
    )

    with pytest.raises(RuntimeError, match="with_download_uri=True"):
        _predict(ClassifierConfig(), ClientOptions(with_download_uri=False))


def test_server_may_answer_with_a_url_unasked(mock_server):
    # Left unset, the server picks the transport, so the client must follow
    # the body it gets rather than the flag it sent.
    mock_server.router.post("/tabpfn/predict").respond(
        200, json=_uri_body("classification")
    )
    mock_server.router.get(DOWNLOAD_URL).respond(200, json=[1] * N_ROWS)

    result = _predict(ClassifierConfig())

    np.testing.assert_array_equal(result.y_pred, np.ones(N_ROWS, dtype=int))


def test_downloads_the_prediction_from_the_signed_url(mock_server):
    predict_route = mock_server.router.post("/tabpfn/predict").respond(
        200, json=_uri_body("classification")
    )
    download_route = mock_server.router.get(DOWNLOAD_URL).respond(
        200, json=[1] * N_ROWS
    )

    result = _predict(ClassifierConfig(), OPT_IN)

    sent = json.loads(predict_route.calls.last.request.content)
    assert sent["with_download_uri"] is True
    # The bearer token belongs to the API, not to the object store.
    assert "authorization" not in download_route.calls.last.request.headers
    np.testing.assert_array_equal(result.y_pred, np.ones(N_ROWS, dtype=int))
    assert result.metadata["task"] == "classification"


def test_full_output_download_matches_the_inline_parsing(mock_server):
    mock_server.router.post("/tabpfn/predict").respond(
        200, json=_uri_body("regression")
    )
    # `null` stands in for -inf on the wire, exactly as in inline responses, so
    # the downloaded body must go through the same NaN conversion.
    mock_server.router.get(DOWNLOAD_URL).respond(
        200,
        json={
            "mean": [0.5] * N_ROWS,
            "logits": [[None, 0.1, 0.2]] * N_ROWS,
            "borders": [0.0, 1.0, 2.0, 3.0],
        },
    )

    result = _predict(
        RegressorConfig(
            predict_params=RegressorPredictParams(output_type=RegressorOutputType.FULL)
        ),
        OPT_IN,
    )

    y_pred = result.y_pred
    assert isinstance(y_pred, dict)
    assert y_pred["logits"].shape == (N_ROWS, 3)
    assert np.isnan(y_pred["logits"][0, 0])
    assert y_pred["logits"].dtype == float


def test_failed_download_raises(mock_server):
    mock_server.router.post("/tabpfn/predict").respond(
        200, json=_uri_body("classification")
    )
    mock_server.router.get(DOWNLOAD_URL).respond(403, text="expired")

    with pytest.raises(RuntimeError, match="download permanently failed"):
        _predict(ClassifierConfig(), OPT_IN)


def test_transient_download_error_is_retried(mock_server):
    mock_server.router.post("/tabpfn/predict").respond(
        200, json=_uri_body("classification")
    )
    download_route = mock_server.router.get(DOWNLOAD_URL)
    download_route.side_effect = [
        httpx.Response(503, text="try again"),
        httpx.Response(200, json=[1] * N_ROWS),
    ]

    result = _predict(ClassifierConfig(), OPT_IN)

    assert download_route.call_count == 2
    np.testing.assert_array_equal(result.y_pred, np.ones(N_ROWS, dtype=int))
