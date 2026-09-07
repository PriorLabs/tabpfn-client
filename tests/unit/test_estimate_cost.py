import json
from types import SimpleNamespace
from typing import Literal, cast
from unittest.mock import Mock

import httpx
import numpy as np
import pandas as pd
import pytest

from tabpfn_client import estimate_cost
from tabpfn_client.config import Config
from tabpfn_client.api_models import EstimateCostResponse
from tabpfn_client.client import ServiceClient
from tabpfn_client.errors import RetryableServerError


@pytest.fixture
def transport(monkeypatch):
    handler = Mock()
    monkeypatch.setattr(Config, "is_initialized", True)
    monkeypatch.setattr(ServiceClient, "_access_token", "estimate-test-token")
    with httpx.Client(
        base_url="https://estimate.invalid", transport=httpx.MockTransport(handler)
    ) as client:
        monkeypatch.setattr(ServiceClient, "httpx_client", client)
        yield handler


def quote(request):
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "estimated_cost": 20284,
            "pricing_version": "quota_v3",
            "inputs": {
                "model_version": "v3",
                "n_estimators": 8,
                "thinking_effort": None,
                **body,
            },
        },
    )


@pytest.mark.parametrize("as_frame", [False, True])
def test_estimate_posts_only_shape_and_options(transport, as_frame):
    transport.side_effect = quote
    X_train = np.zeros((3, 2))
    X_test = np.zeros((4, 2))
    if as_frame:
        X_train = pd.DataFrame(X_train)
        X_test = pd.DataFrame(X_test)
    result = estimate_cost(X_train, X_test)
    assert isinstance(result, EstimateCostResponse)
    assert result.estimated_cost == 20284
    assert result.inputs.n_estimators == 8
    transport.assert_called_once()
    request = transport.call_args.args[0]
    assert request.method == "POST"
    assert request.url.path == "/tabpfn/estimate_cost"
    assert request.headers["Authorization"] == "Bearer estimate-test-token"
    assert json.loads(request.content) == {
        "train_rows": 3,
        "test_rows": 4,
        "raw_columns": 2,
        "operation": "predict",
    }


@pytest.mark.parametrize(
    "operation", ["thinking_fit", "thinking_predict", "cache_predict"]
)
def test_options_and_shape_without_reading_array_values(
    transport,
    operation: Literal["thinking_fit", "thinking_predict", "cache_predict"],
):
    transport.side_effect = quote
    # No contents, array conversion, or upload methods exist on this object.
    X = cast(np.ndarray, SimpleNamespace(shape=(100000, 100)))
    effort: Literal["high"] | None = "high" if operation == "thinking_fit" else None
    estimate_cost(
        X,
        model_version="v3.5",
        operation=operation,
        n_estimators=4,
        thinking_effort=effort,
    )
    body = json.loads(transport.call_args.args[0].content)
    assert body == {
        "train_rows": 100000,
        "test_rows": 0,
        "raw_columns": 100,
        "operation": operation,
        "model_version": "v3.5",
        "n_estimators": 4,
        **({"thinking_effort": "high"} if operation == "thinking_fit" else {}),
    }


@pytest.mark.parametrize(
    "X_train,X_test,options",
    [
        (np.zeros(3), None, {}),
        (np.zeros((0, 2)), None, {}),
        (np.zeros((2, 0)), None, {}),
        (np.zeros((2, 3)), np.zeros((2, 4)), {}),
        (np.zeros((2, 3)), np.zeros(2), {}),
        (np.zeros((2, 3)), np.zeros((2, 3)), {"operation": "thinking_fit"}),
        (np.zeros((2, 3)), None, {"n_estimators": True}),
        (np.zeros((2, 3)), None, {"n_estimators": 1.5}),
        (np.zeros((2, 3)), None, {"n_estimators": 0}),
    ],
)
def test_invalid_dimensions_and_counts_do_not_make_http_requests(
    transport, X_train, X_test, options
):
    with pytest.raises(ValueError):
        estimate_cost(X_train, X_test, **options)
    transport.assert_not_called()


@pytest.mark.parametrize("status", [401, 422, 503])
def test_server_errors_surface_without_local_price_fallback(transport, status):
    transport.return_value = httpx.Response(
        status, json={"detail": "estimate unavailable"}
    )
    error = RetryableServerError if status == 503 else RuntimeError
    with pytest.raises(error, match="estimate unavailable"):
        estimate_cost(np.zeros((2, 3)))
    transport.assert_called_once()
