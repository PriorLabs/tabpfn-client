#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""`sample_weight` and `label_reliability` on the hosted, SageMaker and Foundry estimators."""

import json
from io import BytesIO
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest

from tabpfn_client import foundry, hosted, sagemaker
from tests.unit.test_hosted_estimator import Recorder

X = np.array([[0.1, 1.0], [0.2, 2.0], [0.8, 3.0], [0.9, 4.0]])
Y = np.array([0, 0, 1, 1])
WEIGHTS = {"sample_weight": [1.0, 2.0, 3.0, 4.0], "label_reliability": [1, 0.5, 1, 0]}


def _hosted_body(**kwargs: Any) -> Dict[str, Any]:
    recorder = Recorder()
    model = hosted.TabPFNClassifier(endpoint_url="https://example.test/predict")
    recorder.install(model)
    model.fit(X, Y, **kwargs)
    model.predict_proba(X[:2])
    return recorder.json_body()


def _sagemaker_body(**kwargs: Any) -> Dict[str, Any]:
    model = sagemaker.TabPFNClassifier(endpoint_name="ep", region_name="us-east-1")
    runtime = MagicMock()
    answer = json.dumps({"prediction": [[0.4, 0.6]] * 2, "metadata": {}}).encode()
    runtime.invoke_endpoint.return_value = {"Body": BytesIO(answer)}
    with patch.object(model, "_runtime_client", return_value=runtime):
        model.fit(X, Y, **kwargs)
        model.predict_proba(X[:2])
    return json.loads(runtime.invoke_endpoint.call_args.kwargs["Body"])


def _foundry_body(**kwargs: Any) -> Dict[str, Any]:
    sent: Dict[str, Any] = {}

    def answer(request: httpx.Request) -> httpx.Response:
        sent.update(json.loads(request.content))
        return httpx.Response(200, json={"prediction": [[0.4, 0.6]] * 2})

    model = foundry.TabPFNClassifier(
        endpoint_url="https://example.test/predict", api_key="key"
    )
    model._cached_client = httpx.Client(transport=httpx.MockTransport(answer))
    model.fit(X, Y, **kwargs)
    model.predict_proba(X[:2])
    return sent


BODIES = pytest.mark.parametrize(
    "body_of", [_hosted_body, _sagemaker_body, _foundry_body]
)


@BODIES
def test_the_weights_go_out_with_the_training_data(body_of: Any) -> None:
    body = body_of(**WEIGHTS)

    assert body["sample_weight"] == [1.0, 2.0, 3.0, 4.0]
    assert body["label_reliability"] == [1.0, 0.5, 1.0, 0.0]


@BODIES
def test_a_fit_without_weights_sends_no_weight_field(body_of: Any) -> None:
    assert not {"sample_weight", "label_reliability"} & set(body_of())


def test_the_parquet_payload_carries_the_weights_in_the_request_part() -> None:
    recorder = Recorder()
    model = hosted.TabPFNClassifier(
        endpoint_url="https://example.test/predict", payload_format="parquet"
    )
    recorder.install(model)

    model.fit(X, Y, sample_weight=WEIGHTS["sample_weight"])
    model.predict_proba(X[:2])

    config, frames = recorder.multipart()
    assert config["sample_weight"] == [1.0, 2.0, 3.0, 4.0]
    assert set(frames) == {"x_train", "y_train", "x_test"}


@pytest.mark.parametrize(
    "model",
    [
        hosted.TabPFNClassifier(endpoint_url="u", use_kv_cache=True),
        hosted.TabPFNClassifier(endpoint_url="u", fit_mode="fit_with_cache"),
        sagemaker.TabPFNClassifier(endpoint_name="ep", use_kv_cache=True),
        sagemaker.TabPFNClassifier(endpoint_name="ep", thinking_mode=True),
        foundry.TabPFNClassifier(endpoint_url="u", api_key="k", use_kv_cache=True),
        foundry.TabPFNClassifier(endpoint_url="u", api_key="k", thinking_mode=True),
    ],
)
def test_weights_are_refused_for_a_fit_the_endpoint_keeps(model: Any) -> None:
    with pytest.raises(ValueError, match="fit that the endpoint keeps"):
        model.fit(X, Y, sample_weight=WEIGHTS["sample_weight"])

    assert not hasattr(model, "X_train_")


@pytest.mark.parametrize(
    "model",
    [
        hosted.TabPFNClassifier(endpoint_url="u"),
        sagemaker.TabPFNClassifier(endpoint_name="ep"),
        foundry.TabPFNClassifier(endpoint_url="u", api_key="k"),
    ],
)
def test_weights_of_the_wrong_length_are_refused(model: Any) -> None:
    with pytest.raises(ValueError, match="one value per row"):
        model.fit(X, Y, sample_weight=[1.0, 2.0])
