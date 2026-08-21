#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Unit tests for the self-hosted endpoint estimators.

Cover the wire format: JSON stays the default, and `payload_format="parquet"`
moves the datasets into multipart file parts.
"""

import io
import json
from typing import Any, Dict, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

from tabpfn_client.hosted import TabPFNClassifier, TabPFNRegressor


URL = "https://example.test/predict"

X = pd.DataFrame({"a": [0.1, 0.2, 0.8, 0.9], "b": [1.0, np.nan, 3.0, 4.0]})
X_COMPLETE = X.fillna(2.0)
Y = pd.Series([0, 0, 1, 1])


class Recorder:
    """Captures the outgoing request and answers with a canned prediction."""

    def __init__(self, prediction: Any = None, model_id: Optional[str] = None):
        self.prediction = [[0.4, 0.6]] * 2 if prediction is None else prediction
        self.model_id = model_id
        self.request: Optional[httpx.Request] = None

    def install(self, estimator: Any) -> None:
        """Point the estimator's cached client at this recorder."""
        estimator._cached_client = httpx.Client(transport=httpx.MockTransport(self))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request = request
        body: Dict[str, Any] = {"prediction": self.prediction}
        if self.model_id is not None:
            body["model_id"] = self.model_id
        return httpx.Response(200, json=body)

    def sent(self) -> httpx.Request:
        assert self.request is not None, "no request was sent"
        return self.request

    def content_type(self) -> str:
        return self.sent().headers["content-type"]

    def json_body(self) -> Dict[str, Any]:
        return json.loads(self.sent().content)

    def multipart(self) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
        """The JSON `request` field and each Parquet part, by field name."""
        boundary = self.content_type().split("boundary=")[1]
        config: Dict[str, Any] = {}
        frames: Dict[str, pd.DataFrame] = {}
        for part in self.sent().content.split(f"--{boundary}".encode())[1:-1]:
            head, payload = part.split(b"\r\n\r\n", 1)
            name = head.decode().split('name="')[1].split('"')[0]
            payload = payload.rstrip(b"\r\n")
            if name == "request":
                config = json.loads(payload)
            else:
                frames[name] = pd.read_parquet(io.BytesIO(payload))
        return config, frames


def _classifier(recorder: Recorder, **kwargs: Any) -> TabPFNClassifier:
    model = TabPFNClassifier(endpoint_url=URL, **kwargs)
    recorder.install(model)
    return model


def _regressor(recorder: Recorder, **kwargs: Any) -> TabPFNRegressor:
    model = TabPFNRegressor(endpoint_url=URL, **kwargs)
    recorder.install(model)
    return model


class TestJsonPayload:
    def test_is_the_default(self):
        recorder = Recorder()
        model = _classifier(recorder)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        assert recorder.content_type() == "application/json"
        body = recorder.json_body()
        assert body["task_config"]["task"] == "classification"
        assert len(body["X_train"]) == 4
        # y_train on the wire is 2D.
        assert body["y_train"] == [[0], [0], [1], [1]]


class TestParquetPayload:
    def test_sends_datasets_as_multipart_files(self):
        recorder = Recorder()
        model = _classifier(recorder, payload_format="parquet")
        model.fit(X, Y)
        model.predict_proba(X.iloc[:2])

        assert recorder.content_type().startswith("multipart/form-data")
        config, frames = recorder.multipart()
        assert config["task_config"]["task"] == "classification"
        assert set(frames) == {"x_train", "y_train", "x_test"}
        assert frames["y_train"].shape == (4, 1)
        assert list(frames["x_test"].columns) == ["a", "b"]

    def test_carries_missing_values(self):
        recorder = Recorder()
        model = _classifier(recorder, payload_format="parquet")
        model.fit(X, Y)
        model.predict_proba(X.iloc[:2])

        _, frames = recorder.multipart()
        assert frames["x_train"]["b"].isna().sum() == 1

    def test_cached_predict_omits_training_data(self):
        recorder = Recorder()
        model = _classifier(recorder, payload_format="parquet", model_id="abc")
        model.predict_proba(X.iloc[:2])

        config, frames = recorder.multipart()
        assert config["context"] == {"model_id": "abc"}
        assert set(frames) == {"x_test"}

    def test_captures_the_returned_model_id(self):
        recorder = Recorder(model_id="xyz")
        model = _classifier(
            recorder, payload_format="parquet", fit_mode="fit_with_cache"
        )
        model.fit(X, Y)
        model.predict_proba(X.iloc[:2])

        assert model.model_id_ == "xyz"

    def test_regressor_round_trips(self):
        recorder = Recorder(prediction=[1.5, 2.5])
        model = _regressor(recorder, payload_format="parquet")
        model.fit(X, pd.Series([1.0, 2.0, 3.0, 4.0]))

        np.testing.assert_allclose(model.predict(X.iloc[:2]), [1.5, 2.5])
        config, _ = recorder.multipart()
        assert config["task_config"]["task"] == "regression"

    def test_numpy_input_keeps_column_count(self):
        """Integer column names are coerced by `to_parquet`, as gapi relies on."""
        recorder = Recorder()
        model = _classifier(recorder, payload_format="parquet")
        model.fit(np.zeros((4, 3)), np.array([0, 0, 1, 1]))
        model.predict_proba(np.zeros((2, 3)))

        _, frames = recorder.multipart()
        assert list(frames["x_train"].columns) == ["0", "1", "2"]
