#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Unit tests for the self-hosted endpoint estimators.

Cover the wire format (JSON stays the default, and `payload_format="parquet"`
moves the datasets into multipart file parts), the `use_kv_cache` fast path,
and the sklearn estimator contract.
"""

import io
import json
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pytest
import numpy as np
import pandas as pd

from tabpfn_client.hosted import TabPFNClassifier, TabPFNRegressor


URL = "https://example.test/predict"

X = pd.DataFrame({"a": [0.1, 0.2, 0.8, 0.9], "b": [1.0, np.nan, 3.0, 4.0]})
X_COMPLETE = X.fillna(2.0)
Y = pd.Series([0, 0, 1, 1])


class Recorder:
    """Captures the outgoing request and answers with a canned prediction."""

    def __init__(
        self,
        prediction: Any = None,
        model_id: Optional[str] = None,
        statuses: Optional[List[int]] = None,
    ):
        self.prediction = [[0.4, 0.6]] * 2 if prediction is None else prediction
        self.model_id = model_id
        # Status code per request, in order; anything past the end answers 200.
        self.statuses = list(statuses or [])
        self.requests: List[httpx.Request] = []

    def install(self, estimator: Any) -> None:
        """Point the estimator's cached client at this recorder."""
        estimator._cached_client = httpx.Client(transport=httpx.MockTransport(self))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status = self.statuses.pop(0) if self.statuses else 200
        if status != 200:
            return httpx.Response(status, json={"detail": "unknown model_id"})
        body: Dict[str, Any] = {"prediction": self.prediction}
        if self.model_id is not None:
            body["model_id"] = self.model_id
        return httpx.Response(200, json=body)

    def sent(self) -> httpx.Request:
        assert self.requests, "no request was sent"
        return self.requests[-1]

    def bodies(self) -> List[Dict[str, Any]]:
        """Every JSON body sent so far, oldest first."""
        return [json.loads(r.content) for r in self.requests]

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


class TestKvCache:
    def test_is_off_by_default(self):
        """Without the flag, every predict re-sends the training data."""
        recorder = Recorder(model_id="mid")
        model = _classifier(recorder)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])
        model.predict_proba(X_COMPLETE.iloc[:2])

        assert all("X_train" in body for body in recorder.bodies())
        assert not any("context" in body for body in recorder.bodies())

    def test_reuses_the_model_id_after_the_first_predict(self):
        recorder = Recorder(model_id="mid")
        model = _classifier(recorder, use_kv_cache=True)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])
        model.predict_proba(X_COMPLETE.iloc[:2])
        model.predict(X_COMPLETE.iloc[:2])

        first, *rest = recorder.bodies()
        assert "X_train" in first and "context" not in first
        for body in rest:
            assert body["context"] == {"model_id": "mid"}
            assert "X_train" not in body and "y_train" not in body

    def test_implies_fit_with_cache(self):
        recorder = Recorder(model_id="mid")
        model = _classifier(recorder, use_kv_cache=True)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        assert recorder.bodies()[0]["task_config"]["tabpfn_config"]["fit_mode"] == (
            "fit_with_cache"
        )

    def test_explicit_fit_mode_wins(self):
        recorder = Recorder(model_id="mid")
        model = _classifier(recorder, use_kv_cache=True, fit_mode="batched")
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        assert (
            recorder.bodies()[0]["task_config"]["tabpfn_config"]["fit_mode"]
            == "batched"
        )

    def test_refit_invalidates_the_cached_id(self):
        recorder = Recorder(model_id="mid")
        model = _classifier(recorder, use_kv_cache=True)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        assert all("X_train" in body for body in recorder.bodies())

    def test_eviction_surfaces_rather_than_refitting(self):
        """An evicted id is the caller's to handle, as in the other backends."""
        recorder = Recorder(model_id="mid", statuses=[200, 404])
        model = _classifier(recorder, use_kv_cache=True)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        with pytest.raises(httpx.HTTPStatusError):
            model.predict_proba(X_COMPLETE.iloc[:2])
        assert len(recorder.requests) == 2

    def test_constructor_model_id_still_skips_fit(self):
        recorder = Recorder()
        model = _classifier(recorder, model_id="abc")
        model.predict_proba(X_COMPLETE.iloc[:2])

        body = recorder.bodies()[0]
        assert body["context"] == {"model_id": "abc"}
        assert "X_train" not in body


class TestNonFiniteJson:
    def test_missing_and_infinite_values_survive_the_json_path(self):
        """Explainers mask features with +inf; httpx's `json=` would reject it."""
        recorder = Recorder()
        model = _classifier(recorder)
        model.fit(X, Y)
        masked = X.iloc[:2].copy()
        masked["a"] = np.inf
        model.predict_proba(masked)

        body = recorder.bodies()[0]
        assert np.isnan(body["X_train"][1][1])
        assert body["X_test"][0][0] == float("inf")


class TestSklearnContract:
    def test_estimator_type_is_detected(self):
        from sklearn.base import is_classifier, is_regressor

        assert is_classifier(TabPFNClassifier(endpoint_url=URL))
        assert is_regressor(TabPFNRegressor(endpoint_url=URL))

    def test_use_kv_cache_is_a_constructor_param(self):
        """`_cached_model_id` must stay off get_params, so clone() drops it."""
        from sklearn.base import clone

        model = TabPFNClassifier(endpoint_url=URL, use_kv_cache=True)
        cloned = clone(model)
        assert isinstance(cloned, TabPFNClassifier)
        assert model.get_params()["use_kv_cache"] is True
        assert "_cached_model_id" not in model.get_params()
        assert cloned.get_params() == model.get_params()
