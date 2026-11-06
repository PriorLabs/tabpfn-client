#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Unit tests for the self-hosted endpoint estimators.

Cover the wire format (JSON stays the default, and `payload_format="parquet"`
moves the datasets into multipart file parts), the `use_kv_cache` fast path,
async jobs, the routing headers, and the sklearn estimator contract. Async
jobs run on a fake clock, so nothing sleeps.
"""

import io
import json
import pickle
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import pytest
import numpy as np
import pandas as pd
from sklearn.base import clone

from tabpfn_client.hosted import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.hosted.estimator import _STATE_VERSION
from tests.unit.test_client import _FakeTime


URL = "https://example.test/predict"
# A model_id as the endpoint mints one: the header must parse as a UUID.
MID = "0123456789abcdef0123456789abcdef"

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
        answers: Optional[List[Union[httpx.Response, Exception]]] = None,
    ):
        self.prediction = [[0.4, 0.6]] * 2 if prediction is None else prediction
        self.model_id = model_id
        # Status code per request, in order; anything past the end answers 200.
        self.statuses = list(statuses or [])
        # Served as they are before `statuses`, in order; an exception is raised.
        self.answers = list(answers or [])
        self.requests: List[httpx.Request] = []

    def install(self, estimator: Any) -> None:
        """Point the estimator's cached client at this recorder."""
        estimator._cached_client = httpx.Client(transport=httpx.MockTransport(self))

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.answers:
            answer = self.answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        status = self.statuses.pop(0) if self.statuses else 200
        if status != 200:
            return httpx.Response(status, json={"detail": "unknown model_id"})
        body: Dict[str, Any] = {"prediction": self.prediction}
        if self.model_id is not None:
            body["model_id"] = self.model_id
        return httpx.Response(200, json=body)

    def sent(self, index: int = -1) -> httpx.Request:
        assert self.requests, "no request was sent"
        return self.requests[index]

    def bodies(self) -> List[Dict[str, Any]]:
        """Every JSON body sent so far, oldest first."""
        return [json.loads(r.content) for r in self.requests]

    def content_type(self, index: int = -1) -> str:
        return self.sent(index).headers["content-type"]

    def json_body(self) -> Dict[str, Any]:
        return json.loads(self.sent().content)

    def multipart(
        self, index: int = -1
    ) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
        """The JSON `request` field and each Parquet part, by field name."""
        boundary = self.content_type(index).split("boundary=")[1]
        config: Dict[str, Any] = {}
        frames: Dict[str, pd.DataFrame] = {}
        for part in self.sent(index).content.split(f"--{boundary}".encode())[1:-1]:
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


def _response(status: int, retry_after: Optional[str] = None) -> httpx.Response:
    """An answer without a prediction, optionally asking to retry later."""
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    return httpx.Response(status, json={"detail": "not done"}, headers=headers)


def _accepted(retry_after: Optional[str] = "5") -> httpx.Response:
    """The endpoint's 202 for a job that still runs."""
    return _response(202, retry_after)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeTime:
    """Replace the estimator's clock, so a wait only advances it."""
    fake = _FakeTime()
    monkeypatch.setattr("tabpfn_client.hosted.estimator.time", fake)
    return fake


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


class TestModelVersion:
    def test_default_omits_the_field(self):
        """Compat with older containers: an unknown field would 422."""
        recorder = Recorder()
        model = _classifier(recorder)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        assert "model_version" not in recorder.json_body()

    def test_explicit_value_sits_at_the_top_level(self):
        """The serving container reads `req.model_version`, not under `tabpfn_config`."""
        recorder = Recorder()
        model = _classifier(recorder, model_version="v3.5")
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        body = recorder.json_body()
        assert body["model_version"] == "v3.5"
        assert "model_version" not in body["task_config"]["tabpfn_config"]

    def test_travels_on_the_parquet_payload_too(self):
        recorder = Recorder()
        model = _classifier(recorder, payload_format="parquet", model_version="v3")
        model.fit(X, Y)
        model.predict_proba(X.iloc[:2])

        config, _ = recorder.multipart()
        assert config["model_version"] == "v3"


class TestTabpfnSystems:
    def test_default_omits_the_field(self):
        """Compat with older containers: an unknown field would 422."""
        recorder = Recorder()
        model = _classifier(recorder)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        assert "tabpfn_systems" not in recorder.json_body()

    def test_explicit_value_sits_at_the_top_level(self):
        """The serving container reads `req.tabpfn_systems`, not under `tabpfn_config`."""
        recorder = Recorder()
        model = _classifier(recorder, tabpfn_systems=["preprocessing", "text"])
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        body = recorder.json_body()
        assert body["tabpfn_systems"] == ["preprocessing", "text"]
        assert "tabpfn_systems" not in body["task_config"]["tabpfn_config"]

    def test_empty_list_is_sent(self):
        """`[]` asks for only minimal cleaning, so it must not read as unset."""
        recorder = Recorder(prediction=[1.5, 2.5])
        model = _regressor(recorder, tabpfn_systems=[])
        model.fit(X_COMPLETE, pd.Series([1.0, 2.0, 3.0, 4.0]))
        model.predict(X_COMPLETE.iloc[:2])

        assert recorder.json_body()["tabpfn_systems"] == []

    def test_travels_on_the_parquet_payload_too(self):
        recorder = Recorder()
        model = _classifier(
            recorder,
            payload_format="parquet",
            tabpfn_systems=["preprocessing", "text", "thinking"],
        )
        model.fit(X, Y)
        model.predict_proba(X.iloc[:2])

        config, _ = recorder.multipart()
        assert config["tabpfn_systems"] == ["preprocessing", "text", "thinking"]

    def test_is_sent_on_a_cached_predict(self):
        recorder = Recorder()
        model = _classifier(recorder, model_id="abc", tabpfn_systems=["preprocessing"])
        model.predict_proba(X_COMPLETE.iloc[:2])

        body = recorder.json_body()
        assert body["context"] == {"model_id": "abc"}
        assert body["tabpfn_systems"] == ["preprocessing"]

    def test_tuple_is_sent_as_a_list(self):
        recorder = Recorder()
        model = _classifier(recorder, tabpfn_systems=("preprocessing", "text"))
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        assert recorder.json_body()["tabpfn_systems"] == ["preprocessing", "text"]

    def test_leaves_order_to_the_container(self):
        """Order and dependencies are the container's to check, not the client's."""
        recorder = Recorder()
        model = _classifier(recorder, tabpfn_systems=["text", "preprocessing"])
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        assert recorder.json_body()["tabpfn_systems"] == ["text", "preprocessing"]

    def test_sends_other_values_unchanged(self):
        """A container may accept values beyond the known ones, so it decides."""
        recorder = Recorder()
        model = _classifier(recorder, tabpfn_systems=["custom"])
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        assert recorder.json_body()["tabpfn_systems"] == ["custom"]

    @pytest.mark.parametrize("systems", ["preprocessing", [1]])
    def test_invalid_value_raises_before_sending(self, systems: Any):
        """Checked on predict, which also covers the `model_id` path that skips fit."""
        recorder = Recorder()
        model = _classifier(recorder, model_id="abc", tabpfn_systems=systems)

        with pytest.raises(ValueError, match="list of system names"):
            model.predict_proba(X_COMPLETE.iloc[:2])
        assert recorder.requests == []

    def test_get_params_and_clone_keep_it(self):
        """Stored as passed for sklearn's clone; the caller's list is left alone."""
        recorder = Recorder()
        systems = ["preprocessing"]
        model = _classifier(recorder, tabpfn_systems=systems)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        cloned = clone(model)
        assert isinstance(cloned, TabPFNClassifier)
        assert model.get_params()["tabpfn_systems"] is systems
        assert cloned.get_params()["tabpfn_systems"] == ["preprocessing"]
        assert systems == ["preprocessing"]

    def test_older_pickle_without_the_field_still_predicts(self):
        """Estimators pickled by v0.7.1 or earlier have no `tabpfn_systems`."""
        model = TabPFNClassifier(endpoint_url=URL)
        model.fit(X_COMPLETE, Y)
        del model.tabpfn_systems
        restored = pickle.loads(pickle.dumps(model))
        recorder = Recorder()
        recorder.install(restored)
        restored.predict_proba(X_COMPLETE.iloc[:2])

        assert restored.get_params()["tabpfn_systems"] is None
        assert "tabpfn_systems" not in recorder.json_body()


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


class TestModelIdHeader:
    """A proxy routes by `X-TabPFN-Model-Id` without reading the body."""

    @pytest.mark.parametrize("payload_format", ["json", "parquet"])
    def test_constructor_model_id_is_sent(self, payload_format):
        recorder = Recorder()
        model = _classifier(recorder, model_id=MID, payload_format=payload_format)
        model.predict_proba(X_COMPLETE.iloc[:2])

        assert recorder.sent().headers["x-tabpfn-model-id"] == MID

    def test_cached_predict_sends_it(self):
        recorder = Recorder(model_id=MID)
        model = _classifier(recorder, use_kv_cache=True)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])
        model.predict_proba(X_COMPLETE.iloc[:2])

        first, cached = recorder.requests
        assert "x-tabpfn-model-id" not in first.headers
        assert cached.headers["x-tabpfn-model-id"] == MID

    def test_training_data_sends_none(self):
        """fit() supersedes the constructor id, in the header as in the body."""
        recorder = Recorder()
        model = _classifier(recorder, model_id=MID)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        assert "context" not in recorder.json_body()
        assert "x-tabpfn-model-id" not in recorder.sent().headers

    def test_extra_headers_cannot_override_it(self):
        recorder = Recorder()
        model = _classifier(
            recorder, model_id=MID, extra_headers={"x-tabpfn-model-id": "spoofed"}
        )
        model.predict_proba(X_COMPLETE.iloc[:2])

        assert recorder.sent().headers.get_list("x-tabpfn-model-id") == [MID]


class TestAsyncJobs:
    @pytest.mark.parametrize("payload_format", ["json", "parquet"])
    def test_is_off_by_default(self, clock, payload_format):
        recorder = Recorder()
        model = _classifier(recorder, payload_format=payload_format)
        model.fit(X, Y)
        model.predict_proba(X.iloc[:2])

        config = (
            recorder.multipart()[0]
            if payload_format == "parquet"
            else recorder.json_body()
        )
        assert "execution" not in config and "job_id" not in config
        assert "x-tabpfn-job-id" not in recorder.sent().headers
        assert len(recorder.requests) == 1
        assert clock.sleeps == []

    def test_polls_until_the_answer(self, clock):
        recorder = Recorder(answers=[_accepted("5"), _accepted("7")])
        model = _classifier(recorder, use_async=True)
        model.fit(X_COMPLETE, Y)
        proba = model.predict_proba(X_COMPLETE.iloc[:2])

        sync = _classifier(Recorder())
        sync.fit(X_COMPLETE, Y)
        np.testing.assert_array_equal(proba, sync.predict_proba(X_COMPLETE.iloc[:2]))

        submit, *polls = recorder.requests
        body = json.loads(submit.content)
        job_id = body["job_id"]
        assert body["execution"] == "async" and "X_train" in body
        # The endpoint takes only a UUID as a job id.
        assert uuid.UUID(job_id).hex == job_id
        assert submit.headers["x-tabpfn-job-id"] == job_id
        assert "x-tabpfn-model-id" not in submit.headers
        assert len(polls) == 2
        for poll in polls:
            assert json.loads(poll.content) == {"job_id": job_id}
            assert poll.headers["content-type"] == "application/json"
            assert poll.headers["x-tabpfn-job-id"] == job_id
        assert clock.sleeps == [5, 7]

    def test_parquet_submit_carries_the_job(self, clock):
        recorder = Recorder(answers=[_accepted()])
        model = _classifier(recorder, payload_format="parquet", use_async=True)
        model.fit(X, Y)
        model.predict_proba(X.iloc[:2])

        config, frames = recorder.multipart(0)
        job_id = config["job_id"]
        assert config["execution"] == "async"
        assert set(frames) == {"x_train", "y_train", "x_test"}
        assert recorder.sent(0).headers["x-tabpfn-job-id"] == job_id
        poll = recorder.sent()
        assert poll.headers["content-type"] == "application/json"
        assert json.loads(poll.content) == {"job_id": job_id}
        assert poll.headers["x-tabpfn-job-id"] == job_id

    @pytest.mark.parametrize("status", [503, 429])
    def test_busy_submit_is_sent_again_under_a_new_id(self, clock, status):
        """A refused submit registers no job, so its id names nothing."""
        recorder = Recorder(answers=[_response(status, "3"), _accepted()])
        model = _classifier(recorder, use_async=True)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        refused, submit, poll = recorder.bodies()
        assert refused["execution"] == submit["execution"] == "async"
        assert refused["job_id"] != submit["job_id"]
        assert poll == {"job_id": submit["job_id"]}
        assert recorder.sent(1).headers["x-tabpfn-job-id"] == submit["job_id"]
        assert clock.sleeps == [3, 5]

    @pytest.mark.parametrize("status", [503, 422, 413])
    def test_refused_submit_raises_at_once(self, clock, status):
        """Without `Retry-After` a refusal is final, e.g. a container without jobs."""
        recorder = Recorder(answers=[_response(status)])
        model = _classifier(recorder, use_async=True)
        model.fit(X_COMPLETE, Y)

        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            model.predict_proba(X_COMPLETE.iloc[:2])
        assert excinfo.value.response.status_code == status
        assert len(recorder.requests) == 1
        assert clock.sleeps == []

    def test_busy_poll_asks_again_for_the_same_job(self, clock):
        recorder = Recorder(answers=[_accepted(), _response(503, "2")])
        model = _classifier(recorder, use_async=True)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        submit, *polls = recorder.bodies()
        assert polls == [{"job_id": submit["job_id"]}] * 2
        assert clock.sleeps == [5, 2]

    def test_timed_out_poll_asks_again(self, clock):
        recorder = Recorder(answers=[_accepted(), httpx.ReadTimeout("timed out")])
        model = _classifier(recorder, use_async=True)
        model.fit(X_COMPLETE, Y)
        proba = model.predict_proba(X_COMPLETE.iloc[:2])

        np.testing.assert_allclose(proba, [[0.4, 0.6]] * 2)
        submit, *polls = recorder.bodies()
        assert polls == [{"job_id": submit["job_id"]}] * 2
        assert clock.sleeps == [5, 5]

    @pytest.mark.parametrize("status", [404, 504])
    def test_final_poll_error_raises_without_resubmitting(self, clock, status):
        """A lost job (404) or a kept error (504) is the caller's to handle."""
        recorder = Recorder(answers=[_accepted(), _response(status)])
        model = _classifier(recorder, use_async=True)
        model.fit(X_COMPLETE, Y)

        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            model.predict_proba(X_COMPLETE.iloc[:2])
        assert excinfo.value.response.status_code == status
        submit, poll = recorder.bodies()
        assert poll == {"job_id": submit["job_id"]}

    def test_gives_up_at_the_deadline(self, clock):
        recorder = Recorder(answers=[_accepted() for _ in range(5)])
        model = _classifier(recorder, use_async=True, async_timeout_s=12)
        model.fit(X_COMPLETE, Y)

        with pytest.raises(TimeoutError) as excinfo:
            model.predict_proba(X_COMPLETE.iloc[:2])
        job_id = recorder.bodies()[0]["job_id"]
        assert job_id in str(excinfo.value)
        assert "may still run" in str(excinfo.value)
        # Sent at 0, 5 and 10; the next one would go out at 15.
        assert len(recorder.requests) == 3
        assert clock.sleeps == [5, 5]

    def test_zero_timeout_sends_only_the_submit(self, clock):
        recorder = Recorder(answers=[_accepted()])
        model = _classifier(recorder, use_async=True, async_timeout_s=0)
        model.fit(X_COMPLETE, Y)

        with pytest.raises(TimeoutError):
            model.predict_proba(X_COMPLETE.iloc[:2])
        assert len(recorder.requests) == 1
        assert clock.sleeps == []

    @pytest.mark.parametrize(
        ("retry_after", "wait_s"),
        [(None, 5), ("soon", 5), ("Wed, 21 Oct 2026 07:28:00 GMT", 5), ("0", 1)],
        ids=["missing", "not_an_integer", "http_date", "zero"],
    )
    def test_wait_follows_retry_after(self, clock, retry_after, wait_s):
        recorder = Recorder(answers=[_accepted(retry_after)])
        model = _classifier(recorder, use_async=True)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])

        assert clock.sleeps == [wait_s]

    def test_kv_cache_names_the_model_on_submit_and_poll(self, clock):
        recorder = Recorder(model_id=MID, answers=[_accepted()])
        model = _classifier(recorder, use_kv_cache=True, use_async=True)
        model.fit(X_COMPLETE, Y)
        model.predict_proba(X_COMPLETE.iloc[:2])
        assert model.model_id_ == MID

        recorder.answers.append(_accepted())
        model.predict_proba(X_COMPLETE.iloc[:2])

        first_submit, first_poll, submit, poll = recorder.requests
        assert "x-tabpfn-model-id" not in first_submit.headers
        assert "x-tabpfn-model-id" not in first_poll.headers
        body = json.loads(submit.content)
        assert body["context"] == {"model_id": MID}
        assert "X_train" not in body
        assert submit.headers["x-tabpfn-model-id"] == MID
        assert poll.headers["x-tabpfn-model-id"] == MID

    def test_extra_headers_cannot_override_routing(self, clock):
        recorder = Recorder(answers=[_accepted()])
        model = _classifier(
            recorder,
            model_id=MID,
            api_key="secret",
            extra_headers={
                "x-tabpfn-job-id": "spoofed",
                "X-TABPFN-MODEL-ID": "spoofed",
                "content-type": "text/plain",
            },
            use_async=True,
        )
        model.predict_proba(X_COMPLETE.iloc[:2])

        submit, poll = recorder.requests
        job_id = json.loads(submit.content)["job_id"]
        for request in (submit, poll):
            assert request.headers.get_list("x-tabpfn-job-id") == [job_id]
            assert request.headers.get_list("x-tabpfn-model-id") == [MID]
        assert poll.headers.get_list("content-type") == ["application/json"]
        assert poll.headers["authorization"] == "Bearer secret"


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

    def test_async_params_are_off_by_default(self):
        params = TabPFNClassifier(endpoint_url=URL).get_params()
        assert params["use_async"] is False
        assert params["async_timeout_s"] == 3600.0

    def test_async_params_survive_clone_and_pickle(self, clock):
        recorder = Recorder(prediction=[1.5, 2.5], answers=[_accepted()])
        model = _regressor(recorder, use_async=True, async_timeout_s=30.0)
        model.fit(X_COMPLETE, pd.Series([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(model.predict(X_COMPLETE.iloc[:2]), [1.5, 2.5])

        for restored in (clone(model), pickle.loads(pickle.dumps(model))):
            assert isinstance(restored, TabPFNRegressor)
            assert restored.get_params()["use_async"] is True
            assert restored.get_params()["async_timeout_s"] == 30.0

    def test_unpickles_a_state_from_before_async(self):
        """v0.7.1 pickled neither async param, and no state version."""
        state = TabPFNClassifier(endpoint_url=URL).__getstate__()
        del state["use_async"], state["async_timeout_s"], state["_state_version"]
        sent_state = dict(state)
        restored = TabPFNClassifier.__new__(TabPFNClassifier)
        restored.__setstate__(state)

        assert state == sent_state
        assert "_state_version" not in vars(restored)
        assert restored.get_params()["use_async"] is False
        assert restored.get_params()["async_timeout_s"] == 3600.0
        assert isinstance(clone(restored), TabPFNClassifier)

        recorder = Recorder()
        recorder.install(restored)
        restored.fit(X_COMPLETE, Y)
        restored.predict_proba(X_COMPLETE.iloc[:2])

        assert "execution" not in recorder.json_body()
        assert len(recorder.requests) == 1

    def test_pickle_round_trip_keeps_the_version_off_the_estimator(self):
        model = TabPFNClassifier(endpoint_url=URL, use_async=True)
        assert model.__getstate__()["_state_version"] == _STATE_VERSION

        restored = pickle.loads(pickle.dumps(model))

        assert isinstance(restored, TabPFNClassifier)
        assert "_state_version" not in vars(restored)
        assert "_state_version" not in restored.get_params()
        assert restored.get_params() == model.get_params()

    def test_refuses_a_state_from_a_newer_version(self):
        state = TabPFNClassifier(endpoint_url=URL).__getstate__()
        state["_state_version"] = _STATE_VERSION + 1
        restored = TabPFNClassifier.__new__(TabPFNClassifier)

        with pytest.raises(ValueError, match="newer tabpfn-client"):
            restored.__setstate__(state)
