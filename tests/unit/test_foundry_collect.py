#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Collecting a result the endpoint could not return in one request.

A fit that outlives the platform's request window answers HTTP 202 naming the
fit instead of the prediction. `predict*` collects it by re-sending that id,
so the caller still makes a single call — and the follow-ups carry only the id
and the test rows, never the training data again, which at a large context is
the whole payload on every attempt.
"""

import json

import httpx
import numpy as np
import pytest
import respx

from tabpfn_client.foundry import FoundryEndpointError, TabPFNClassifier
from tabpfn_client.foundry.estimator import _DEFAULT_RETRY_AFTER_S, _retry_after_s


URL = "https://example.inference.ml.azure.com/invocations"
KEY = "test-key"


def _fitted(**kwargs):
    """A thinking classifier with training data attached, plus that data."""
    clf = TabPFNClassifier(
        endpoint_url=URL, api_key=KEY, thinking_effort="medium", **kwargs
    )
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    y = (X[:, 0] > 0).astype(int)
    clf.fit(X, y)
    return clf, X


def _in_progress(model_id="mid-1", retry_after="0"):
    return httpx.Response(
        202,
        headers={"Retry-After": retry_after},
        json={
            "message": "The thinking fit for this dataset is still running.",
            "error_code": "FIT_IN_PROGRESS",
            "model_id": model_id,
        },
    )


def _done(model_id="mid-1"):
    return httpx.Response(
        200, json={"prediction": [0, 1], "metadata": {}, "model_id": model_id}
    )


def _body(call):
    return json.loads(call.request.content)


class TestCollect:
    @respx.mock
    def test_collects_the_result_after_a_202(self):
        route = respx.post(URL).mock(side_effect=[_in_progress(), _done()])
        clf, X = _fitted()

        assert list(clf.predict(X[:2])) == [0, 1]
        assert route.call_count == 2

    @respx.mock
    def test_resend_drops_the_training_data(self):
        """The reason the id is worth returning at all."""
        route = respx.post(URL).mock(side_effect=[_in_progress(), _done()])
        clf, X = _fitted()
        clf.predict(X[:2])

        first, second = _body(route.calls[0]), _body(route.calls[1])
        assert "x_train" in first and "y_train" in first
        assert "x_train" not in second and "y_train" not in second
        assert second["context"] == {"model_id": "mid-1"}
        assert "x_test" in second

    @respx.mock
    def test_several_202s_are_all_collected(self):
        route = respx.post(URL).mock(
            side_effect=[_in_progress(), _in_progress(), _in_progress(), _done()]
        )
        clf, X = _fitted()

        assert list(clf.predict(X[:2])) == [0, 1]
        assert route.call_count == 4

    @respx.mock
    def test_202_without_a_model_id_is_reported(self):
        """Nothing to collect with, so say so rather than looping forever."""
        respx.post(URL).mock(
            return_value=httpx.Response(
                202, json={"message": "still running", "error_code": "FIT_IN_PROGRESS"}
            )
        )
        clf, X = _fitted()

        with pytest.raises(FoundryEndpointError, match="without a model_id"):
            clf.predict(X[:2])

    @respx.mock
    def test_gives_up_once_the_collect_budget_is_spent(self):
        respx.post(URL).mock(return_value=_in_progress())
        clf, X = _fitted(collect_timeout_s=0)

        with pytest.raises(TimeoutError, match="collect_timeout_s"):
            clf.predict(X[:2])

    def test_rejects_a_negative_collect_budget(self):
        with pytest.raises(ValueError, match="collect_timeout_s"):
            TabPFNClassifier(endpoint_url=URL, api_key=KEY, collect_timeout_s=-1)


class TestRetryAfter:
    @staticmethod
    def _resp(**kwargs):
        return httpx.Response(202, request=httpx.Request("POST", URL), **kwargs)

    def test_defaults_when_the_endpoint_says_nothing(self):
        assert _retry_after_s(self._resp()) == _DEFAULT_RETRY_AFTER_S

    def test_honours_delta_seconds(self):
        assert _retry_after_s(self._resp(headers={"Retry-After": "3"})) == 3.0

    def test_falls_back_on_an_http_date(self):
        # Only the delta-seconds form is honoured; an unparseable value must not
        # fail a collect that is otherwise progressing.
        stamp = "Wed, 21 Oct 2026 07:28:00 GMT"
        assert _retry_after_s(self._resp(headers={"Retry-After": stamp})) == (
            _DEFAULT_RETRY_AFTER_S
        )

    def test_clamps_a_negative_delta(self):
        assert _retry_after_s(self._resp(headers={"Retry-After": "-5"})) == 0.0
