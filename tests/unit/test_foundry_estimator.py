#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Unit tests for the Azure AI Foundry estimators.

The endpoint validates the thinking-mode parameters server-side and answers
an out-of-range value with an opaque HTTP 424, so these tests pin the
client-side mirror of the accepted sets, and the error surfacing that makes
a server-side failure reportable (`error_code` / `trace_id`).
"""

import httpx
import numpy as np
import pytest

from tabpfn_client.foundry import (
    FoundryEndpointError,
    TabPFNClassifier,
    TabPFNRegressor,
)
from tabpfn_client.foundry.estimator import _build_request_body, _raise_for_status


URL = "https://example.inference.ml.azure.com/predict"
KEY = "test-key"


def _clf(**kwargs):
    return TabPFNClassifier(endpoint_url=URL, api_key=KEY, **kwargs)


def _reg(**kwargs):
    return TabPFNRegressor(endpoint_url=URL, api_key=KEY, **kwargs)


class TestThinkingValidation:
    @pytest.mark.parametrize("effort", ["medium", "high"])
    def test_accepts_supported_efforts(self, effort):
        assert _clf(thinking_effort=effort)._build_thinking_block() == {
            "thinking_effort": effort
        }

    @pytest.mark.parametrize("effort", ["low", "minimal", "none", "auto", "MEDIUM"])
    def test_rejects_unsupported_efforts(self, effort):
        # The endpoint rejects everything outside {medium, high}; fail locally
        # rather than surfacing an opaque 424.
        with pytest.raises(ValueError, match="thinking_effort"):
            _clf(thinking_effort=effort)

    @pytest.mark.parametrize("metric", ["accuracy", "log_loss", "balanced_accuracy"])
    def test_accepts_classification_metrics(self, metric):
        assert _clf(thinking_mode=True, thinking_metric=metric) is not None

    @pytest.mark.parametrize("metric", ["rmse", "mse", "mae", "r2"])
    def test_accepts_regression_metrics(self, metric):
        assert _reg(thinking_mode=True, thinking_metric=metric) is not None

    def test_rejects_metric_from_the_other_task(self):
        with pytest.raises(ValueError, match="thinking_metric"):
            _clf(thinking_mode=True, thinking_metric="rmse")
        with pytest.raises(ValueError, match="thinking_metric"):
            _reg(thinking_mode=True, thinking_metric="accuracy")

    @pytest.mark.parametrize("metric", ["roc_auc", "f1", "neg_log_loss", "bogus"])
    def test_rejects_unsupported_metrics(self, metric):
        with pytest.raises(ValueError, match="thinking_metric"):
            _clf(thinking_mode=True, thinking_metric=metric)

    def test_rejects_negative_timeout(self):
        with pytest.raises(ValueError, match="thinking_timeout_s"):
            _clf(thinking_mode=True, thinking_timeout_s=-1)

    def test_allows_zero_timeout(self):
        assert _clf(thinking_mode=True, thinking_timeout_s=0) is not None

    def test_thinking_mode_defaults_effort_to_medium(self):
        assert _clf(thinking_mode=True)._build_thinking_block() == {
            "thinking_effort": "medium"
        }

    def test_thinking_block_empty_when_inactive(self):
        assert _clf()._build_thinking_block() == {}

    def test_thinking_implies_cache_fit_mode(self):
        assert _clf(thinking_mode=True)._effective_fit_mode == "fit_with_cache"
        assert _clf()._effective_fit_mode == "fit_preprocessors"

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"fit_mode": "fit_preprocessors", "use_kv_cache": True},
            {"fit_mode": "fit_preprocessors", "thinking_mode": True},
            {"fit_mode": "fit_preprocessors", "thinking_effort": "high"},
        ],
    )
    def test_rejects_conflicting_fit_mode(self, kwargs):
        with pytest.raises(ValueError, match="Conflicting settings"):
            _clf(**kwargs)


class TestRequestBody:
    def test_thinking_fields_go_at_the_top_level(self):
        # Nesting these under task_config / tabpfn_config / predict_params is
        # rejected by the endpoint the same way an unknown field is.
        body = _build_request_body(
            task="classification",
            tabpfn_config={},
            predict_params={"output_type": "preds"},
            X_test=[[1.0]],
            X_train=[[0.0]],
            y_train=[0],
            thinking_block={"thinking_effort": "high", "thinking_timeout_s": 30.0},
        )
        assert body["thinking_effort"] == "high"
        assert body["thinking_timeout_s"] == 30.0
        assert "thinking_effort" not in body["task_config"]
        assert "thinking_effort" not in body["task_config"]["predict_params"]

    def test_y_train_is_two_dimensional_on_the_wire(self):
        body = _build_request_body(
            task="classification",
            tabpfn_config={},
            predict_params={},
            X_test=[[1.0]],
            X_train=[[0.0], [1.0]],
            y_train=np.array([0, 1]),
        )
        assert body["y_train"] == [[0], [1]]

    def test_cache_hit_sends_model_id_instead_of_training_data(self):
        body = _build_request_body(
            task="classification",
            tabpfn_config={},
            predict_params={},
            X_test=[[1.0]],
            X_train=[[0.0]],
            y_train=[0],
            cached_model_id="abc-123",
        )
        assert body["context"] == {"model_id": "abc-123"}
        assert "x_train" not in body and "y_train" not in body


class TestErrorSurfacing:
    def _response(self, status, json_body=None, text=None):
        request = httpx.Request("POST", URL)
        if json_body is not None:
            return httpx.Response(status, json=json_body, request=request)
        return httpx.Response(status, text=text or "", request=request)

    def test_passes_through_success(self):
        assert _raise_for_status(self._response(200, {"prediction": [1]})) is None

    def test_surfaces_error_code_and_trace_id(self):
        resp = self._response(
            424,
            {
                "message": "Inference failed; report the trace id to your provider.",
                "error_code": "INTERNAL_ERROR",
                "trace_id": "abc123",
            },
        )
        with pytest.raises(FoundryEndpointError) as exc:
            _raise_for_status(resp)

        err = exc.value
        assert err.error_code == "INTERNAL_ERROR"
        assert err.trace_id == "abc123"
        assert err.response.status_code == 424
        # the trace id is what the endpoint provider needs, so it must be in
        # the message a user actually sees
        assert "abc123" in str(err)
        assert "INTERNAL_ERROR" in str(err)
        assert "424" in str(err)

    def test_remains_catchable_as_httpx_error(self):
        resp = self._response(424, {"error_code": "X", "trace_id": "t"})
        with pytest.raises(httpx.HTTPStatusError):
            _raise_for_status(resp)

    def test_handles_non_json_error_body(self):
        resp = self._response(401, text="key_auth_access_denied")
        with pytest.raises(FoundryEndpointError) as exc:
            _raise_for_status(resp)
        assert exc.value.error_code is None
        assert "key_auth_access_denied" in str(exc.value)


class TestFit:
    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same number of samples"):
            _clf().fit(np.zeros((3, 2)), np.zeros(4))

    def test_fit_resets_cached_model_id(self):
        clf = _clf(use_kv_cache=True)
        clf.fit(np.zeros((3, 2)), np.array([0, 1, 0]))
        clf._cached_model_id = "stale-id"
        clf.fit(np.zeros((3, 2)), np.array([0, 1, 0]))
        assert clf._cached_model_id is None
