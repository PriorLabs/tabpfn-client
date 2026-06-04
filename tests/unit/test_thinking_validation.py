"""Validator contract for the thinking_* knobs on TabPFNClassifier/Regressor.

Pins the rule that thinking is enabled when either `thinking_mode=True` OR
`thinking_effort` is set, so callers can pass either or both without surprise.
"""

from typing import Any

import pytest

from tabpfn_client.estimator import (
    THINKING_TIMEOUT_MAX_S,
    validate_thinking_mode,
)


def _v(**overrides):
    args: dict[str, Any] = dict(
        thinking_mode=False,
        thinking_effort=None,
        thinking_timeout_s=None,
        thinking_metric=None,
        model_path="auto",
    )
    args.update(overrides)
    return validate_thinking_mode(**args)


class TestThinkingValidator:
    def test_neither_flag_is_off(self):
        # No flags: thinking off, no errors.
        _v()

    def test_thinking_mode_alone_is_on(self):
        # Just `thinking_mode=True` is enough; downstream defaults effort to "medium".
        _v(thinking_mode=True)

    def test_thinking_effort_alone_implies_on(self):
        # The whole point of this contract: setting thinking_effort enables
        # thinking even without thinking_mode=True.
        _v(thinking_effort="medium")
        _v(thinking_effort="high")

    def test_extra_knobs_with_thinking_effort_set_are_allowed(self):
        # If thinking is on (via either flag), the budget/metric knobs apply.
        _v(thinking_effort="high", thinking_timeout_s=60.0, thinking_metric="rmse")
        _v(thinking_mode=True, thinking_timeout_s=60.0, thinking_metric="rmse")

    def test_extra_knobs_without_thinking_are_rejected(self):
        # Knobs that only matter when thinking is on must error if neither flag is set.
        with pytest.raises(ValueError, match="thinking is enabled"):
            _v(thinking_timeout_s=60.0)
        with pytest.raises(ValueError, match="thinking is enabled"):
            _v(thinking_metric="rmse")

    def test_invalid_effort_level_rejected(self):
        with pytest.raises(ValueError, match="thinking_effort must be one of"):
            _v(thinking_effort="extreme")

    def test_timeout_above_cap_rejected(self):
        with pytest.raises(ValueError, match="exceeds the"):
            _v(thinking_effort="high", thinking_timeout_s=THINKING_TIMEOUT_MAX_S + 1)

    def test_timeout_at_cap_allowed(self):
        _v(thinking_effort="high", thinking_timeout_s=THINKING_TIMEOUT_MAX_S)

    def test_thinking_with_v3_model_path_allowed(self):
        _v(thinking_mode=True, model_path="v3_default")
        _v(thinking_effort="high", model_path="v3_default")

    def test_thinking_with_auto_model_path_allowed(self):
        # "auto" defers selection to the server (which picks v3 currently).
        _v(thinking_mode=True, model_path="auto")
        # "default" is the backward-compatible alias for "auto".
        _v(thinking_mode=True, model_path="default")

    def test_thinking_with_v25_model_path_rejected(self):
        with pytest.raises(ValueError, match="only supported on v3 models"):
            _v(thinking_mode=True, model_path="v2.5_default")
        with pytest.raises(ValueError, match="only supported on v3 models"):
            _v(thinking_effort="high", model_path="v2.5_default")

    def test_thinking_with_v2_model_path_rejected(self):
        with pytest.raises(ValueError, match="only supported on v3 models"):
            _v(thinking_mode=True, model_path="v2_default")

    def test_thinking_with_v26_model_path_rejected(self):
        with pytest.raises(ValueError, match="only supported on v3 models"):
            _v(thinking_mode=True, model_path="v2.6_default")

    def test_thinking_off_with_non_v3_model_path_allowed(self):
        # The guard only kicks in when thinking is enabled.
        _v(model_path="v2.5_default")
        _v(model_path="v2_default")
