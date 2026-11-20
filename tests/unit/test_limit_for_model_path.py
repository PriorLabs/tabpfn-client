"""A model_path's limits are those of the server's version that its name carries."""

import time

import pytest

from tabpfn_client import config
from tabpfn_client.api_models import ModelLimit, ThinkingLimit
from tabpfn_client.client import GetSettingsResponse, ServiceClient
from tabpfn_client.estimator import _limit_for_model_path, _thinking_limit
from tests.unit.test_tabpfn_classifier import _api_settings_payload


def _rows(limit: ModelLimit | ThinkingLimit | None) -> int:
    assert limit is not None
    return limit.train_set_max_rows


@pytest.fixture(autouse=True)
def skip_init():
    config.Config.is_initialized = True
    yield
    ServiceClient._api_settings = None
    ServiceClient._api_settings_ts = 0.0
    config.reset()


def test_a_version_newer_than_the_client_gets_its_own_limits():
    payload = _api_settings_payload()
    payload["model_limits"] = {
        "v2": _api_settings_payload(max_rows=2)["max_model_limit"],
        # A version this client does not know.
        "v99": _api_settings_payload(max_rows=99)["max_model_limit"],
    }
    payload["thinking_model_limits"] = {
        "v99": {**payload["model_limits"]["v99"], "efforts": ["medium"]}
    }
    settings = GetSettingsResponse(**payload)
    ServiceClient._api_settings = settings
    ServiceClient._api_settings_ts = time.monotonic()

    assert _rows(_limit_for_model_path("v99_default")) == 99
    assert _rows(_thinking_limit(settings, "v99_default")) == 99
    assert _rows(_limit_for_model_path("gn2p4bpt")) == 2


def test_a_known_version_the_server_omits_gets_the_closest_lower_limits():
    payload = _api_settings_payload()
    payload["model_limits"] = {
        "v2": _api_settings_payload(max_rows=2)["max_model_limit"],
        "v2.5": _api_settings_payload(max_rows=25)["max_model_limit"],
    }
    ServiceClient._api_settings = GetSettingsResponse(**payload)
    ServiceClient._api_settings_ts = time.monotonic()

    assert _rows(_limit_for_model_path("v2.6_default")) == 25
