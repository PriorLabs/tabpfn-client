"""A model_path's limits come from the version the server gives its name."""

import time

import pytest

from tabpfn_client import config
from tabpfn_client.client import GetSettingsResponse, ServiceClient
from tabpfn_client.api_models import ModelLimit, ThinkingLimit
from tabpfn_client.estimator import _limit_for_model_path, _thinking_limit
from tests.unit.test_tabpfn_classifier import _api_settings_payload


def _use_settings(model_names: dict[str, str] | None) -> GetSettingsResponse:
    payload = _api_settings_payload()
    payload["model_limits"] = {
        "v2": _api_settings_payload(max_rows=1)["max_model_limit"],
        # A version this client does not know.
        "v99": _api_settings_payload(max_rows=99)["max_model_limit"],
    }
    payload["thinking_model_limits"] = {
        "v99": {**payload["model_limits"]["v99"], "efforts": ["medium"]}
    }
    if model_names is not None:
        payload["model_names"] = model_names
    settings = GetSettingsResponse(**payload)
    ServiceClient._api_settings = settings
    ServiceClient._api_settings_ts = time.monotonic()
    return settings


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


@pytest.mark.parametrize("model_path", ["v99_default", "checkpoints/tabpfn-v99.ckpt"])
def test_a_version_unknown_to_the_client_gets_its_own_limits(model_path):
    settings = _use_settings(
        {"v99_default": "v99", "tabpfn-v99.ckpt": "v99", "gn2p4bpt": "v2"}
    )
    assert _rows(_limit_for_model_path(model_path)) == 99
    assert _rows(_thinking_limit(settings, model_path)) == 99
    assert _rows(_limit_for_model_path("gn2p4bpt")) == 1


def test_a_name_the_server_does_not_list_skips_the_local_check():
    _use_settings({"v99_default": "v99"})
    assert _limit_for_model_path("not_a_model") is None


def test_an_older_server_without_model_names_reads_the_version_from_the_name():
    _use_settings(None)
    assert _rows(_limit_for_model_path("v99_default")) == 1
