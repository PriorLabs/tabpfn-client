"""Text presets travel from sklearn estimators to both fit endpoints."""

import json
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.base import clone

from tabpfn_client.api_models import ClassifierFitTaskConfig
from tabpfn_client.client import ServiceClient
from tabpfn_client.config import Config
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.models import ApiMode
from tests.mock_tabpfn_server import MockTabPFNServer
from tests.unit import test_timings


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
@pytest.mark.parametrize("mode", [ApiMode.SYNC, ApiMode.ASYNC])
@pytest.mark.parametrize("preset", [None, "advanced", "simple"])
def test_text_handling_reaches_fit_request(
    estimator_cls: type[TabPFNClassifier] | type[TabPFNRegressor],
    mode: ApiMode,
    preset: str | None,
) -> None:
    params: dict[str, Any] = {} if preset is None else {"text_handling": preset}
    estimator = clone(estimator_cls(api_mode=mode, **params))
    assert isinstance(estimator, (TabPFNClassifier, TabPFNRegressor))
    x = np.arange(60).reshape(20, 3)
    y = np.arange(20) % 2
    endpoint = "/tabpfn/fit" if mode == ApiMode.SYNC else "/tabpfn/submit_fit_job"
    with (
        MockTabPFNServer() as server,
        patch("tabpfn_client.estimator.init"),
        patch.object(Config, "use_server", True),
        patch.object(ServiceClient, "get_settings", return_value=None),
        patch.object(ServiceClient, "_upload_to_gcs"),
    ):
        assert server.router is not None
        test_timings.TestClientTimings._route_train_upload(server)
        route = server.router.post(endpoint).respond(
            200,
            json={"fitted_train_set_id": test_timings.FIT_ID, "status": "completed"},
        )
        if mode == ApiMode.ASYNC:
            server.router.post("/tabpfn/get_fit_status").respond(
                200,
                json={
                    "fitted_train_set_id": test_timings.FIT_ID,
                    "status": "completed",
                },
            )
        estimator.fit(x, y)
        payload = json.loads(route.calls.last.request.content)
        if preset == "simple":
            assert payload["text_handling"] == "simple"
        else:
            assert "text_handling" not in payload


@pytest.mark.parametrize("estimator_cls", [TabPFNClassifier, TabPFNRegressor])
@pytest.mark.parametrize("preset", ["invalid", None, False])
def test_invalid_text_handling_fails_before_auth(
    estimator_cls: type[TabPFNClassifier] | type[TabPFNRegressor], preset: Any
) -> None:
    estimator = estimator_cls().set_params(text_handling=preset)
    with patch("tabpfn_client.estimator.init") as initialize:
        with pytest.raises(ValueError, match="text_handling"):
            estimator.fit(np.ones((20, 2)), np.arange(20) % 2)
        initialize.assert_not_called()


@pytest.mark.parametrize("preset", ["invalid", None, False])
def test_direct_client_rejects_invalid_text_handling_before_network(
    preset: Any,
) -> None:
    with patch.object(ServiceClient, "get_settings") as get_settings:
        with pytest.raises(ValueError, match="text_handling"):
            ServiceClient.fit_with_result(
                np.ones((20, 2)),
                np.arange(20) % 2,
                task_config=ClassifierFitTaskConfig(),
                tabpfn_systems=["preprocessing", "text"],
                text_handling=preset,
            )
        get_settings.assert_not_called()
