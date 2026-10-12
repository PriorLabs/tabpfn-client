#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""`save_model_to`: what `fit()` saves there as a fit progresses, and how
`load_model()` takes on a fit that was saved before it completed.

Hermetic by default: `init` and the `InferenceClient` boundary are patched. The
mock-server test at the end runs the real HTTP layer for a fit whose process
stopped while waiting, loaded by a later process.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from uuid import UUID, uuid4

import httpx
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_is_fitted

from tabpfn_client import FitFailedError, FittedModelNotFoundError, init, reset
from tabpfn_client.api_models import GetFitStatusResponse
from tabpfn_client.client import PredictionResult, ServiceClient
from tabpfn_client.config import Config
from tabpfn_client.constants import CACHE_DIR
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.models import ApiMode, FitResult
from tabpfn_client.service_wrapper import InferenceClient, UserAuthenticationClient
from tests.mock_tabpfn_server import with_mock_server

FIT_ID = UUID("00000000-0000-0000-0000-000000000abc")
NEW_FIT_ID = UUID("00000000-0000-0000-0000-000000000def")
TIMINGS = {
    "elapsed_s": 42.0,
    "queue_wait_s": 2.0,
    "train_set_transform_s": 10.0,
    "fit_s": 30.0,
}

X_TRAIN = np.arange(18, dtype=float).reshape(9, 2)
Y_TRAIN = np.array([0, 1, 2] * 3)
X_TEST = np.ones((3, 2))

FIT_ERRORS = pytest.mark.parametrize(
    "error",
    [
        FitFailedError("Fit failed: boom"),
        FittedModelNotFoundError("Fitted model not found"),
    ],
    ids=["failed", "gone"],
)


def _completed(fit_id: UUID = FIT_ID) -> GetFitStatusResponse:
    return GetFitStatusResponse.model_validate(
        {"fitted_train_set_id": str(fit_id), "status": "completed", "timings": TIMINGS}
    )


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _offline_server() -> Generator[SimpleNamespace]:
    """Run the estimators with no network: no auth, no settings, mocked inference.

    By default the server accepts the fit as `FIT_ID` and is still running it,
    waiting for a fit completes it with `TIMINGS`, and a prediction is a row of
    zeros per test row.
    """
    use_server_before = Config.use_server
    Config.use_server = True
    try:
        with (
            patch("tabpfn_client.estimator.init"),
            patch.object(ServiceClient, "get_settings", return_value=None),
            patch.object(
                InferenceClient,
                "fit_with_result",
                return_value=FitResult(fitted_train_set_id=FIT_ID, pending=True),
            ) as fit,
            patch.object(
                InferenceClient,
                "wait_for_fit",
                side_effect=lambda fit_id, **kwargs: _completed(fit_id),
            ) as wait,
            patch.object(
                InferenceClient,
                "predict",
                side_effect=lambda X, **kwargs: PredictionResult(
                    y_pred=np.zeros(len(X), dtype=int)
                ),
            ) as predict,
        ):
            yield SimpleNamespace(fit=fit, wait=wait, predict=predict)
    finally:
        Config.use_server = use_server_before


def _stopped_while_waiting(path: Path) -> None:
    """Fit with `save_model_to`, stopping the process during the wait."""
    with _offline_server() as server:
        server.wait.side_effect = KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            TabPFNClassifier(save_model_to=path).fit(X_TRAIN, Y_TRAIN)


class TestFit:
    def test_saves_an_async_fit_when_accepted_and_again_when_completed(self, tmp_path):
        path = tmp_path / "model.json"
        saved_while_waiting: list[dict[str, Any]] = []

        def wait(fit_id: UUID, **kwargs: Any) -> GetFitStatusResponse:
            saved_while_waiting.append(_read(path))
            return _completed(fit_id)

        with _offline_server() as server:
            server.wait.side_effect = wait
            clf = TabPFNClassifier(save_model_to=path).fit(X_TRAIN, Y_TRAIN)

        # Saved before the wait, so a process stopped while waiting keeps it.
        (pending,) = saved_while_waiting
        assert pending["fit_status"] == "pending"
        assert pending["model_id"] == str(FIT_ID)
        record = _read(path)
        assert record["fit_status"] == "completed"
        assert record["model_id"] == str(FIT_ID)
        assert record["classes"] == [0, 1, 2]
        assert record["n_train_rows"] == 9
        assert "save_model_to" not in record["params"]
        assert record == clf.save_model()
        assert clf.get_timings()["fit"] == TIMINGS

    def test_saves_a_sync_fit_once_it_completed(self, tmp_path):
        path = tmp_path / "model.json"

        def fit(*args: Any, **kwargs: Any) -> FitResult:
            assert not path.exists()
            return FitResult(fitted_train_set_id=FIT_ID, timings=TIMINGS)

        with _offline_server() as server:
            server.fit.side_effect = fit
            TabPFNRegressor(api_mode=ApiMode.SYNC, save_model_to=path).fit(
                X_TRAIN, np.arange(9.0)
            )

        server.wait.assert_not_called()
        assert _read(path)["fit_status"] == "completed"

    def test_leaves_the_api_mode_to_the_estimator(self, tmp_path):
        with _offline_server() as server:
            TabPFNClassifier(save_model_to=tmp_path / "model.json").fit(
                X_TRAIN, Y_TRAIN
            )

        assert server.fit.call_args.kwargs["api_mode"] == ApiMode.AUTO
        assert server.fit.call_args.kwargs["wait"] is False

    def test_creates_the_directory(self, tmp_path):
        path = tmp_path / "runs" / "today" / "model.json"

        with _offline_server():
            TabPFNClassifier(save_model_to=path).fit(X_TRAIN, Y_TRAIN)

        assert _read(path)["model_id"] == str(FIT_ID)

    def test_a_stopped_wait_leaves_the_estimator_unfitted_and_the_fit_saved(
        self, tmp_path
    ):
        path = tmp_path / "model.json"

        _stopped_while_waiting(path)

        record = _read(path)
        assert record["fit_status"] == "pending"
        assert record["model_id"] == str(FIT_ID)

    @FIT_ERRORS
    def test_a_fit_lost_while_waiting_removes_the_file(self, tmp_path, error):
        path = tmp_path / "model.json"
        clf = TabPFNClassifier(save_model_to=path)

        with _offline_server() as server:
            server.wait.side_effect = error
            with pytest.raises(type(error)):
                clf.fit(X_TRAIN, Y_TRAIN)

        assert not path.exists()
        with pytest.raises(NotFittedError):
            check_is_fitted(clf)

    def test_every_fit_overwrites_the_file(self, tmp_path):
        path = tmp_path / "model.json"
        clf = TabPFNClassifier(save_model_to=path)

        with _offline_server() as server:
            clf.fit(X_TRAIN, Y_TRAIN)
            server.fit.return_value = FitResult(
                fitted_train_set_id=NEW_FIT_ID, pending=True
            )
            clf.fit(X_TRAIN, Y_TRAIN)

        assert server.fit.call_count == 2
        assert _read(path)["model_id"] == str(NEW_FIT_ID)

    def test_save_model_to_is_an_estimator_param(self, tmp_path):
        clf = TabPFNClassifier(save_model_to=tmp_path / "model.json")

        assert clf.get_params()["save_model_to"] == tmp_path / "model.json"
        cloned = clone(clf)
        assert isinstance(cloned, TabPFNClassifier)
        assert cloned.save_model_to == tmp_path / "model.json"

    def test_clones_in_a_cross_validation_share_the_file(self, tmp_path):
        path = tmp_path / "model.json"
        fit_ids: list[UUID] = []

        def fit(*args: Any, **kwargs: Any) -> FitResult:
            fit_ids.append(uuid4())
            return FitResult(fitted_train_set_id=fit_ids[-1], pending=True)

        with _offline_server() as server:
            server.fit.side_effect = fit
            cross_val_score(
                TabPFNClassifier(save_model_to=path), X_TRAIN, Y_TRAIN, cv=3
            )

        assert len(fit_ids) == 3
        assert _read(path)["model_id"] == str(fit_ids[-1])


class TestLoadModel:
    def test_waits_for_a_pending_fit_and_saves_it_as_completed(self, tmp_path):
        path = tmp_path / "model.json"
        _stopped_while_waiting(path)

        with _offline_server() as server:
            loaded = TabPFNClassifier.load_model(path)
            loaded.predict(X_TEST)

        server.fit.assert_not_called()
        assert server.wait.call_args.args == (FIT_ID,)
        assert server.predict.call_args.kwargs["fitted_train_set_id"] == FIT_ID
        np.testing.assert_array_equal(loaded.classes_, [0, 1, 2])
        assert loaded._n_train_rows == 9
        assert loaded.get_timings()["fit"] == TIMINGS
        assert loaded.save_model_to is None
        record = _read(path)
        assert record["fit_status"] == "completed"
        assert record == loaded.save_model()

    def test_loads_a_completed_fit_without_a_request(self, tmp_path):
        path = tmp_path / "model.json"
        with _offline_server():
            TabPFNClassifier(save_model_to=path).fit(X_TRAIN, Y_TRAIN)
        before = path.read_text()

        with _offline_server() as server:
            loaded = TabPFNClassifier.load_model(path)

        server.wait.assert_not_called()
        check_is_fitted(loaded)
        assert loaded.get_timings()["fit"] is None
        assert path.read_text() == before

    @FIT_ERRORS
    def test_a_pending_fit_the_server_lost_raises(self, tmp_path, error):
        path = tmp_path / "model.json"
        _stopped_while_waiting(path)
        before = path.read_text()

        with _offline_server() as server, pytest.raises(type(error)):
            server.wait.side_effect = error
            TabPFNClassifier.load_model(path)

        assert path.read_text() == before

    def test_a_pending_record_given_as_a_dict_is_waited_for(self, tmp_path):
        path = tmp_path / "model.json"
        _stopped_while_waiting(path)
        record = _read(path)

        with _offline_server() as server:
            loaded = TabPFNClassifier.load_model(record)

        assert server.wait.call_args.args == (FIT_ID,)
        assert loaded.model_id_ == FIT_ID
        assert record["fit_status"] == "pending"

    def test_a_file_that_cannot_be_updated_still_loads(self, tmp_path, caplog):
        path = tmp_path / "model.json"
        _stopped_while_waiting(path)

        with (
            _offline_server(),
            patch(
                "tabpfn_client.persistence._write_record",
                side_effect=PermissionError("read-only"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            loaded = TabPFNClassifier.load_model(path)

        assert loaded.model_id_ == FIT_ID
        assert _read(path)["fit_status"] == "pending"
        assert "read-only" in caplog.text


class _Stopped(Exception):
    """Stands for the process being stopped while it waited for the fit."""


class TestWithServer:
    dummy_token = "dummy_token"

    def setup_method(self):
        reset()
        ServiceClient.reset_authorization()
        ServiceClient._api_settings = None

    def teardown_method(self):
        Config.is_initialized = False
        ServiceClient.reset_authorization()
        ServiceClient._api_settings = None
        shutil.rmtree(CACHE_DIR, ignore_errors=True)

    # `with_mock_server` hides the test's signature, so pytest cannot pass
    # `tmp_path` to the test itself.
    @pytest.fixture(autouse=True)
    def _tmp_path(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path

    @with_mock_server()
    def test_a_fit_stopped_while_waiting_loads_in_a_later_run(self, mock_server):
        fit_id = "00000000-0000-0000-0000-000000000002"
        path = self.tmp_path / "model.json"
        UserAuthenticationClient.CACHED_TOKEN_FILE.parent.mkdir(
            parents=True, exist_ok=True
        )
        UserAuthenticationClient.CACHED_TOKEN_FILE.write_text(self.dummy_token)

        auth_route = mock_server.router.get(mock_server.endpoints.protected_root.path)
        auth_route.respond(200)
        mock_server.router.get(
            mock_server.endpoints.retrieve_greeting_messages.path
        ).respond(200, json={"messages": []})
        upload_route = mock_server.router.post("/tabpfn/prepare_train_set_upload")
        upload_route.respond(
            409,
            json={
                "message": "duplicate",
                "error_code": "DUPLICATE_TRAIN_SET_UPLOAD",
                "train_set_upload_id": "00000000-0000-0000-0000-000000000001",
            },
        )
        submit_route = mock_server.router.post("/tabpfn/submit_fit_job")
        submit_route.respond(200, json={"fitted_train_set_id": fit_id})
        status_route = mock_server.router.post("/tabpfn/get_fit_status")
        status_route.side_effect = [
            _Stopped("the process stopped"),
            httpx.Response(
                200,
                json={
                    "fitted_train_set_id": fit_id,
                    "status": "completed",
                    "timings": TIMINGS,
                },
            ),
        ]
        mock_server.router.post("/tabpfn/prepare_test_set_upload").respond(
            409,
            json={
                "message": "duplicate",
                "error_code": "DUPLICATE_TEST_SET_UPLOAD",
                "test_set_upload_id": "00000000-0000-0000-0000-000000000003",
            },
        )
        predict_route = mock_server.router.post(mock_server.endpoints.predict.path)
        predict_route.respond(
            200,
            json={
                "prediction": [1, 0, 1],
                "metadata": {
                    "task": "classification",
                    "package_version": "0.3.0",
                    "tabpfn_config": {},
                    "test_set_num_rows": 3,
                    "test_set_num_cols": 2,
                },
            },
        )

        with patch.object(ServiceClient, "get_settings", return_value=None):
            # A run that submits the fit and stops while waiting for it.
            init(use_server=True)
            with pytest.raises(_Stopped):
                TabPFNClassifier(
                    thinking_mode=True, api_mode=ApiMode.ASYNC, save_model_to=path
                ).fit(X_TRAIN, Y_TRAIN)
            assert _read(path)["fit_status"] == "pending"
            assert _read(path)["model_id"] == fit_id

            # A later run, in a new process.
            Config.is_initialized = False
            ServiceClient.reset_authorization()
            clf = TabPFNClassifier.load_model(path)
            result = clf.predict(X_TEST)

        np.testing.assert_array_equal(result, [1, 0, 1])
        assert upload_route.call_count == 1
        assert submit_route.call_count == 1
        assert status_route.call_count == 2
        assert auth_route.call_count == 2, "each run authenticates once"
        predict_request = json.loads(predict_route.calls.last.request.content)
        assert predict_request["fitted_train_set_id"] == fit_id
        assert _read(path)["fit_status"] == "completed"
        assert clf.get_timings()["fit"] == TIMINGS


def test_an_interrupted_fit_without_save_model_to_keeps_the_previous_state():
    clf = TabPFNClassifier()

    with _offline_server() as server:
        server.wait.side_effect = KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            clf.fit(X_TRAIN, Y_TRAIN)

    with pytest.raises(NotFittedError):
        check_is_fitted(clf)
