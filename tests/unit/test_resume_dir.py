#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""`resume_dir`: what an estimator records there as a fit progresses, and what a
later fit of the same estimator on the same data takes from it.

Hermetic by default: `init` and the `InferenceClient` boundary are patched. The
mock-server test at the end runs the real HTTP layer for a fit whose process
stopped while waiting, resumed by a later process.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterator
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
from tabpfn_client.estimator import (
    TabPFNClassifier,
    TabPFNRegressor,
    _data_fingerprint,
)
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


def _completed(fit_id: UUID = FIT_ID) -> GetFitStatusResponse:
    return GetFitStatusResponse.model_validate(
        {"fitted_train_set_id": str(fit_id), "status": "completed", "timings": TIMINGS}
    )


def _record_files(resume_dir: Path) -> list[Path]:
    return sorted(resume_dir.glob("*.json"))


def _only_record(resume_dir: Path) -> dict[str, Any]:
    (path,) = _record_files(resume_dir)
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _offline_server() -> Iterator[SimpleNamespace]:
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


def _stopped_while_waiting(resume_dir: Path, fit_id: UUID = FIT_ID) -> None:
    """Fit with `resume_dir`, stopping the process during the wait."""
    with _offline_server() as server:
        server.fit.return_value = FitResult(fitted_train_set_id=fit_id, pending=True)
        server.wait.side_effect = KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            TabPFNClassifier(resume_dir=resume_dir).fit(X_TRAIN, Y_TRAIN)


class TestFirstFit:
    def test_records_the_fit_when_accepted_and_again_when_completed(self, tmp_path):
        recorded_while_waiting: list[dict[str, Any]] = []

        def wait(fit_id: UUID, **kwargs: Any) -> GetFitStatusResponse:
            recorded_while_waiting.append(_only_record(tmp_path))
            return _completed(fit_id)

        with _offline_server() as server:
            server.wait.side_effect = wait
            clf = TabPFNClassifier(resume_dir=tmp_path).fit(X_TRAIN, Y_TRAIN)

        # Recorded before the wait, so a process stopped while waiting can resume.
        (pending,) = recorded_while_waiting
        assert pending["fit_status"] == "pending"
        assert pending["model_id"] == str(FIT_ID)
        (path,) = _record_files(tmp_path)
        assert path.name.startswith("TabPFNClassifier-")
        record = _only_record(tmp_path)
        assert record["fit_status"] == "completed"
        assert record["model_id"] == str(FIT_ID)
        assert record["classes"] == [0, 1, 2]
        assert record["n_train_rows"] == 9
        assert record["train_data_fingerprint"] == pending["train_data_fingerprint"]
        assert record["train_data_fingerprint"] is not None
        assert "resume_dir" not in record["params"]
        assert clf.model_id_ == FIT_ID
        assert clf.get_timings()["fit"] == TIMINGS

    def test_auto_submits_a_job_so_the_fit_is_recorded_before_it_runs(self, tmp_path):
        with _offline_server() as server:
            TabPFNClassifier(resume_dir=tmp_path).fit(X_TRAIN, Y_TRAIN)

        assert server.fit.call_args.kwargs["api_mode"] == ApiMode.ASYNC
        assert server.fit.call_args.kwargs["wait"] is False

    def test_sync_api_mode_records_the_fit_once_it_completed(self, tmp_path):
        with _offline_server() as server:
            server.fit.return_value = FitResult(
                fitted_train_set_id=FIT_ID, timings=TIMINGS
            )
            TabPFNRegressor(api_mode="sync", resume_dir=tmp_path).fit(
                X_TRAIN, np.arange(9.0)
            )

        assert server.fit.call_args.kwargs["api_mode"] == "sync"
        server.wait.assert_not_called()
        assert _only_record(tmp_path)["fit_status"] == "completed"

    def test_creates_the_directory(self, tmp_path):
        resume_dir = tmp_path / "runs" / "today"

        with _offline_server():
            TabPFNClassifier(resume_dir=resume_dir).fit(X_TRAIN, Y_TRAIN)

        assert _only_record(resume_dir)["model_id"] == str(FIT_ID)

    def test_without_resume_dir_the_transport_is_the_one_asked_for(self):
        clf = TabPFNClassifier()

        with _offline_server() as server:
            clf.fit(X_TRAIN, Y_TRAIN)

        assert server.fit.call_args.kwargs["api_mode"] == ApiMode.AUTO
        assert clf.model_id_ == FIT_ID
        assert clf._train_data_fingerprint is None

    def test_a_stopped_wait_leaves_the_estimator_unfitted_and_the_fit_recorded(
        self, tmp_path
    ):
        _stopped_while_waiting(tmp_path)

        record = _only_record(tmp_path)
        assert record["fit_status"] == "pending"
        assert record["model_id"] == str(FIT_ID)

    def test_resume_dir_is_an_estimator_param(self, tmp_path):
        clf = TabPFNClassifier(resume_dir=tmp_path)

        assert clf.get_params()["resume_dir"] == tmp_path
        assert clone(clf).resume_dir == tmp_path


class TestResume:
    def test_a_completed_fit_is_taken_on_without_fitting_again(self, tmp_path):
        with _offline_server():
            TabPFNClassifier(n_estimators=4, resume_dir=tmp_path).fit(X_TRAIN, Y_TRAIN)
        (path,) = _record_files(tmp_path)
        before = path.read_text()

        with _offline_server() as server:
            clf = TabPFNClassifier(n_estimators=4, resume_dir=tmp_path)
            clf.fit(X_TRAIN, Y_TRAIN)
            clf.predict(X_TEST)

        server.fit.assert_not_called()
        # One status request, which also confirms the server still has the fit.
        assert server.wait.call_args.args == (FIT_ID,)
        assert server.predict.call_args.kwargs["fitted_train_set_id"] == FIT_ID
        np.testing.assert_array_equal(clf.classes_, [0, 1, 2])
        assert clf._n_train_rows == 9
        assert clf.get_timings()["fit"] == TIMINGS
        assert path.read_text() == before

    def test_a_fit_stopped_while_waiting_is_waited_for(self, tmp_path):
        _stopped_while_waiting(tmp_path)

        with _offline_server() as server:
            clf = TabPFNClassifier(resume_dir=tmp_path).fit(X_TRAIN, Y_TRAIN)

        server.fit.assert_not_called()
        assert server.wait.call_args.args == (FIT_ID,)
        assert clf.model_id_ == FIT_ID
        assert _only_record(tmp_path)["fit_status"] == "completed"

    @pytest.mark.parametrize(
        "error",
        [
            FitFailedError("Fit failed: boom"),
            FittedModelNotFoundError("Fitted model not found"),
        ],
        ids=["failed", "gone"],
    )
    def test_a_fit_that_cannot_be_resumed_is_fitted_again(
        self, tmp_path, caplog, error
    ):
        _stopped_while_waiting(tmp_path)

        def wait(fit_id: UUID, **kwargs: Any) -> GetFitStatusResponse:
            if fit_id == FIT_ID:
                raise error
            return _completed(fit_id)

        with _offline_server() as server, caplog.at_level(logging.WARNING):
            server.fit.return_value = FitResult(
                fitted_train_set_id=NEW_FIT_ID, pending=True
            )
            server.wait.side_effect = wait
            clf = TabPFNClassifier(resume_dir=tmp_path).fit(X_TRAIN, Y_TRAIN)

        assert clf.model_id_ == NEW_FIT_ID
        record = _only_record(tmp_path)
        assert record["model_id"] == str(NEW_FIT_ID)
        assert record["fit_status"] == "completed"
        assert str(FIT_ID) in caplog.text

    @pytest.mark.parametrize(
        "change",
        [{"params": {"n_estimators": 8}}, {"y": np.array([0, 1, 1] * 3)}],
        ids=["params", "data"],
    )
    def test_a_different_fit_gets_a_file_of_its_own(self, tmp_path, change):
        with _offline_server():
            TabPFNClassifier(n_estimators=4, resume_dir=tmp_path).fit(X_TRAIN, Y_TRAIN)
        (first,) = _record_files(tmp_path)
        before = first.read_text()
        params = {"n_estimators": 4, **change.get("params", {})}

        with _offline_server() as server:
            server.fit.return_value = FitResult(
                fitted_train_set_id=NEW_FIT_ID, pending=True
            )
            TabPFNClassifier(resume_dir=tmp_path, **params).fit(
                X_TRAIN, change.get("y", Y_TRAIN)
            )

        # The recorded fit is not even asked about, and stays for its own data.
        assert [c.args for c in server.wait.call_args_list] == [(NEW_FIT_ID,)]
        assert len(_record_files(tmp_path)) == 2
        assert first.read_text() == before

    def test_a_classifier_and_a_regressor_share_a_directory(self, tmp_path):
        with _offline_server():
            TabPFNClassifier(resume_dir=tmp_path).fit(X_TRAIN, Y_TRAIN)
            TabPFNRegressor(resume_dir=tmp_path).fit(X_TRAIN, Y_TRAIN.astype(float))

        names = [p.name.split("-")[0] for p in _record_files(tmp_path)]
        assert sorted(names) == ["TabPFNClassifier", "TabPFNRegressor"]

    def test_the_directory_can_move(self, tmp_path):
        with _offline_server():
            TabPFNClassifier(resume_dir=tmp_path / "here").fit(X_TRAIN, Y_TRAIN)
        shutil.move(tmp_path / "here", tmp_path / "there")

        with _offline_server() as server:
            TabPFNClassifier(resume_dir=tmp_path / "there").fit(X_TRAIN, Y_TRAIN)

        server.fit.assert_not_called()

    def test_a_record_of_another_fit_copied_in_is_replaced(self, tmp_path, caplog):
        clf = TabPFNClassifier(resume_dir=tmp_path)
        path = clf._checkpoint_path(tmp_path, _data_fingerprint(X_TRAIN, Y_TRAIN))
        with _offline_server():
            regressor = TabPFNRegressor().fit(X_TRAIN, Y_TRAIN.astype(float))
        regressor.save_model(path)

        with _offline_server() as server, caplog.at_level(logging.WARNING):
            server.fit.return_value = FitResult(
                fitted_train_set_id=NEW_FIT_ID, pending=True
            )
            clf.fit(X_TRAIN, Y_TRAIN)

        assert json.loads(path.read_text())["task"] == "classification"
        assert "a regression model" in caplog.text

    def test_a_file_that_is_not_a_model_record_is_left_alone(self, tmp_path):
        clf = TabPFNClassifier(resume_dir=tmp_path)
        path = clf._checkpoint_path(tmp_path, _data_fingerprint(X_TRAIN, Y_TRAIN))
        path.write_text("my notes, not a model\n")

        with (
            _offline_server() as server,
            pytest.raises(ValueError, match="not a model file"),
        ):
            clf.fit(X_TRAIN, Y_TRAIN)

        server.fit.assert_not_called()
        assert path.read_text() == "my notes, not a model\n"


class TestChained:
    def test_each_fold_of_a_cross_validation_resumes_its_own_fit(self, tmp_path):
        estimator = TabPFNClassifier(resume_dir=tmp_path)
        with _offline_server() as server:
            server.fit.side_effect = lambda *args, **kwargs: FitResult(
                fitted_train_set_id=uuid4(), pending=True
            )
            first = cross_val_score(estimator, X_TRAIN, Y_TRAIN, cv=3)
        assert len(_record_files(tmp_path)) == 3

        with _offline_server() as server:
            again = cross_val_score(estimator, X_TRAIN, Y_TRAIN, cv=3)

        server.fit.assert_not_called()
        assert server.wait.call_count == 3
        np.testing.assert_array_equal(first, again)


class TestLoadModel:
    def test_refuses_a_fit_that_had_not_completed(self, tmp_path):
        _stopped_while_waiting(tmp_path)
        (path,) = _record_files(tmp_path)

        with pytest.raises(ValueError, match="resume_dir"):
            TabPFNClassifier.load_model(path)

    def test_loads_a_completed_record_without_its_directory(self, tmp_path):
        with _offline_server():
            TabPFNClassifier(resume_dir=tmp_path).fit(X_TRAIN, Y_TRAIN)
        (path,) = _record_files(tmp_path)

        loaded = TabPFNClassifier.load_model(path)

        check_is_fitted(loaded)
        assert loaded.resume_dir is None
        assert loaded.save_model() == _only_record(tmp_path)


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
    def test_a_fit_stopped_while_waiting_resumes_in_a_later_run(self, mock_server):
        fit_id = "00000000-0000-0000-0000-000000000002"
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
                TabPFNClassifier(thinking_mode=True, resume_dir=self.tmp_path).fit(
                    X_TRAIN, Y_TRAIN
                )
            assert _only_record(self.tmp_path)["fit_status"] == "pending"
            assert _only_record(self.tmp_path)["model_id"] == fit_id

            # A later run, in a new process, with the same script.
            Config.is_initialized = False
            ServiceClient.reset_authorization()
            clf = TabPFNClassifier(thinking_mode=True, resume_dir=self.tmp_path)
            clf.fit(X_TRAIN, Y_TRAIN)
            result = clf.predict(X_TEST)

        np.testing.assert_array_equal(result, [1, 0, 1])
        assert upload_route.call_count == 1
        assert submit_route.call_count == 1
        assert status_route.call_count == 2
        assert auth_route.call_count == 2, "each run authenticates once"
        predict_request = json.loads(predict_route.calls.last.request.content)
        assert predict_request["fitted_train_set_id"] == fit_id
        assert _only_record(self.tmp_path)["fit_status"] == "completed"
        assert clf.get_timings()["fit"] == TIMINGS


def test_an_interrupted_fit_without_resume_dir_keeps_the_previous_state():
    clf = TabPFNClassifier()

    with _offline_server() as server:
        server.wait.side_effect = KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            clf.fit(X_TRAIN, Y_TRAIN)

    with pytest.raises(NotFittedError):
        check_is_fitted(clf)
