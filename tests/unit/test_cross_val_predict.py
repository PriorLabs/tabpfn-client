#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""`cross_val_predict()`, `submit_cross_val_predict()` and
`wait_for_cross_val_predict()`: out-of-fold predictions from one server-side
cross-validation job.

The estimator tests patch `init` and the `InferenceClient` boundary. The
`ServiceClient` tests run the real HTTP layer against a mocked server.
"""

from __future__ import annotations

import io
import json
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch
from uuid import UUID

import httpx
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    TimeSeriesSplit,
)
from sklearn.utils.validation import check_is_fitted

from tabpfn_client import CrossValidateFailedError
from tabpfn_client.api_models import (
    ClassifierConfig,
    GetCrossValidateResultResponse,
    RegressorConfig,
)
from tabpfn_client.client import ResolvedAsyncSettings, ServiceClient, _fold_sizes
from tabpfn_client.config import Config
from tabpfn_client.estimator import (
    TabPFNClassifier,
    TabPFNRegressor,
    _cv_fold_ids,
)
from tabpfn_client.models import ApiMode, ClientOptions
from tabpfn_client.service_wrapper import InferenceClient
from tests.mock_tabpfn_server import MockTabPFNServer
from tests.unit.test_client import _FakeTime

CV_JOB_ID = UUID("00000000-0000-0000-0000-0000000000c1")
NEW_CV_JOB_ID = UUID("00000000-0000-0000-0000-0000000000c2")
FIT_ID = UUID("00000000-0000-0000-0000-000000000abc")
OOF_URL = "https://storage.example/oof_r0.parquet"

N_ROWS = 12
X = np.arange(N_ROWS * 2, dtype=float).reshape(N_ROWS, 2)
CLASSES = np.array(["a", "b", "c"])
Y_CLASSES = CLASSES[np.arange(N_ROWS) % 3]
Y_REGRESSION = np.arange(N_ROWS, dtype=float)


def _completed(cv_job_id: UUID = CV_JOB_ID, **fields: Any) -> Any:
    return GetCrossValidateResultResponse(
        cv_job_id=cv_job_id, status="completed", oof_urls=[OOF_URL], **fields
    )


def _pending(cv_job_id: UUID = CV_JOB_ID) -> Any:
    return GetCrossValidateResultResponse(
        cv_job_id=cv_job_id, status="pending", retry_in_secs=2.0
    )


def _oof_frame(fold_ids: list[int], classes: list[Any] | None) -> pd.DataFrame:
    """What the server writes: every row, the ones in no test fold null.

    A classifier's row `r` is most likely class `r % 3`; a regressor predicts
    `10 * r`.
    """
    rows = np.arange(len(fold_ids))
    held_out = np.asarray(fold_ids) == -1
    frame = pd.DataFrame({"row": rows})
    if classes is None:
        frame["prediction"] = np.where(held_out, np.nan, 10.0 * rows)
        return frame
    for k, label in enumerate(classes):
        proba = np.where(rows % 3 == k, 0.8, 0.1)
        frame[f"proba_{label}"] = np.where(held_out, np.nan, proba)
    # The server's row order is not the train set's.
    return frame.iloc[::-1].reset_index(drop=True)


def _pending_id(model: Any) -> UUID | None:
    """The id of the cross-validation the model holds, or None."""
    pending = model._pending_cross_val_predict
    return None if pending is None else pending.cv_job_id


@contextmanager
def _offline_server(
    *,
    first: str = "completed",
    wait_outcome: BaseException | None = None,
    download_outcome: BaseException | None = None,
    classes: list[Any] = list(CLASSES),
) -> Generator[SimpleNamespace]:
    """Run the estimators with no network: no auth, mocked inference.

    The server answers a cross-validation `first` (completed or pending), as
    `CV_JOB_ID`; waiting for it raises `wait_outcome`, or finds it completed.
    The first download of the predictions raises `download_outcome`.
    A classification job reports `classes`. The predictions follow
    `_oof_frame` for the fold ids it was sent.
    """
    use_server_before = Config.use_server
    Config.use_server = True
    sent: dict[str, Any] = {}

    def job_classes() -> list[Any] | None:
        is_classifier = isinstance(sent["task_config"], ClassifierConfig)
        return classes if is_classifier else None

    def cross_validate(X: Any, y: Any, **kwargs: Any) -> Any:
        sent.update(kwargs, X=X, y=y)
        return _pending() if first == "pending" else _completed(classes=job_classes())

    def wait_for_cross_validate(cv_job_id: UUID, **kwargs: Any) -> Any:
        if wait_outcome is not None:
            raise wait_outcome
        return _completed(classes=job_classes())

    failures = [download_outcome] if download_outcome is not None else []

    def download(urls: list[str], client_options: Any = None) -> list[pd.DataFrame]:
        assert urls == [OOF_URL]
        if failures:
            raise failures.pop()
        return [_oof_frame(sent["fold_ids"][0], job_classes())]

    try:
        with (
            patch("tabpfn_client.estimator.init") as init_,
            patch.object(ServiceClient, "get_settings", return_value=None),
            patch.object(
                InferenceClient, "cross_validate", side_effect=cross_validate
            ) as cv,
            patch.object(
                InferenceClient,
                "wait_for_cross_validate",
                side_effect=wait_for_cross_validate,
            ) as wait,
            patch.object(
                InferenceClient,
                "download_cross_validate_predictions",
                side_effect=download,
            ),
        ):
            yield SimpleNamespace(init=init_, cv=cv, wait=wait, sent=sent)
    finally:
        Config.use_server = use_server_before


class TestFoldIds:
    def test_an_int_is_a_stratified_k_fold_for_a_classifier(self):
        fold_ids = _cv_fold_ids(3, X, Y_CLASSES, None, classifier=True)
        expected = np.full(N_ROWS, -1)
        for fold, (_, test) in enumerate(StratifiedKFold(3).split(X, Y_CLASSES)):
            expected[test] = fold
        assert fold_ids == expected.tolist()

    def test_an_int_is_a_k_fold_for_a_regressor(self):
        fold_ids = _cv_fold_ids(4, X, Y_REGRESSION, None, classifier=False)
        assert fold_ids == np.repeat(np.arange(4), 3).tolist()

    def test_a_splitter_gets_the_groups(self):
        groups = np.arange(N_ROWS) // 4
        fold_ids = _cv_fold_ids(
            GroupKFold(3), X, Y_REGRESSION, groups, classifier=False
        )
        # Every group is one fold.
        assert len({fold_ids[i] for i in range(0, 4)}) == 1
        assert len(set(fold_ids)) == 3

    def test_rows_in_no_test_set_are_minus_one(self):
        rows = np.arange(N_ROWS)
        splits = [
            (rows[2:], rows[:2]),
            (np.concatenate([rows[:2], rows[4:]]), rows[2:4]),
        ]
        fold_ids = _cv_fold_ids(splits, X, Y_REGRESSION, None, classifier=False)
        assert fold_ids == [0, 0, 1, 1] + [-1] * (N_ROWS - 4)

    @pytest.mark.parametrize(
        "cv",
        [
            ShuffleSplit(n_splits=3, test_size=0.5, random_state=0),
            [(np.arange(6, 12), np.arange(6)), (np.arange(3, 12), np.arange(3))],
        ],
        ids=["shuffle_split", "overlapping_pairs"],
    )
    def test_overlapping_test_sets_are_rejected(self, cv):
        with pytest.raises(ValueError, match="partitions"):
            _cv_fold_ids(cv, X, Y_REGRESSION, None, classifier=False)

    def test_a_split_that_trains_on_fewer_rows_is_rejected(self):
        with pytest.raises(ValueError, match="TimeSeriesSplit"):
            _cv_fold_ids(TimeSeriesSplit(3), X, Y_REGRESSION, None, classifier=False)

    def test_an_index_out_of_range_is_rejected(self):
        splits = [(np.arange(N_ROWS - 1), np.array([N_ROWS]))]
        with pytest.raises(ValueError, match="outside"):
            _cv_fold_ids(splits, X, Y_REGRESSION, None, classifier=False)

    def test_a_single_fold_over_every_row_is_rejected(self):
        splits = [(np.array([], dtype=int), np.arange(N_ROWS))]
        with pytest.raises(ValueError, match="empty train set"):
            _cv_fold_ids(splits, X, Y_REGRESSION, None, classifier=False)


class TestClassifier:
    def test_predict_proba_orders_rows_and_classes(self):
        with _offline_server() as server:
            probas = TabPFNClassifier().cross_val_predict(
                X, Y_CLASSES, cv=3, method="predict_proba"
            )
        assert probas.shape == (N_ROWS, 3)
        assert (probas.argmax(axis=1) == np.arange(N_ROWS) % 3).all()
        sent = server.sent
        assert sent["task_config"].predict_params.output_type == "probas"
        assert sent["tabpfn_systems"] == ["preprocessing", "text"]
        assert len(sent["fold_ids"]) == 1
        assert len(sent["fold_ids"][0]) == N_ROWS

    def test_predict_returns_the_labels_of_the_targets(self):
        with _offline_server():
            labels = TabPFNClassifier().cross_val_predict(X, Y_CLASSES, cv=3)
        assert labels.tolist() == Y_CLASSES.tolist()

    def test_rows_in_no_test_set_are_nan(self):
        rows = np.arange(N_ROWS)
        splits = [(rows[rows != 0], rows[:1]), (rows[rows != 1], rows[1:2])]
        with _offline_server():
            model = TabPFNClassifier()
            probas = model.cross_val_predict(
                X, Y_CLASSES, splits, method="predict_proba"
            )
            labels = model.cross_val_predict(X, Y_CLASSES, splits)
        assert not np.isnan(probas[:2]).any()
        assert np.isnan(probas[2:]).all()
        assert labels[:2].tolist() == ["a", "b"]
        assert all(pd.isna(label) for label in labels[2:])

    def test_numeric_labels_with_rows_in_no_test_set_are_float(self):
        y = np.arange(N_ROWS) % 3
        rows = np.arange(N_ROWS)
        splits = [(rows[rows != 0], rows[:1])]
        with _offline_server(classes=[0, 1, 2]):
            labels = TabPFNClassifier().cross_val_predict(X, y, splits)
        assert labels.dtype == float
        assert labels[0] == 0.0
        assert np.isnan(labels[1:]).all()

    def test_an_unknown_method_raises_before_any_request(self):
        with _offline_server() as server:
            with pytest.raises(ValueError, match="method"):
                TabPFNClassifier().cross_val_predict(
                    X, Y_CLASSES, method=cast(Any, "decision")
                )
        server.cv.assert_not_called()

    def test_thinking_mode_raises_before_any_request(self):
        with _offline_server() as server:
            with pytest.raises(ValueError, match="thinking"):
                TabPFNClassifier(thinking_mode=True).cross_val_predict(X, Y_CLASSES)
        server.cv.assert_not_called()


class TestRegressor:
    @pytest.mark.parametrize("output_type", ["mean", "median", "mode"])
    def test_returns_the_predictions_in_row_order(self, output_type):
        with _offline_server() as server:
            predictions = TabPFNRegressor().cross_val_predict(
                X, Y_REGRESSION, cv=4, output_type=output_type
            )
        assert predictions.tolist() == (10.0 * np.arange(N_ROWS)).tolist()
        assert server.sent["task_config"].predict_params.output_type == output_type
        assert isinstance(server.sent["task_config"], RegressorConfig)

    def test_sends_one_k_fold_vector(self):
        with _offline_server() as server:
            TabPFNRegressor().cross_val_predict(X, Y_REGRESSION, cv=KFold(4))
        assert server.sent["fold_ids"] == [np.repeat(np.arange(4), 3).tolist()]

    def test_an_output_with_several_values_per_row_is_rejected(self):
        with _offline_server() as server:
            with pytest.raises(ValueError, match="output_type"):
                TabPFNRegressor().cross_val_predict(
                    X, Y_REGRESSION, output_type=cast(Any, "quantiles")
                )
        server.cv.assert_not_called()


class TestFittedState:
    @pytest.mark.parametrize(
        ("estimator_cls", "y"),
        [(TabPFNClassifier, Y_CLASSES), (TabPFNRegressor, Y_REGRESSION)],
        ids=["classifier", "regressor"],
    )
    def test_an_unfitted_model_stays_unfitted(self, estimator_cls, y):
        model = estimator_cls()
        with _offline_server():
            model.cross_val_predict(X, y, cv=3)
        with pytest.raises(NotFittedError):
            check_is_fitted(model)

    def test_a_fitted_model_keeps_its_fit(self):
        model = TabPFNClassifier()
        model.model_id_ = FIT_ID
        model.classes_ = np.array([0, 1])
        with _offline_server():
            model.cross_val_predict(X, Y_CLASSES, cv=3)
        assert model.model_id_ == FIT_ID
        assert model.classes_.tolist() == [0, 1]


class TestApiMode:
    @pytest.mark.parametrize("api_mode", list(ApiMode))
    def test_cross_val_predict_runs_in_the_api_mode(self, api_mode):
        with _offline_server() as server:
            TabPFNRegressor(api_mode=api_mode).cross_val_predict(X, Y_REGRESSION)
        assert server.sent["api_mode"] == api_mode
        assert server.sent["wait"] is False

    def test_submit_runs_as_a_background_job_whatever_the_api_mode(self):
        with _offline_server(first="pending") as server:
            TabPFNRegressor(api_mode=ApiMode.SYNC).submit_cross_val_predict(
                X, Y_REGRESSION
            )
        assert server.sent["api_mode"] == ApiMode.ASYNC

    def test_a_pending_answer_is_waited_for_after_the_server_hint(self):
        with _offline_server(first="pending") as server:
            predictions = TabPFNRegressor().cross_val_predict(X, Y_REGRESSION)
        assert predictions.tolist() == (10.0 * np.arange(N_ROWS)).tolist()
        assert server.wait.call_args.kwargs["first_poll_delay_secs"] == 2.0


class TestSubmitAndWait:
    def test_submit_holds_the_job_and_wait_returns_the_predictions(self):
        model = TabPFNClassifier()
        with _offline_server(first="pending") as server:
            assert model.submit_cross_val_predict(X, Y_CLASSES, cv=3) is model
            assert _pending_id(model) == CV_JOB_ID
            server.wait.assert_not_called()
            labels = model.wait_for_cross_val_predict()
        assert labels.tolist() == Y_CLASSES.tolist()
        assert model._pending_cross_val_predict is None

    def test_wait_without_a_pending_job_raises(self):
        with pytest.raises(RuntimeError, match="submit_cross_val_predict"):
            TabPFNRegressor().wait_for_cross_val_predict()

    def test_a_wait_that_stops_keeps_the_job_pending(self):
        model = TabPFNRegressor()
        with _offline_server(
            first="pending", wait_outcome=TimeoutError("late")
        ) as server:
            with pytest.raises(TimeoutError):
                model.cross_val_predict(X, Y_REGRESSION)
            assert _pending_id(model) == CV_JOB_ID
            server.wait.side_effect = None
            server.wait.return_value = _completed()
            assert len(model.wait_for_cross_val_predict()) == N_ROWS
        assert model._pending_cross_val_predict is None

    def test_a_failed_job_is_no_longer_pending(self):
        model = TabPFNRegressor()
        failure = CrossValidateFailedError("boom", cv_job_id=CV_JOB_ID)
        with _offline_server(first="pending", wait_outcome=failure):
            model.submit_cross_val_predict(X, Y_REGRESSION)
            with pytest.raises(CrossValidateFailedError):
                model.wait_for_cross_val_predict()
        assert model._pending_cross_val_predict is None

    def test_a_job_completed_on_submit_replaces_the_pending_job(self):
        model = TabPFNRegressor()
        with _offline_server(first="pending") as server:
            model.submit_cross_val_predict(X, Y_REGRESSION)
            server.cv.side_effect = None
            server.cv.return_value = _completed(NEW_CV_JOB_ID)
            assert len(model.cross_val_predict(X, Y_REGRESSION)) == N_ROWS
        assert model._pending_cross_val_predict is None

    def test_a_completed_job_whose_download_fails_stays_pending(self):
        model = TabPFNRegressor()
        with _offline_server(download_outcome=OSError("reset")) as server:
            with pytest.raises(OSError):
                model.cross_val_predict(X, Y_REGRESSION)
            assert _pending_id(model) == CV_JOB_ID
            server.wait.assert_not_called()
            assert len(model.wait_for_cross_val_predict()) == N_ROWS
        assert model._pending_cross_val_predict is None

    def test_a_new_submit_replaces_the_pending_job(self, caplog):
        model = TabPFNRegressor()
        with _offline_server(first="pending") as server:
            model.submit_cross_val_predict(X, Y_REGRESSION)
            server.cv.side_effect = None
            server.cv.return_value = _pending(NEW_CV_JOB_ID)
            with caplog.at_level("WARNING"):
                model.submit_cross_val_predict(X, Y_REGRESSION)
        assert _pending_id(model) == NEW_CV_JOB_ID
        assert str(CV_JOB_ID) in caplog.text


@pytest.fixture
def mock_server() -> Iterator[MockTabPFNServer]:
    with MockTabPFNServer() as server:
        yield server


def _parquet(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


class TestFoldSizes:
    def test_counts_each_fold_per_repeat(self):
        assert _fold_sizes([[0, 1, 0, -1], [1, 1, 0, 0]], 4) == [[2, 1], [2, 2]]

    @pytest.mark.parametrize(
        "fold_ids",
        [
            [[0, 1, 0]],  # fewer ids than rows
            [[0, 2, 0, 2]],  # fold 1 is empty
            [[-1, -1, -1, -1]],  # no fold
            [[0, 1, -2, 0]],  # below -1
        ],
    )
    def test_unusable_fold_ids_are_rejected(self, fold_ids):
        with pytest.raises(ValueError):
            _fold_sizes(fold_ids, 4)


class TestServiceClient:
    UPLOAD_ID = "00000000-0000-0000-0000-000000000001"
    FOLD_IDS_UPLOAD_ID = "00000000-0000-0000-0000-000000000002"
    FOLD_IDS = [[0, 1, 0, 1, -1, 0]]

    def setup_method(self):
        ServiceClient.authorize("dummy_token")

    def teardown_method(self):
        ServiceClient.reset_authorization()

    @contextmanager
    def _patched(self) -> Generator[_FakeTime]:
        fake_time = _FakeTime()
        settings = ResolvedAsyncSettings(
            use_above_trainset_size_bytes=50 * 1024 * 1024, poll_timeout_secs=100.0
        )
        with (
            patch("tabpfn_client.client.time", fake_time),
            patch.object(ServiceClient, "get_settings", return_value=None),
            patch.object(
                ServiceClient, "_resolve_async_settings", return_value=settings
            ),
            patch.object(ServiceClient, "_upload_to_gcs"),
        ):
            yield fake_time

    def _route_upload(self, mock_server: MockTabPFNServer) -> None:
        assert mock_server.router is not None
        mock_server.router.post("/tabpfn/prepare_train_set_upload").respond(
            409,
            json={
                "message": "duplicate",
                "error_code": "DUPLICATE_TRAIN_SET_UPLOAD",
                "train_set_upload_id": self.UPLOAD_ID,
            },
        )
        mock_server.router.post("/tabpfn/prepare_fold_ids_upload").respond(
            200,
            json={
                "fold_ids_upload_id": self.FOLD_IDS_UPLOAD_ID,
                "fold_ids_info": {
                    "signed_urls": ["https://storage.test/folds"],
                    "expires_at": 0.0,
                    "required_headers": {},
                },
            },
        )

    def _cross_validate(self, api_mode: ApiMode) -> Any:
        return ServiceClient.cross_validate(
            np.arange(12.0).reshape(6, 2),
            np.array([0, 1, 0, 1, 0, 1]),
            fold_ids=self.FOLD_IDS,
            task_config=ClassifierConfig(),
            tabpfn_systems=["preprocessing", "text"],
            api_mode=api_mode,
        )

    def test_sync_polls_a_pending_answer_until_it_completes(self, mock_server):
        self._route_upload(mock_server)
        assert mock_server.router is not None
        sync_route = mock_server.router.post("/tabpfn/cross_validate")
        sync_route.respond(
            200,
            content=b"  \n "
            + json.dumps(
                {"cv_job_id": str(CV_JOB_ID), "status": "pending", "retry_in_secs": 3}
            ).encode(),
            headers={"content-type": "application/json"},
        )
        result_route = mock_server.router.post("/tabpfn/get_cross_validate_result")
        result_route.side_effect = [
            httpx.Response(
                200,
                json={
                    "cv_job_id": str(CV_JOB_ID),
                    "status": "pending",
                    "retry_in_secs": 1.5,
                },
            ),
            httpx.Response(
                200,
                json={
                    "cv_job_id": str(CV_JOB_ID),
                    "status": "completed",
                    "oof_urls": [OOF_URL],
                    "classes": [0, 1],
                },
            ),
        ]
        with (
            self._patched() as fake_time,
            patch.object(ServiceClient, "_upload_to_gcs") as upload_to_gcs,
        ):
            result = self._cross_validate(ApiMode.SYNC)

        assert result.status == "completed"
        assert result.oof_urls == [OOF_URL]
        assert fake_time.sleeps == [3.0, 1.5]
        body = json.loads(sync_route.calls[0].request.content)
        assert body["train_set_upload_id"] == self.UPLOAD_ID
        assert body["fold_ids_upload_id"] == self.FOLD_IDS_UPLOAD_ID
        assert body["fold_sizes"] == [[3, 2]]
        assert "fold_ids" not in body
        (name, data, _), _ = upload_to_gcs.call_args
        assert name == "fold_ids"
        assert pd.read_parquet(io.BytesIO(data))["r0"].tolist() == self.FOLD_IDS[0]
        assert body["task_config"]["task"] == "classification"
        assert "splits" not in body
        polled = json.loads(result_route.calls[0].request.content)
        assert polled == {"cv_job_id": str(CV_JOB_ID)}

    def test_async_submits_a_job(self, mock_server):
        self._route_upload(mock_server)
        assert mock_server.router is not None
        submit_route = mock_server.router.post("/tabpfn/submit_cross_validate_job")
        submit_route.respond(200, json={"cv_job_id": str(CV_JOB_ID)})
        mock_server.router.post("/tabpfn/get_cross_validate_result").respond(
            200,
            json={
                "cv_job_id": str(CV_JOB_ID),
                "status": "completed",
                "oof_urls": [OOF_URL],
            },
        )
        with self._patched():
            result = self._cross_validate(ApiMode.ASYNC)
        assert result.status == "completed"
        assert submit_route.call_count == 1

    def test_a_failed_job_raises_with_its_reason(self, mock_server):
        assert mock_server.router is not None
        mock_server.router.post("/tabpfn/get_cross_validate_result").respond(
            200,
            json={
                "cv_job_id": str(CV_JOB_ID),
                "status": "failed",
                "error": "out of memory",
                "error_code": "GENERAL_ERROR",
            },
        )
        with self._patched(), pytest.raises(CrossValidateFailedError) as exc:
            ServiceClient.wait_for_cross_validate(CV_JOB_ID)
        assert "out of memory" in str(exc.value)
        assert exc.value.cv_job_id == CV_JOB_ID

    def test_an_unknown_job_raises(self, mock_server):
        assert mock_server.router is not None
        mock_server.router.post("/tabpfn/get_cross_validate_result").respond(
            404,
            json={
                "message": "Cross-validation job not found",
                "error_code": "NOT_FOUND",
            },
        )
        with (
            self._patched(),
            pytest.raises(CrossValidateFailedError, match="not found"),
        ):
            ServiceClient.wait_for_cross_validate(CV_JOB_ID)

    def test_a_rejected_request_raises_with_the_server_message(self, mock_server):
        self._route_upload(mock_server)
        assert mock_server.router is not None
        mock_server.router.post("/tabpfn/cross_validate").respond(
            403,
            json={
                "message": "Cross-validation is not enabled for this account.",
                "error_code": "FORBIDDEN",
            },
        )
        with self._patched(), pytest.raises(RuntimeError, match="not enabled"):
            self._cross_validate(ApiMode.SYNC)

    def test_downloads_the_predictions_as_a_frame(self):
        frame = _oof_frame([0, 1, -1], ["a", "b"])
        response = httpx.Response(200, content=_parquet(frame))
        with patch.object(ServiceClient.storage_client, "get", return_value=response):
            (downloaded,) = ServiceClient.download_cross_validate_predictions(
                [OOF_URL], ClientOptions()
            )
        pd.testing.assert_frame_equal(downloaded, frame)
