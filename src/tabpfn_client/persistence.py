#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Save and load fitted estimators, so a fit outlives the process that made it.

A TabPFN fit lives on the server: `fit()` uploads the training data, the server
fits and returns an id, and every later `predict()` refers to that id. The
fitted state held on the client is therefore small and fully describable:

- `model_id_`, the id the server assigned to the fit;
- the hyperparameters, which travel with the id on every predict request;
- the class labels of a classifier;
- the number of training rows, which bounds the test rows per predict call.

`ModelPersistenceMixin` owns that state: it declares the slots `fit()` writes,
derives fittedness from them, and turns them into a small JSON record with
`save_model()` and back into a ready-to-predict estimator with `load_model()`.
Nothing else needs to survive, which is also what keeps pickling cheap.

The same record lets a fit outlive the process that started it. The server
assigns the id when it accepts a fit, before it runs it, so a model that
`submit_fit()` left pending can be saved too. Its record says the fit is
pending, and `load_model()` restores it as such, without asking the server
about it, for `wait_for_fit()` to wait for. A pending prediction travels the
same way: the record holds its id, for `wait_for_predict()` to wait for.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import UUID, uuid4

import numpy as np
from pydantic import BaseModel, ValidationError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self

from tabpfn_client.api_models import FitStatus, PredictionTask

# Constructor params that describe the connection rather than the model: they
# change how the client talks to the server, never what the server computes.
# They are left out of the record, so that a loaded estimator runs with the
# defaults of the process it is loaded into. `client_options` holds per-session
# headers and timeouts (and is not JSON), and `api_mode` how the client sends
# requests.
_TRANSPORT_PARAMS = frozenset({"client_options", "api_mode"})
# The ones a record may still carry, which loading drops: tabpfn-client 0.6
# saved `api_mode`. A record that carries any other is rejected.
_DROPPED_TRANSPORT_PARAMS = frozenset({"api_mode"})


@dataclass(frozen=True)
class _PendingPredict:
    """A prediction the model holds until `wait_for_predict()` returns it.

    One `submit_predict()` started, or one `predict()` stopped waiting for.

    Attributes:
        fitted_test_set_id: The id of the prediction.
        output_type: The output it was asked for, which shapes what
            `wait_for_predict()` returns.
    """

    fitted_test_set_id: UUID
    output_type: str


class _ModelRecord(BaseModel):
    """What `save_model()` writes. Field names are part of the file format.

    `tabpfn_client_version` is the version that wrote the record; it is
    informational today and lets error messages say where a file came from.
    """

    tabpfn_client_version: str
    task: PredictionTask
    model_id: UUID
    params: dict[str, Any]
    # None when the estimator was made fitted by assigning `model_id_` directly,
    # in which case the client cannot bound the test rows per call.
    n_train_rows: int | None = None
    # Classification only; None for regressors.
    classes: list[Any] | None = None
    # PENDING for a fit `submit_fit()` started that had not been waited for when
    # the record was saved, so the server may still be running it. A failed fit
    # is never saved. Records written before this field existed only held
    # completed fits.
    fit_status: Literal[FitStatus.PENDING, FitStatus.COMPLETED] = FitStatus.COMPLETED
    # The prediction the model held when the record was saved, which the server
    # may still be running. Records written before this field existed held none.
    pending_prediction: _PendingPredict | None = None


def _task_of(estimator_cls: type) -> PredictionTask:
    # sklearn's mixin is what makes an estimator a classifier, and with it what
    # entitles it to `classes_`.
    if issubclass(estimator_cls, ClassifierMixin):
        return PredictionTask.CLASSIFICATION
    return PredictionTask.REGRESSION


def _installed_version() -> str:
    # get_client_version() fallback is not what we want here.
    # Imported here to avoid circular import.
    from tabpfn_client import __version__

    return __version__


def _jsonable(value: Any) -> Any:
    """Return `value` with numpy scalars and arrays replaced by Python ones.

    `fit()` accepts them as hyperparameters (an `np.int64` out of a parameter
    grid, an index array from `np.where`), but JSON has no representation for
    them.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_record(path: str | Path, record: dict[str, Any]) -> None:
    """Write `record` to `path` as JSON, replacing the file in one step.

    Written to a file of its own beside `path` and renamed over it, so that a
    process stopped mid-write leaves the previous record whole instead of a
    truncated one, and processes saving to one path at once each install a
    whole record.
    """
    path = Path(path)
    aside = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        aside.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        os.replace(aside, path)
    except BaseException:
        aside.unlink(missing_ok=True)
        raise


def _read_record(source: str | Path | dict[str, Any]) -> _ModelRecord:
    """Parse a record from a `save_model()` dict or JSON file.

    Args:
        source: The dict returned by `save_model()`, or the path of the JSON
            file it wrote.

    Returns:
        The validated record.

    Raises:
        ValueError: If `source` is not a valid record; the message names the
            source rather than showing a bare validation error.
    """
    if isinstance(source, dict):
        raw: Any = source
        origin = "The given dict"
    else:
        origin = str(source)
        try:
            raw = json.loads(Path(source).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(
                f"{origin} is not a model file written by save_model(): {exc}"
            ) from exc
    try:
        return _ModelRecord.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(
            f"{origin} is not a model record written by save_model(): {exc}"
        ) from exc


class ModelPersistenceMixin(BaseEstimator):
    """Fitted state of an estimator whose fit lives on the TabPFN server.

    Declares the slots `fit()` fills, derives fittedness from them, and
    persists them through `save_model()` / `load_model()`. Extends
    `BaseEstimator` because the record is built from `get_params()`.
    """

    # The server-side id predictions run against. Written by `fit()`, restored
    # by `load_model()`, or assigned directly to reuse a previous fit; absent on
    # unfitted instances, which is what `__sklearn_is_fitted__` keys on.
    model_id_: UUID  # annotation only, no class attribute

    # Rows in the fitted training set. Prediction compute scales with
    # n_train_rows * n_test_rows, so this bounds the test rows per call.
    _n_train_rows: int | None = None

    # Whether the server may still be running the fit `model_id_` refers to:
    # set by `submit_fit()`, or by `load_model()` from a record of a pending
    # fit, until `wait_for_fit()` sees the fit complete.
    _fit_pending: bool = False

    # The prediction that `submit_predict()` started, or that `predict()`
    # stopped waiting for, until `wait_for_predict()` returns it. Class-level
    # default so estimators pickled before this attribute existed read it as
    # None.
    _pending_predict: _PendingPredict | None = None

    # NOTE: Some "*_" attributes may be assigned before a fit succeeds (it used
    # to be the case for `classes_`), so we override sklearn's default of
    # inferring fittedness from any "*_" attribute and use `model_id_` as the
    # single source of truth instead. A pending fit has its `model_id_`, but
    # nothing to predict with until `wait_for_fit()` has seen it complete.
    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "model_id_", None) is not None and not self._fit_pending

    def save_model(self, path: str | Path | None = None) -> dict[str, Any]:
        """Save the fitted model so it can be loaded later without re-fitting.

        The fit itself lives on the TabPFN server; what is saved is a small,
        human-readable record: the id the server assigned to the fit
        (`model_id_`), the estimator's hyperparameters, a classifier's class
        labels and the training-set size. No training data is included, nor
        `api_mode` and `client_options`, which describe how the client talks to
        the server: the loading process sets its own.

        `load_model()` turns the record back into a fitted estimator, in a
        later run or on another machine. Fitted models are only visible to the
        account that created them, so the loading process has to authenticate
        with the same account, and they stay usable for as long as the training
        data remains on the server (see `UserDataClient` to delete it).

        A model `submit_fit()` left pending is saved too, as a pending fit, so
        that a later run can load it and `wait_for_fit()`. So is a pending
        prediction, one `submit_predict()` started or `predict()` stopped
        waiting for: a later run can load the model and `wait_for_predict()`,
        for at least a day after the prediction completes, while the server
        keeps its result.

        Args:
            path: Where to write the record as JSON. Nothing is written when
                omitted.

        Returns:
            The record as a JSON-serialisable dict, whether or not `path` was
            given, so it can also be kept elsewhere (a database, an experiment
            tracker) and handed to `load_model()` directly.

        Raises:
            NotFittedError: If the model has no fit, completed or pending.
        """
        # A pending fit is not fitted yet, but saving it is what lets a later
        # run wait for it.
        if not self._fit_pending:
            check_is_fitted(self)
        params = {
            k: _jsonable(v)
            for k, v in self.get_params(deep=False).items()
            if k not in _TRANSPORT_PARAMS
        }
        classes = getattr(self, "classes_", None)
        record = _ModelRecord(
            tabpfn_client_version=_installed_version(),
            task=_task_of(type(self)),
            model_id=self.model_id_,
            params=params,
            n_train_rows=self._n_train_rows,
            classes=None if classes is None else np.asarray(classes).tolist(),
            fit_status=FitStatus.PENDING if self._fit_pending else FitStatus.COMPLETED,
            pending_prediction=self._pending_predict,
        ).model_dump(mode="json")
        if path is not None:
            _write_record(path, record)
        return record

    @classmethod
    def load_model(cls, source: str | Path | dict[str, Any]) -> Self:
        """Re-create a fitted estimator from a record written by `save_model()`.

        The estimator comes back with the saved hyperparameters and fitted
        state, ready to `predict()` without calling `fit()`, and with the
        default `api_mode` and `client_options`: set them on the loaded
        estimator to change them. A record from tabpfn-client 0.6 that holds
        `api_mode` loads too, without it. No request is made
        here; the first `predict()` authenticates (like `fit()` would) and
        raises `FittedModelNotFoundError` if the server no longer has the model.

        A record of a fit that was still pending when it was saved comes back
        pending, as `submit_fit()` left it: call `wait_for_fit()` before
        `predict()`. A record that holds a pending prediction comes back
        holding it, for `wait_for_predict()`.

        Args:
            source: The path of a file written by `save_model(path)`, or the
                dict it returned.

        Returns:
            A fitted estimator of this class, or a pending one if the record
            holds a pending fit.

        Raises:
            ValueError: If `source` is not a record written by `save_model()`,
                holds a model of the other task (a regression model loaded into
                a classifier), or has parameters this class does not accept,
                for instance because a newer tabpfn-client saved it.
        """
        record = _read_record(source)
        task = _task_of(cls)
        if record.task != task:
            raise ValueError(
                f"Cannot load a {record.task.value} model into {cls.__name__}."
            )
        params = {
            k: v for k, v in record.params.items() if k not in _DROPPED_TRANSPORT_PARAMS
        }
        transport = _TRANSPORT_PARAMS & set(params)
        if transport:
            raise ValueError(
                f"Cannot load this model into {cls.__name__}: {sorted(transport)} "
                "describe the connection rather than the model and do not belong "
                "in a model record. Drop them, and set them on the loaded "
                "estimator instead."
            )
        # Unknown parameters are an error, not silently dropped: the record is
        # the inference config (`_get_tabpfn_config()` rebuilds the request from
        # `get_params()` on every `predict()`), so a key this class does not know
        # is almost always one a newer tabpfn-client added that changes what the
        # server computes. Dropping it would give different predictions from the
        # ones the user validated when saving, with no signal. Unlike xgboost or
        # CatBoost files, nothing else (no weights) pins the behaviour.
        unknown = set(params) - set(cls._get_param_names())
        if unknown:
            advice = "Drop them from the record."
            installed = _installed_version()
            if record.tabpfn_client_version != installed:
                advice = (
                    f"It was saved with tabpfn-client {record.tabpfn_client_version} "
                    f"(installed: {installed}); upgrade tabpfn-client or drop them "
                    "from the record."
                )
            raise ValueError(
                f"Cannot load this model into {cls.__name__}: it has parameters "
                f"{cls.__name__} does not accept: {sorted(unknown)}. {advice}"
            )
        # Deep-copied so that estimators loaded from the same dict share nothing
        # mutable with it, or with each other (as `sklearn.base.clone` does).
        estimator = cls(**copy.deepcopy(params))
        estimator.model_id_ = record.model_id
        estimator._n_train_rows = record.n_train_rows
        if record.classes is not None:
            setattr(estimator, "classes_", np.asarray(record.classes))
        estimator._fit_pending = record.fit_status == FitStatus.PENDING
        if record.pending_prediction is not None:
            estimator._pending_predict = record.pending_prediction
        return estimator
