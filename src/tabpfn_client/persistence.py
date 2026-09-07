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
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from pydantic import BaseModel, ValidationError
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Self

from tabpfn_client.api_models import PredictionTask

# Constructor params that describe the connection rather than the model. They
# are left out of the record, and a record that carries them is rejected:
# `client_options` holds per-session headers and timeouts (and is not JSON), and
# a loaded estimator should run with the defaults of the process it is loaded
# into.
_TRANSPORT_PARAMS = frozenset({"client_options"})


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


def _read_record(source: str | Path | dict[str, Any]) -> _ModelRecord:
    """Parse a record from a `save_model()` dict or JSON file, failing with a
    message that names the source rather than a bare validation error."""
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

    # NOTE: Some "*_" attributes may be assigned before a fit succeeds (it used
    # to be the case for `classes_`), so we override sklearn's default of
    # inferring fittedness from any "*_" attribute and use `model_id_` as the
    # single source of truth instead.
    def __sklearn_is_fitted__(self) -> bool:
        return getattr(self, "model_id_", None) is not None

    def save_model(self, path: str | Path | None = None) -> dict[str, Any]:
        """Save the fitted model so it can be loaded later without re-fitting.

        The fit itself lives on the TabPFN server; what is saved is a small,
        human-readable record: the id the server assigned to the fit
        (`model_id_`), the estimator's hyperparameters, a classifier's class
        labels and the training-set size. No training data is included.

        `load_model()` turns the record back into a fitted estimator, in a
        later run or on another machine. Fitted models are only visible to the
        account that created them, so the loading process has to authenticate
        with the same account, and they stay usable for as long as the training
        data remains on the server (see `UserDataClient` to delete it).

        Parameters
        ----------
        path : str or Path, optional
            Where to write the record as JSON. Nothing is written when omitted.

        Returns
        -------
        dict
            The record as a JSON-serialisable dict, whether or not `path` was
            given, so it can also be kept elsewhere (a database, an experiment
            tracker) and handed to `load_model()` directly.
        """
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
        ).model_dump(mode="json")
        if path is not None:
            Path(path).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return record

    @classmethod
    def load_model(cls, source: str | Path | dict[str, Any]) -> Self:
        """Re-create a fitted estimator from a record written by `save_model()`.

        The estimator comes back with the saved hyperparameters and fitted
        state, ready to `predict()` without calling `fit()`. No request is made
        here; the first `predict()` authenticates (like `fit()` would) and
        raises `FittedModelNotFoundError` if the server no longer has the model.

        Parameters
        ----------
        source : str, Path or dict
            The path of a file written by `save_model(path)`, or the dict it
            returned.

        Raises
        ------
        ValueError
            If `source` is not a record written by `save_model()`, holds a model
            of the other task (a regression model loaded into a classifier),
            or has parameters this class does not accept, for instance because
            a newer tabpfn-client saved it.
        """
        record = _read_record(source)
        task = _task_of(cls)
        if record.task != task:
            raise ValueError(
                f"Cannot load a {record.task.value} model into {cls.__name__}."
            )
        transport = _TRANSPORT_PARAMS & set(record.params)
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
        unknown = set(record.params) - set(cls._get_param_names())
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
        estimator = cls(**copy.deepcopy(record.params))
        estimator.model_id_ = record.model_id
        estimator._n_train_rows = record.n_train_rows
        if record.classes is not None:
            setattr(estimator, "classes_", np.asarray(record.classes))
        return estimator
