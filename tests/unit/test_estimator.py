#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import inspect
import sys
import types
import typing
from enum import Enum
from typing import (
    Annotated,
    Callable,
    Literal,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
from unittest.mock import patch
from uuid import UUID

import numpy as np
import pytest
from eval_type_backport import eval_type_backport
from sklearn.utils.estimator_checks import check_estimator

from tabpfn_client import estimator as estimator_module
from tabpfn_client.client import PredictionResult, ServiceClient
from tabpfn_client.config import Config
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.api_models import (
    ClassifierTabPFNConfig,
    RegressorTabPFNConfig,
    ClassifierPredictParams,
    RegressorPredictParams,
    UnknownEnum,
)

# ``types.UnionType`` (the ``X | Y`` form) only exists on Python 3.10+; fall back
# to an empty tuple so ``isinstance(..., _UNION_TYPES)`` is a no-op on 3.9.
_UNION_TYPE = getattr(types, "UnionType", ())


def _resolve_hints(fn: Callable) -> dict:
    """Like ``typing.get_type_hints`` but works on Python < 3.10.

    The estimators use ``from __future__ import annotations``, so their PEP 604
    ``X | Y`` annotations are stored as strings. ``get_type_hints`` evaluates
    them, which raises ``TypeError`` on 3.9 where ``|`` is unsupported between
    types. In that case we resolve each annotation through ``eval-type-backport``
    (already a runtime dependency), which rewrites ``X | Y`` into ``Union[X, Y]``.
    """
    try:
        return get_type_hints(fn)
    except TypeError:
        globalns = getattr(sys.modules.get(fn.__module__, None), "__dict__", {})
        return {
            name: (
                eval_type_backport(typing.ForwardRef(ann), globalns, None)
                if isinstance(ann, str)
                else ann
            )
            for name, ann in getattr(fn, "__annotations__", {}).items()
        }


def _normalize_type(tp: object) -> object:
    """Reduce a type hint to a comparable core: unwrap ``Annotated[T, ...]`` and
    drop ``None`` from unions, so the server's all-optional ``X | None`` fields
    compare equal to the estimator's concrete ``X`` params."""
    if get_origin(tp) is Annotated:
        tp = get_args(tp)[0]
    if get_origin(tp) is Union or isinstance(tp, _UNION_TYPE):
        args = [a for a in get_args(tp) if a is not type(None)]
        # Special case: a two-element ``Leading | Fallback`` union where the
        # trailing member is only a permissive fallback (used on the server with
        # ``union_mode="left_to_right"``) is compared against the leading member
        # alone. This covers ``Literal[...] | str`` and ``SomeEnum | UnknownEnum``.
        if len(args) == 2 and args[1] in (str, UnknownEnum):
            return _normalize_type(args[0])
        parts = tuple(_normalize_type(a) for a in args)
        return parts[0] if len(parts) == 1 else frozenset(parts)
    # A str-backed ``Enum`` compares equal to the ``Literal`` of its values, so
    # the server's enum fields line up with the estimator's ``Literal[...]`` ones.
    if isinstance(tp, type) and issubclass(tp, Enum):
        return Literal[tuple(member.value for member in tp)]
    return tp


@pytest.mark.parametrize(
    "estimator, config_model",
    [
        (TabPFNClassifier, ClassifierTabPFNConfig),
        (TabPFNRegressor, RegressorTabPFNConfig),
    ],
)
def test_client_config_includes_server_config(
    estimator: Type[TabPFNClassifier | TabPFNRegressor],
    config_model: Type[ClassifierTabPFNConfig | RegressorTabPFNConfig],
):
    init_sig = inspect.signature(estimator.__init__).parameters
    init_params = set(init_sig) - {"self"}
    # Resolve stringized annotations (the estimator uses `from __future__ import
    # annotations`, so raw signature annotations are str, not types).
    init_hints = _resolve_hints(estimator.__init__)

    for param, field in config_model.model_fields.items():
        assert param in init_params, (
            f"Field `{param}` is absent from {estimator.__name__}.__init__()."
        )
        expected = _normalize_type(field.annotation)
        actual = _normalize_type(init_hints[param])
        assert actual == expected, (
            f"Type of `{param}` on {estimator.__name__}.__init__() is {actual}, "
            f"but the server schema expects {expected}."
        )
        default = init_sig[param].default
        assert default is inspect.Parameter.empty or default is None, (
            f"Default of `{param}` on {estimator.__name__}.__init__() is {default!r}, "
            f"but server-backed params must default to None."
        )


@pytest.mark.parametrize(
    "predict_fn, predict_params_model",
    [
        (TabPFNClassifier._predict, ClassifierPredictParams),
        (TabPFNRegressor.predict, RegressorPredictParams),
    ],
)
def test_client_predict_params_include_server_predict_params(
    predict_fn: Callable,
    predict_params_model: Type[ClassifierPredictParams | RegressorPredictParams],
):
    predict_sig = inspect.signature(predict_fn).parameters
    predict_params = set(predict_sig) - {"self"}
    predict_hints = _resolve_hints(predict_fn)

    for param, field in predict_params_model.model_fields.items():
        assert param in predict_params, (
            f"Field `{param}` is absent from {predict_fn.__qualname__}()."
        )
        expected = _normalize_type(field.annotation)
        actual = _normalize_type(predict_hints[param])
        assert actual == expected, (
            f"Type of `{param}` on {predict_fn.__qualname__}() is {actual}, "
            f"but the server schema expects {expected}."
        )
        default = predict_sig[param].default
        assert default is inspect.Parameter.empty or default is None, (
            f"Default of `{param}` on {predict_fn.__qualname__}() is {default!r}, "
            f"but server-backed params must default to None."
        )


# sklearn's `check_estimator` runs a large battery of conformance checks. The
# ones below don't hold for TabPFN's server-backed, in-context-learning design
# (or assume the sklearn input-validation pipeline we intentionally don't use),
# so we mark them as expected failures rather than asserting on them.
_EXPECTED_FAILED_CHECKS = {
    "check_n_features_in_after_fitting": "TabPFN does not set n_features_in_.",
    "check_n_features_in": "TabPFN does not set n_features_in_.",
    "check_dtype_object": "Object-dtype handling differs from sklearn's pipeline.",
    "check_estimators_empty_data_messages": "Empty-data errors are not validated client-side.",
    "check_estimators_nan_inf": "TabPFN accepts NaN/inf inputs.",
    "check_estimator_sparse_tag": "Sparse input is not supported with an sklearn-style message.",
    "check_estimator_sparse_array": "Sparse input is not supported with an sklearn-style message.",
    "check_estimator_sparse_matrix": "Sparse input is not supported with an sklearn-style message.",
    "check_classifier_data_not_an_array": "Non-array array-likes are not coerced.",
    "check_classifiers_train": "Non-array array-likes are not coerced.",
    "check_parameters_default_constructible": "client_options default is materialized in __init__.",
    "check_estimators_overwrite_params": "fit() mutates client_options.headers (e.g. sentry-trace).",
    "check_do_not_raise_errors_in_init_or_set_params": "__init__ evaluates client_options truthiness.",
    "check_fit1d": "1D input is not validated client-side.",
    "check_fit2d_predict1d": "1D input is not validated client-side.",
}


class _FakeInferenceServer:
    """Stand-in for the remote inference service used by ``check_estimator``.

    ``check_estimator`` calls ``fit``/``predict`` many times with datasets of
    varying size and label sets, so a static HTTP mock can't serve it. Instead we
    mock the ``InferenceClient`` boundary directly: ``fit`` receives ``y`` (so we
    know the classes) and ``predict`` receives ``X`` (so we know the row count),
    which lets us return correctly-shaped, valid responses without a network call.
    """

    _DUMMY_ID = UUID("00000000-0000-0000-0000-000000000002")

    def __init__(self):
        self.classes_ = np.array([0])

    def fit(self, X, y, *args, **kwargs) -> UUID:
        self.classes_ = np.unique(np.asarray(y))
        return self._DUMMY_ID

    def predict(self, X, *, task_config, **kwargs) -> PredictionResult:
        n_rows = np.asarray(X).shape[0]
        output_type = task_config.predict_params.output_type
        if output_type == "probas":
            n_classes = len(self.classes_)
            y_pred = np.full((n_rows, n_classes), 1.0 / n_classes)
        else:
            # A constant, valid label keeps predictions deterministic and within
            # ``classes_`` (enough for the structural/consistency checks we run).
            y_pred = np.full(n_rows, self.classes_[0])
        return PredictionResult(y_pred=y_pred, metadata={})


@pytest.mark.parametrize(
    "estimator",
    [TabPFNClassifier],
)
def test_sklearn_compatible(
    estimator: Type[TabPFNClassifier | TabPFNRegressor],
):
    # Run the sklearn conformance suite fully offline: ``init`` (which would
    # trigger interactive auth / open a browser) is neutralized and the remote
    # inference calls are served by an in-process fake, so this is CI-safe.
    fake = _FakeInferenceServer()
    use_server_before = Config.use_server
    Config.use_server = True
    try:
        with (
            patch.object(estimator_module, "init", lambda *a, **k: None),
            patch.object(estimator_module.InferenceClient, "fit", side_effect=fake.fit),
            patch.object(
                estimator_module.InferenceClient, "predict", side_effect=fake.predict
            ),
            patch.object(ServiceClient, "get_model_limits", return_value=None),
        ):
            check_estimator(
                estimator(),
                expected_failed_checks=_EXPECTED_FAILED_CHECKS,
            )
    finally:
        Config.use_server = use_server_before
