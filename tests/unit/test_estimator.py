#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import inspect
import types
from typing import (
    Annotated,
    Callable,
    Type,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)
import pytest

from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.sdks.gapi import (
    ClassifierTabPFNConfig,
    RegressorTabPFNConfig,
    ClassifierPredictParams,
    RegressorPredictParams,
)

# Server-schema fields the client intentionally does NOT surface as estimator
# arguments, so we skip them entirely (presence and type). Prefer exposing a
# field over adding it here.
#   - categorical_features_indices: not currently part of the client surface.
#   - model_id: an internal predict-time field, not a user-facing argument.
_NOT_EXPOSED = {"categorical_features_indices", "model_id"}

# Fields whose estimator annotation intentionally differs from the server schema,
# so we check only their presence, not their type:
#   - random_state: the estimator also accepts np.random.RandomState/Generator,
#     a deliberately broader client-side surface than the server's `int | None`.
#   - inference_precision: the estimator pins the tighter `Literal["autocast",
#     "auto"]`; the server type is a looser Optional[Union[Literal[...], str]].
_TYPE_CHECK_IGNORE = {"random_state", "inference_precision"}


def _normalize_type(tp: object) -> object:
    """Reduce a type hint to a comparable core: unwrap ``Annotated[T, ...]`` and
    drop ``None`` from unions, so the server's all-optional ``X | None`` fields
    compare equal to the estimator's concrete ``X`` params."""
    if get_origin(tp) is Annotated:
        tp = get_args(tp)[0]
    if get_origin(tp) is Union or isinstance(tp, types.UnionType):
        parts = tuple(_normalize_type(a) for a in get_args(tp) if a is not type(None))
        return parts[0] if len(parts) == 1 else frozenset(parts)
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
    init_params = set(inspect.signature(estimator.__init__).parameters) - {"self"}
    # Resolve stringized annotations (the estimator uses `from __future__ import
    # annotations`, so raw signature annotations are str, not types).
    init_hints = get_type_hints(estimator.__init__)

    for param, field in config_model.model_fields.items():
        if param in _NOT_EXPOSED:
            continue
        assert param in init_params, (
            f"Field `{param}` is absent from {estimator.__name__}.__init__()."
        )
        if param in _TYPE_CHECK_IGNORE:
            continue
        expected = _normalize_type(field.annotation)
        actual = _normalize_type(init_hints[param])
        assert actual == expected, (
            f"Type of `{param}` on {estimator.__name__}.__init__() is {actual}, "
            f"but the server schema expects {expected}."
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
    predict_params = set(inspect.signature(predict_fn).parameters) - {"self"}

    for param in predict_params_model.model_fields:
        if param in _NOT_EXPOSED:
            continue
        assert param in predict_params, (
            f"Field `{param}` is absent from {predict_fn.__qualname__}()."
        )
