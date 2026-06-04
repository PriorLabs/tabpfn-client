#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import inspect
import types
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
import pytest

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
    init_hints = get_type_hints(estimator.__init__)

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
    predict_hints = get_type_hints(predict_fn)

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
