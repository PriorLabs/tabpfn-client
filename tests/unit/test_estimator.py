#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import inspect
from typing import Type, Callable
import pytest

from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.sdks.gapi import (
    ClassifierTabPFNConfig,
    RegressorTabPFNConfig,
    ClassifierPredictParams,
    RegressorPredictParams,
)


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

    for param in config_model.model_fields:
        assert param in init_params, (
            f"Field `{param}` is absent from {estimator.__name__}.__init__()."
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
        assert param in predict_params, (
            f"Field `{param}` is absent from {predict_fn.__qualname__}()."
        )
