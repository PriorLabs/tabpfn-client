#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

"""Guard against drift between the estimators' explicit ``__init__`` params and the
server-generated tabpfn_config schema.

The estimators list their server-config arguments explicitly (so they show up in
IDEs and stay sklearn-compliant), while the source of truth for what the server
accepts is the generated ``*TabPFNConfigDict``. This test fails loudly if the
server schema grows a config field the estimator neither exposes nor explicitly
opts out of, forcing a conscious add-param-or-ignore decision.
"""

from __future__ import annotations

import inspect
from typing import Type
import pytest

from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor


@pytest.mark.parametrize("estimator", [TabPFNClassifier, TabPFNRegressor])
def test_client_params_include_server_params(estimator: Type[TabPFNClassifier | TabPFNRegressor]):
    init_params = set(inspect.signature(estimator.__init__).parameters) - {"self"}

    for param in estimator._TABPFN_CONFIG_PARAMS:
        assert param in init_params, (
            f"Field `{param}` is absent from {estimator.__name__}.__init__."
        )
