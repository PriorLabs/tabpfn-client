#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""Azure AI Foundry client for TabPFN.

from tabpfn_client.foundry import TabPFNClassifier, TabPFNRegressor
"""

from tabpfn_client.foundry.estimator import (
    FoundryEndpointError,
    TabPFNClassifier,
    TabPFNRegressor,
)


__all__ = ["FoundryEndpointError", "TabPFNClassifier", "TabPFNRegressor"]
