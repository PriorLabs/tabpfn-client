#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""HTTP client for a self-hosted TabPFN inference container.

from tabpfn_client.endpoints import TabPFNClassifier, TabPFNRegressor
"""

from tabpfn_client.endpoints.estimator import TabPFNClassifier, TabPFNRegressor


__all__ = ["TabPFNClassifier", "TabPFNRegressor"]
