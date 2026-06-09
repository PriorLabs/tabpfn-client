#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0
"""SageMaker BYOC client for TabPFN.

from tabpfn_client.sagemaker import TabPFNClassifier, TabPFNRegressor
"""

from tabpfn_client.sagemaker.estimator import TabPFNClassifier, TabPFNRegressor


__all__ = ["TabPFNClassifier", "TabPFNRegressor"]
