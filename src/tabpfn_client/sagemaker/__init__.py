#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
"""SageMaker BYOC client for TabPFN.

Users subscribed to the TabPFN AWS Marketplace listing deploy the BYOC
container to a SageMaker real-time endpoint and invoke it through this
submodule. The wire protocol is the inline JSON form accepted by the
container's /invocations route; auth is the standard boto3 credential chain.

    from tabpfn_client.sagemaker import TabPFNClassifier, TabPFNRegressor
"""

from tabpfn_client.sagemaker.estimator import TabPFNClassifier, TabPFNRegressor


__all__ = ["TabPFNClassifier", "TabPFNRegressor"]
