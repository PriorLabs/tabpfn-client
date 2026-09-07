#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from importlib.metadata import PackageNotFoundError, version

from tabpfn_client.config import (
    init,
    reset,
    get_access_token,
    set_access_token,
    get_api_usage,
)
from tabpfn_client.estimator import TabPFNClassifier, TabPFNRegressor
from tabpfn_client.errors import FittedModelNotFoundError
from tabpfn_client.interactive_auth import InteractiveLoginError, interactive_login
from tabpfn_client.service_wrapper import UserDataClient

try:
    __version__ = version("tabpfn-client")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "init",
    "reset",
    "TabPFNClassifier",
    "TabPFNRegressor",
    "FittedModelNotFoundError",
    "UserDataClient",
    "get_access_token",
    "set_access_token",
    "get_api_usage",
    "interactive_login",
    "InteractiveLoginError",
]
