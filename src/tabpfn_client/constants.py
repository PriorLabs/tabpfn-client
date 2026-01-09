#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from enum import Enum
from pathlib import Path


class ModelVersion(str, Enum):
    """Version of the model."""

    V2 = "v2"
    V2_5 = "v2.5"


CACHE_DIR = Path(__file__).parent.resolve() / ".tabpfn"

URL_TAPFN_CLIENT_GITHUB_ISSUES = "https://github.com/priorlabs/tabpfn-client/issues"
URL_PRIOR_LABS_TERMS_AND_CONDITIONS = "https://priorlabs.ai/general-terms-and-conditions"
URL_TAPFN_EXTENSIONS_GITHUB_MANY_CLASS_CODE = "https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/many_class/many_class_classifier.py"  # noqa: E501