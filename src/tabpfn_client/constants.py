#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import os
import logging
from enum import Enum
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelVersion(str, Enum):
    """Version of the model."""

    V2 = "v2"
    V2_5 = "v2.5"


CACHE_DIR = Path(__file__).parent.resolve() / ".tabpfn"

URL_TABPFN_CLIENT_GITHUB_ISSUES = "https://github.com/priorlabs/tabpfn-client/issues"
URL_PRIOR_LABS_TERMS_AND_CONDITIONS = (
    "https://priorlabs.ai/general-terms-and-conditions"
)
URL_TABPFN_EXTENSIONS_GITHUB_MANY_CLASS_CODE = "https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/many_class/many_class_classifier.py"  # noqa: E501


@cache
def ci_mode_enabled() -> bool:
    val = os.getenv("TABPFN_CI_MODE")
    return str(val).lower() in {"1", "true", "yes", "on"}


@cache
def dedup_datasets_enabled() -> bool:
    # TABPFN_DEDUP_DATASETS: true = enable dedup, false = disable dedup
    # DISABLE_DS_CACHING (legacy): true = disable caching, false = enable caching
    dedup_val = os.getenv("TABPFN_DEDUP_DATASETS")
    disable_val = os.getenv("DISABLE_DS_CACHING")

    if dedup_val is not None:
        enabled = str(dedup_val).lower() in {"1", "true", "yes", "on"}
    elif disable_val is not None:
        enabled = str(disable_val).lower() not in {"1", "true", "yes", "on"}
    else:
        enabled = True

    if not enabled:
        logger.warning("Dataset deduplication is disabled.")

    return enabled


@cache
def force_reupload_enabled() -> bool:
    val = os.getenv("TABPFN_FORCE_REUPLOAD")
    return str(val).lower() in {"1", "true", "yes", "on"}
