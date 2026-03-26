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

TABPFN_TOKEN = os.getenv("TABPFN_TOKEN")
TABPFN_API_URL = os.getenv("TABPFN_API_URL")
TABPFN_MAX_THREAD_PER_UPLOAD = int(os.getenv("TABPFN_MAX_THREAD_PER_UPLOAD", 8))
TABPFN_CLIENT_TIMEOUT = float(os.getenv("TABPFN_CLIENT_TIMEOUT", 900.0))
TABPFN_UPLOAD_TIMEOUT = (
    float(os.getenv("TABPFN_UPLOAD_TIMEOUT", 0.0)) or None
)  # no timeout


@cache
def ci_mode_enabled() -> bool:
    val = os.getenv("TABPFN_CI_MODE")
    return str(val).lower() in {"1", "true", "yes", "on"}


@cache
def force_retransform_enabled() -> bool:
    force_retransform = os.getenv("TABPFN_FORCE_RETRANSFORM")
    return str(force_retransform).lower() in {"1", "true", "yes", "on"}


@cache
def force_reupload_enabled() -> bool:
    # `DISABLE_DS_CACHING` is legacy, we keep it for backward compatibility.
    # Note: The new env var has the opposite meaning.
    force_reupload = os.getenv("TABPFN_FORCE_REUPLOAD")
    disable_caching = os.getenv("DISABLE_DS_CACHING")

    if force_reupload is not None:
        enabled = str(force_reupload).lower() in {"1", "true", "yes", "on"}
    elif disable_caching is not None:
        enabled = str(disable_caching).lower() in {"1", "true", "yes", "on"}
    else:
        enabled = False

    return enabled


@cache
def dedup_datasets_enabled() -> bool:
    val = os.getenv("TABPFN_DEDUP_DATASETS", True)
    enabled = str(val).lower() in {"1", "true", "yes", "on"}

    if not enabled:
        logger.warning("Dataset deduplication is disabled.")

    return enabled
