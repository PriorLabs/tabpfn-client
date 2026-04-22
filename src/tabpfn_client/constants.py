#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0
from __future__ import annotations

import os
import logging
from enum import Enum
from functools import cache
from pathlib import Path
from tabpfn_client.api_models import ModelLimit

logger = logging.getLogger(__name__)


# TODO: This should be exported by a tabpfn-types package at some point.
class ModelVersion(str, Enum):
    V2 = "v2"
    V2_5 = "v2.5"
    V2_6 = "v2.6"
    V3 = "v3"

    def model_limit(self, model_limits: dict[str, ModelLimit]) -> ModelLimit:
        """Resolve limit of a model to the same or closest previous version limit.

        Raises:
            ValueError: If no model limits are registered at or below the model version.
        """
        sorted_versions = sorted(model_limits.keys())
        for k in reversed(sorted_versions):
            if k <= self:
                return model_limits[k]
        raise ValueError(f"No model limits registered at or below {self}")

    @staticmethod
    def from_model_path(model_path: str) -> ModelVersion:
        for version in ModelVersion:
            if f"-{version}-" in model_path:
                return version
        raise ValueError(f"Invalid model path: {model_path}")


CACHE_DIR = Path(__file__).parent.resolve() / ".tabpfn"

URL_TABPFN_CLIENT_GITHUB_ISSUES = "https://github.com/priorlabs/tabpfn-client/issues"
URL_PRIOR_LABS_TERMS_AND_CONDITIONS = (
    "https://priorlabs.ai/general-terms-and-conditions"
)
URL_TABPFN_EXTENSIONS_GITHUB_MANY_CLASS_CODE = "https://github.com/PriorLabs/tabpfn-extensions/blob/main/src/tabpfn_extensions/many_class/many_class_classifier.py"  # noqa: E501

TABPFN_TOKEN = os.getenv("TABPFN_TOKEN")

# Client specific constants
TABPFN_CLIENT_API_URL = os.getenv("TABPFN_CLIENT_API_URL")
TABPFN_CLIENT_MAX_THREAD_PER_UPLOAD = int(
    os.getenv("TABPFN_CLIENT_MAX_THREAD_PER_UPLOAD", 8)
)
TABPFN_CLIENT_TIMEOUT = float(os.getenv("TABPFN_CLIENT_TIMEOUT", 900.0))
TABPFN_CLIENT_UPLOAD_TIMEOUT = float(os.getenv("TABPFN_CLIENT_UPLOAD_TIMEOUT", 7200.0))


@cache
def ci_mode_enabled() -> bool:
    val = os.getenv("TABPFN_CLIENT_CI_MODE")
    return str(val).lower() in {"1", "true", "yes", "on"}


@cache
def force_refit_enabled() -> bool:
    force_refit = os.getenv("TABPFN_CLIENT_FORCE_REFIT")
    return str(force_refit).lower() in {"1", "true", "yes", "on"}


@cache
def force_reupload_enabled() -> bool:
    # `DISABLE_DS_CACHING` is legacy, we keep it for backward compatibility.
    # Note: The new env var has the opposite meaning.
    force_reupload = os.getenv("TABPFN_CLIENT_FORCE_REUPLOAD")
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
    val = os.getenv("TABPFN_CLIENT_DEDUP_DATASETS", True)
    enabled = str(val).lower() in {"1", "true", "yes", "on"}

    if not enabled:
        logger.warning("Dataset deduplication is disabled.")

    return enabled
