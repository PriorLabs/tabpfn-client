#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from typing import Any
from tabpfn_client.api_models import ModelLimit, ModelVersion


def model_limit_from_version(
    model_version: ModelVersion, model_limits: dict[Any, ModelLimit]
) -> ModelLimit:
    """Resolve limit of a model to the same or closest previous version limit.

    Raises:
        ValueError: If no model limits are registered at or below the model version.
    """
    sorted_versions = sorted(model_limits.keys())
    for k in reversed(sorted_versions):
        if k <= model_version:
            return model_limits[k]
    raise ValueError(f"No model limits registered at or below {model_version}")


def model_version_from_path(model_path: str) -> ModelVersion:
    for version in ModelVersion:
        if f"-{version.value}-" in model_path:
            return version
    raise ValueError(f"Invalid model path: {model_path}")
