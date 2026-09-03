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
    """Best-effort model version of a `model_path` as the caller passed it.

    Accepts checkpoint filenames (`tabpfn-v3-classifier-v3_default.ckpt`) and
    short names (`v3_default`, `v2.5_real`). Names without a version marker are
    the v2 hash names (e.g. `gn2p4bpt`), so they resolve to v2. The server is
    the authority on what a name means; this only picks the limits used for
    client-side pre-flight checks.
    """
    for version in ModelVersion:
        # "v3" cannot shadow "v3.5" here: both patterns need the separator that
        # follows the version, and "v3.5" continues with "." instead.
        if f"-{version.value}-" in model_path or model_path.startswith(
            f"{version.value}_"
        ):
            return version
    return ModelVersion.V2
