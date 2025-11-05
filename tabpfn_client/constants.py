#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from enum import Enum
from pathlib import Path


class ModelVersion(str, Enum):
    """Version of the model."""

    V2 = "v2"
    V2_5 = "v2.5"


CACHE_DIR = Path(__file__).parent.resolve() / ".tabpfn"
