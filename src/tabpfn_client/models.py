"""Data classes and type aliases shared by the client and estimators."""

from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from typing import Any, Literal
from uuid import UUID
from tabpfn_client.options import get_opts

from tabpfn_client.api_models import (
    ClassifierTabPFNConfig,
    RegressorTabPFNConfig,
)


TabPFNConfig = ClassifierTabPFNConfig | RegressorTabPFNConfig

# Literal over enum to keep same static checks as the tabpfn package.
FitModeLiteral = Literal["fit_preprocessors", "fit_with_cache"]


class ApiMode(str, Enum):
    """Server API mode: synchronous, asynchronous, or chosen by the client."""

    AUTO = "auto"
    SYNC = "sync"
    ASYNC = "async"


@dataclass(frozen=True)
class FitResult:
    """Outcome of a fit request.

    Attributes:
        fitted_train_set_id: Server-side id of the fitted train set.
        timings: Seconds per stage as reported by the server, or None when it
            reports none.
    """

    fitted_train_set_id: UUID
    # Seconds per stage as reported by the server; None when it reports none.
    timings: dict[str, Any] | None = None


@dataclass(frozen=True)
class PredictionResult:
    """Outcome of a predict request.

    Attributes:
        y_pred: Predictions; an array, a list of arrays, or a dict of arrays
            depending on the output type requested.
        metadata: Additional metadata returned by the server.
        timings: Seconds per stage as reported by the server, or None when it
            reports none.
    """

    y_pred: np.ndarray | list[np.ndarray] | dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    # Seconds per stage as reported by the server; None when it reports none.
    timings: dict[str, Any] | None = None


@dataclass
class ClientOptions:
    """Options for the client.

    Can be used to override default client behavior for a single request.

    Attributes:
        timeout: Timeout for the request in seconds.
        headers: Headers for the request overriding the default headers.
    """

    # Note: timeout=None does not fallback to the client default, rather it disables
    # the timeout altogether.
    timeout: float = get_opts().TABPFN_CLIENT_TIMEOUT
    headers: dict[str, str] = field(default_factory=dict)
