from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from typing import Any
from tabpfn_client.options import get_opts


class ApiCallMode(str, Enum):
    AUTO = "auto"
    SYNC = "sync"
    ASYNC = "async"


@dataclass(frozen=True)
class PredictionResult:
    y_pred: np.ndarray | list[np.ndarray] | dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedAsyncSettings:
    use_above_trainset_size_bytes: int
    poll_timeout_secs: float


@dataclass
class ClientOptions:
    """
    Options for the client.
    Can be used to override default client behavior for a single request.

    Parameters
    ----------
    timeout : float, optional
        Timeout for the request in seconds.
    headers : dict[str, str], optional
        Headers for the request overriding the default headers.
    """

    # Note: timeout=None does not fallback to the client default, rather it disables
    # the timeout altogether.
    timeout: float = get_opts().TABPFN_CLIENT_TIMEOUT
    headers: dict[str, str] = field(default_factory=dict)
