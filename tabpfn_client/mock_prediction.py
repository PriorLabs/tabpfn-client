import threading
import contextlib
from typing import Literal

_thread_local = threading.local()


def is_mock_mode() -> bool:
    """Return whether the current thread should use the mock prediction."""
    return getattr(_thread_local, "use_mock", False)


def set_mock_mode(value: bool):
    """Enable or disable mock mode for the current thread."""
    setattr(_thread_local, "use_mock", value)


def get_cost() -> float:
    """
    Return the accumulated cost for the current thread.
    """
    return getattr(_thread_local, "cost", 0.0)


def reset_cost():
    """
    Reset the accumulated cost for the current thread to zero.
    """
    if hasattr(_thread_local, "cost"):
        _thread_local.cost = 0.0


def mock_predict(
    X,
    task: Literal["classification", "regression"],
    train_set_uid: str,
    config=None,
    predict_params=None,
    X_train=None,
    y_train=None,
):
    if not hasattr(_thread_local, "cost"):
        _thread_local.cost = 0.0
    _thread_local.cost += 2
    return {}


@contextlib.contextmanager
def mock_mode():
    """
    Context manager that enables mock mode in the current thread,
    then restores the previous mode after exiting the block.
    """
    old_value = is_mock_mode()
    set_mock_mode(True)
    reset_cost()
    try:
        yield lambda: get_cost()
    finally:
        set_mock_mode(old_value)
