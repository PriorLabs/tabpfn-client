import threading
import contextlib
from typing import Literal
import numpy as np

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
    X_test,
    task: Literal["classification", "regression"],
    train_set_uid: str,
    X_train,
    y_train,
    config=None,
    predict_params=None,
):
    # Accumulate cost for prediction
    if not hasattr(_thread_local, "cost"):
        _thread_local.cost = 0.0

    cost = (
        (X_train.shape[0] + X_test.shape[0])
        * X_test.shape[1]
        * config.get("n_estimators", 4 if task == "classification" else 8)
    )
    _thread_local.cost += cost

    # Return random result in the correct format
    if task == "classification":
        if (
            not predict_params["output_type"]
            or predict_params["output_type"] == "preds"
        ):
            return np.random.rand(X_test.shape[0])
        elif predict_params["output_type"] == "probas":
            probs = np.random.rand(X_test.shape[0], len(np.unique(y_train)))
            return probs / probs.sum(axis=1, keepdims=True)

    elif task == "regression":
        if not predict_params["output_type"] or predict_params["output_type"] == "mean":
            return np.random.rand(X_test.shape[0])
        elif predict_params["output_type"] == "full":
            return {
                "logits": np.random.rand(X_test.shape[0], 5000),
                "mean": np.random.rand(X_test.shape[0]),
                "median": np.random.rand(X_test.shape[0]),
                "mode": np.random.rand(X_test.shape[0]),
                "quantiles": np.random.rand(3, X_test.shape[0]),
                "borders": np.random.rand(5001),
                "ei": np.random.rand(X_test.shape[0]),
                "pi": np.random.rand(X_test.shape[0]),
            }


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
