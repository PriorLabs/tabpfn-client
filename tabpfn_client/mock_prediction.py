import threading
import contextlib
from typing import Literal
import numpy as np
import time
from unittest import mock
import warnings
import logging
import functools
from tabpfn_client.client import ServiceClient
# from tabpfn_client import get_api_usage


_thread_local = threading.local()


def is_mock_mode() -> bool:
    return getattr(_thread_local, "use_mock", False)


def set_mock_mode(value: bool):
    setattr(_thread_local, "use_mock", value)


def get_cost() -> float:
    return getattr(_thread_local, "cost", 0.0)


def increment_cost(value: float):
    setattr(_thread_local, "cost", get_cost() + value)


def set_cost(value: float = 0.0):
    setattr(_thread_local, "cost", value)


def get_mock_time() -> float:
    return getattr(_thread_local, "mock_time")


def set_mock_time(value: float):
    setattr(_thread_local, "mock_time", value)


def increment_mock_time(seconds: float):
    set_mock_time(get_mock_time() + seconds)


def estimate_duration(
    num_rows: int,
    num_features: int,
    task: Literal["classification", "regression"],
    tabpfn_config: dict = {},
) -> float:
    """
    Estimates the duration of a prediction task.
    """
    # Logic comes from _estimate_model_usage in base.py of the TabPFN codebase.
    CONSTANT_COMPUTE_OVERHEAD = 8000
    NUM_SAMPLES_FACTOR = 4
    NUM_SAMPLES_PLUS_FEATURES = 6.5
    CELLS_FACTOR = 0.25
    CELLS_SQUARED_FACTOR = 1.3e-7
    EMBEDDING_SIZE = 192
    NUM_HEADS = 6
    NUM_LAYERS = 12
    FEATURES_PER_GROUP = 2
    GPU_FACTOR = 1e-11
    LATENCY_OFFSET = 1.0

    n_estimators = tabpfn_config.get(
        "n_estimators", 4 if task == "classification" else 8
    )

    num_samples = num_rows
    num_feature_groups = int(np.ceil(num_features / FEATURES_PER_GROUP))

    num_cells = (num_feature_groups + 1) * num_samples
    compute_cost = (EMBEDDING_SIZE**2) * NUM_HEADS * NUM_LAYERS

    base_duration = (
        n_estimators
        * compute_cost
        * (
            CONSTANT_COMPUTE_OVERHEAD
            + num_samples * NUM_SAMPLES_FACTOR
            + (num_samples + num_feature_groups) * NUM_SAMPLES_PLUS_FEATURES
            + num_cells * CELLS_FACTOR
            + num_cells**2 * CELLS_SQUARED_FACTOR
        )
    )

    return round(base_duration * GPU_FACTOR + LATENCY_OFFSET, 3)


def mock_predict(
    X_test,
    task: Literal["classification", "regression"],
    train_set_uid: str,
    X_train,
    y_train,
    config=None,
    predict_params=None,
):
    duration = estimate_duration(
        X_train.shape[0] + X_test.shape[0], X_test.shape[1], task, config
    )
    increment_mock_time(duration)

    cost = (
        (X_train.shape[0] + X_test.shape[0])
        * X_test.shape[1]
        * config.get("n_estimators", 4 if task == "classification" else 8)
    )
    increment_cost(cost)

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
    Context manager that enables mock mode in the current thread.
    """
    old_value = is_mock_mode()
    set_mock_mode(True)
    set_cost(0.0)
    set_mock_time(time.time())

    # Store original logging levels for all loggers
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    loggers.append(logging.getLogger())  # Add root logger
    original_levels = {logger: logger.level for logger in loggers}

    # Suppress all warnings and logging
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Set all loggers to ERROR level
        for logger in loggers:
            logger.setLevel(logging.ERROR)

        with mock.patch("time.time", side_effect=get_mock_time):
            try:
                yield lambda: get_cost()
            finally:
                set_mock_mode(old_value)
                # Restore original logging levels
                for logger in loggers:
                    logger.setLevel(original_levels[logger])


def check_api_credits(func):
    """
    Decorator that first runs the decorated function in mock mode to simulate its credit usage.
    If user has enough credits, function is then executed for real.
    """
    from tabpfn_client import get_access_token

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with mock_mode() as get_cost:
            func(*args, **kwargs)
            credit_estimate = get_cost()
        access_token = get_access_token()
        api_usage = ServiceClient.get_api_usage(access_token)

        if (
            not api_usage["usage_limit"] == -1
            and api_usage["usage_limit"] - api_usage["current_usage"] < credit_estimate
        ):
            raise RuntimeError(
                f"Not enough credits left. Estimated credit usage: {credit_estimate}, credits left: {api_usage['usage_limit'] - api_usage['current_usage']}"
            )
        else:
            print("Enough credits left.")

        return func(*args, **kwargs)

    return wrapper
