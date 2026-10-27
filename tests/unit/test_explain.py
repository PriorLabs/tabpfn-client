#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

from itertools import combinations
from math import factorial
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from tabpfn_client import TabPFNClassifier
from tabpfn_client.api_models import GetSettingsResponse
from tabpfn_client.client import ServiceClient
from tabpfn_client.explain import permutation_shap


def _exact_shap(f, x: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Shapley values by enumerating every coalition, with an independent masker."""
    d = len(x)

    def value(shown: tuple[int, ...]) -> float:
        rows = background.copy()
        rows[:, list(shown)] = x[list(shown)]
        return float(f(rows).mean())

    phi = np.zeros(d)
    for j in range(d):
        others = [k for k in range(d) if k != j]
        for size in range(d):
            weight = factorial(size) * factorial(d - size - 1) / factorial(d)
            for coalition in combinations(others, size):
                phi[j] += weight * (value(coalition + (j,)) - value(coalition))
    return phi


def _interactions(rows) -> np.ndarray:
    # The three-way term matters: with only pairwise terms, one ordering
    # evaluated both ways is already exact, so sampling would go untested.
    x = np.asarray(rows, dtype=float)
    return x[:, 0] * x[:, 1] * x[:, 2] + np.sin(x[:, 2]) * x[:, 3] + x[:, 1] ** 2


@pytest.mark.parametrize("n_features", [1, 5])
def test_linear_model_values_are_exact(n_features: int) -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4, n_features))
    background = rng.normal(size=(7, n_features))
    w = rng.normal(size=n_features)

    result = permutation_shap(
        lambda rows: rows.to_numpy() @ w, X, background=background
    )

    np.testing.assert_allclose(result.values, w * (X - background.mean(axis=0)))
    np.testing.assert_allclose(result.base_values, (background @ w).mean())


def test_background_is_sampled_to_n_background() -> None:
    rng = np.random.default_rng(8)
    X, background = rng.normal(size=(3, 4)), rng.normal(size=(50, 4))
    w = rng.normal(size=4)
    calls: list[np.ndarray] = []

    def f(rows) -> np.ndarray:
        calls.append(rows.to_numpy())
        return rows.to_numpy() @ w

    def run(n_background: int | None, random_state: int) -> np.ndarray:
        return permutation_shap(
            f,
            X,
            background=background,
            n_background=n_background,
            random_state=random_state,
        ).base_values

    sampled = run(7, random_state=0)
    # The all-hidden rows come first, and they are the sampled background.
    hidden = calls[0][:7]
    assert len({tuple(row) for row in hidden}) == 7
    assert all((background == row).all(axis=1).any() for row in hidden)
    np.testing.assert_allclose(sampled, (hidden @ w).mean())
    np.testing.assert_allclose(run(7, random_state=0), sampled)
    assert not np.allclose(run(7, random_state=1), sampled)
    np.testing.assert_allclose(run(None, random_state=0), (background @ w).mean())
    np.testing.assert_allclose(run(50, random_state=0), (background @ w).mean())


def test_n_background_below_one_raises() -> None:
    X = np.zeros((2, 3))
    with pytest.raises(ValueError, match="n_background"):
        permutation_shap(
            lambda rows: np.zeros(len(rows)), X, background=X, n_background=0
        )


def test_two_features_values_are_exact() -> None:
    # With two features, one ordering evaluated both ways covers every coalition.
    rng = np.random.default_rng(1)
    X, background = rng.normal(size=(3, 2)), rng.normal(size=(6, 2))

    def f(rows) -> np.ndarray:
        rows = np.asarray(rows)
        return rows[:, 0] * rows[:, 1] + rows[:, 0] ** 3

    result = permutation_shap(
        f, X, background=background, n_permutations=1, random_state=0
    )

    for i in range(len(X)):
        np.testing.assert_allclose(result.values[i], _exact_shap(f, X[i], background))


def test_values_converge_to_exact_shap() -> None:
    rng = np.random.default_rng(2)
    X, background = rng.normal(size=(2, 4)), rng.normal(size=(5, 4))

    result = permutation_shap(
        _interactions, X, background=background, n_permutations=500, random_state=0
    )

    for i in range(len(X)):
        exact = _exact_shap(_interactions, X[i], background)
        np.testing.assert_allclose(result.values[i], exact, atol=0.05)


def test_multi_output_values_add_up_to_predictions() -> None:
    rng = np.random.default_rng(3)
    X, background = rng.normal(size=(5, 4)), rng.normal(size=(6, 4))

    def proba(rows) -> np.ndarray:
        p = 1 / (1 + np.exp(-_interactions(rows)))
        return np.stack([1 - p, p], axis=1)

    result = permutation_shap(proba, X, background=background, n_permutations=3)

    assert result.values.shape == (5, 4, 2)
    assert result.base_values.shape == (5, 2)
    np.testing.assert_allclose(result.base_values + result.values.sum(axis=1), proba(X))


# Each row has 14 masks, so batch_size=25 also builds them one row at a time.
@pytest.mark.parametrize("batch_size", [100, 25])
def test_batch_size_splits_calls_without_changing_values(batch_size: int) -> None:
    rng = np.random.default_rng(4)
    X, background = rng.normal(size=(3, 4)), rng.normal(size=(5, 4))
    call_sizes: list[int] = []

    def f(rows) -> np.ndarray:
        call_sizes.append(len(rows))
        return _interactions(rows)

    whole = permutation_shap(
        f, X, background=background, n_permutations=2, random_state=0
    )
    (n_masked,) = call_sizes

    call_sizes.clear()
    split = permutation_shap(
        f,
        X,
        background=background,
        n_permutations=2,
        batch_size=batch_size,
        random_state=0,
    )
    # Calls hold whole groups of 5 background rows.
    per_call = batch_size // 5 * 5
    assert call_sizes == [per_call] * (n_masked // per_call) + [n_masked % per_call]
    np.testing.assert_allclose(split.values, whole.values)


def _settings(
    max_rows: int = 10**9,
    max_cells: int = 10**9,
    max_bytes: int = 10**9,
    row_pairs_budget: int = 10**12,
) -> GetSettingsResponse:
    limit = {
        "train_set_max_rows": 10**9,
        "train_set_max_cells": 10**9,
        "train_set_max_upload_cells": 10**9,
        "test_set_max_rows": max_rows,
        "test_set_max_cells": max_cells,
        "test_set_max_rows_w_full_regression_output": 10**9,
        "max_cols": 1_000,
        "max_classes": 10,
        "predict_row_pairs_budget": row_pairs_budget,
    }
    return GetSettingsResponse.model_validate(
        {
            "default_model_version": "v2.5",
            "max_model_limit": limit,
            "model_limits": {"v2.5": limit},
            "dataset_max_size_bytes": max_bytes,
            "async_settings": {
                "use_above_trainset_size_bytes": 10**9,
                "poll_timeout_secs": 1.0,
                "predict_use_above": {"train_rows": 10**9, "test_rows": 10**9},
                "thinking_predict_use_above": {
                    "train_rows": 10**9,
                    "test_rows": 10**9,
                },
            },
        }
    )


# 4 float features take 32 bytes per row, 1,000 train rows, 5 background rows.
@pytest.mark.parametrize(
    ("settings", "rows_per_call"),
    [
        (_settings(max_rows=50), 50),
        (_settings(max_cells=160), 40),
        (_settings(max_bytes=30 * 32 + pd.RangeIndex(0).memory_usage()), 30),
        (_settings(row_pairs_budget=1_000 * 20), 20),
    ],
    ids=["rows", "cells", "bytes", "row_pairs_budget"],
)
def test_client_estimator_batches_fit_the_api_limits(
    settings: GetSettingsResponse, rows_per_call: int
) -> None:
    rng = np.random.default_rng(7)
    X, background = rng.normal(size=(3, 4)), rng.normal(size=(5, 4))
    model = TabPFNClassifier()
    model._n_train_rows = 1_000
    call_sizes: list[int] = []

    def predict_proba(self, rows) -> np.ndarray:
        call_sizes.append(len(rows))
        return np.zeros((len(rows), 2))

    with (
        patch.object(ServiceClient, "get_settings", return_value=settings),
        patch.object(TabPFNClassifier, "predict_proba", predict_proba),
    ):
        permutation_shap(
            model.predict_proba, X, background=background, n_permutations=2
        )

    assert max(call_sizes) == rows_per_call
    assert len(call_sizes) > 1


def test_api_limit_below_background_rows_raises() -> None:
    X = np.zeros((2, 3))
    model = TabPFNClassifier()
    with (
        patch.object(ServiceClient, "get_settings", return_value=_settings(max_rows=1)),
        pytest.raises(ValueError, match="background rows"),
    ):
        permutation_shap(model.predict_proba, X, background=X)


def test_each_set_of_shown_features_is_predicted_once() -> None:
    # Two features give each row three sets ({0}, {1}, both) however many
    # orderings are sampled, plus all hidden, which every row shares.
    rng = np.random.default_rng(6)
    X, background = rng.normal(size=(4, 2)), rng.normal(size=(5, 2))
    call_sizes: list[int] = []

    def f(rows) -> np.ndarray:
        call_sizes.append(len(rows))
        x = np.asarray(rows)
        return x[:, 0] * x[:, 1]

    result = permutation_shap(f, X, background=background, n_permutations=20)

    assert call_sizes == [5 * (1 + 3 * 4)]
    for i in range(len(X)):
        np.testing.assert_allclose(result.values[i], _exact_shap(f, X[i], background))


def test_dataframe_columns_and_dtypes_reach_predict_fn() -> None:
    X = pd.DataFrame(
        {
            "size": [1.5, 2.5],
            "colour": pd.Categorical(["red", "blue"], categories=["red", "blue"]),
            "label": ["a", "b"],
        }
    )
    background = pd.DataFrame(
        {
            "size": [0.0, 1.0, 2.0],
            "colour": pd.Categorical(["blue"] * 3, categories=["red", "blue"]),
            "label": ["c", "a", "b"],
        }
    )
    seen: list[pd.DataFrame] = []

    def f(rows: pd.DataFrame) -> np.ndarray:
        seen.append(rows)
        return rows["size"].to_numpy() + (rows["colour"] == "red").to_numpy()

    result = permutation_shap(f, X, background=background, n_permutations=2)

    assert (seen[0].dtypes == X.dtypes).all()
    assert set(seen[0]["label"]) == {"a", "b", "c"}
    np.testing.assert_allclose(result.values[:, 2], 0)
    np.testing.assert_allclose(result.values[:, 0], X["size"] - 1.0)


@pytest.mark.parametrize(
    ("x_column", "background_column"),
    [
        (
            pd.Categorical(["red"], categories=["red", "blue"]),
            pd.Categorical(["blue", "green"], categories=["blue", "green"]),
        ),
        ([1], [0.5, 1.5]),
    ],
    ids=["other-categories", "float-for-int"],
)
def test_background_dtype_mismatch_raises(x_column, background_column) -> None:
    X = pd.DataFrame({"a": x_column})
    background = pd.DataFrame({"a": background_column})

    with pytest.raises(ValueError, match="dtypes"):
        permutation_shap(lambda rows: np.zeros(len(rows)), X, background=background)


@pytest.mark.parametrize(
    ("X", "match"),
    [
        (pd.DataFrame([[1.0, 2.0]], columns=pd.Index(["a", "a"])), "duplicate"),
        (np.zeros((2, 0)), "one column"),
    ],
    ids=["duplicate-columns", "no-columns"],
)
def test_invalid_columns_raise(X, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        permutation_shap(lambda rows: np.zeros(len(rows)), X, background=X)


def test_batch_size_below_background_rows_raises() -> None:
    X = np.zeros((2, 3))
    with pytest.raises(ValueError, match="batch_size"):
        permutation_shap(
            lambda rows: np.zeros(len(rows)), X, background=X, batch_size=1
        )


def test_wrong_prediction_count_raises() -> None:
    X = np.zeros((2, 3))
    with pytest.raises(ValueError, match="predictions"):
        permutation_shap(lambda rows: np.zeros(1), X, background=X)


def test_to_shap_keeps_values() -> None:
    shap = pytest.importorskip("shap")

    rng = np.random.default_rng(5)
    X, background = rng.normal(size=(2, 3)), rng.normal(size=(4, 3))

    result = permutation_shap(
        lambda rows: rows.to_numpy().sum(axis=1), X, background=background
    )
    explanation = result.to_shap()

    assert isinstance(explanation, shap.Explanation)
    np.testing.assert_allclose(explanation.values, result.values)
