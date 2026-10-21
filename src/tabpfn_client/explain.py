"""Permutation SHAP values computed with few prediction requests."""

#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ShapExplanation:
    """SHAP values for a set of explained rows.

    Attributes:
        values: Per-feature attributions, shape (n_rows, n_features), or
            (n_rows, n_features, n_outputs) for a model with several outputs
            such as `predict_proba`.
        base_values: Mean prediction over the background rows, shape (n_rows,)
            or (n_rows, n_outputs). For each row,
            `base_values + values.sum(axis=1)` equals the model's prediction.
        data: The explained rows.
        feature_names: The column names of the explained rows.
    """

    values: np.ndarray
    base_values: np.ndarray
    data: pd.DataFrame
    feature_names: list[str]

    def to_shap(self) -> Any:
        """Convert to a `shap.Explanation`, for use with the plots of `shap`.

        Returns:
            A `shap.Explanation` with the same values.

        Raises:
            ImportError: If `shap` is not installed.
        """
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "to_shap() needs the shap package: pip install shap"
            ) from e
        return shap.Explanation(
            values=self.values,
            base_values=self.base_values,
            data=self.data.to_numpy(),
            feature_names=self.feature_names,
        )


def permutation_shap(
    predict_fn: Callable[[pd.DataFrame], Any],
    X: pd.DataFrame | np.ndarray,
    background: pd.DataFrame | np.ndarray,
    *,
    n_permutations: int = 10,
    batch_size: int | None = None,
    random_state: int | None = None,
) -> ShapExplanation:
    """Estimate SHAP values by antithetic permutation sampling.

    This is the algorithm of `shap.PermutationExplainer` with an independent
    masker: a hidden feature takes its values from each background row, and
    the prediction is averaged over the background rows. `shap` calls the
    model once per explained row, so a remote model pays the request overhead
    for every row. This function builds every masked row first and predicts
    them all with `ceil(n_masked_rows / batch_size)` calls of `predict_fn`.

    The number of masked rows is
    `len(X) * len(background) * (2 + 2 * n_permutations * (n_features - 1))`.

    Args:
        predict_fn: Maps a DataFrame with the columns of `X` to predictions of
            shape (n,) or (n, n_outputs), for example `model.predict_proba`.
        X: The rows to explain.
        background: Rows that supply the values of hidden features. 20 to 100
            rows sampled from the training set is usual.
        n_permutations: Feature orderings sampled per explained row. Each one
            is evaluated forwards and backwards.
        batch_size: Most masked rows per call of `predict_fn`. None sends all
            masked rows in one call.
        random_state: Seed for the sampled feature orderings.

    Returns:
        The SHAP values of each row of `X`.

    Raises:
        ValueError: If an argument is invalid, or `predict_fn` returns the
            wrong number of predictions.
    """
    X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    background_df = (
        background
        if isinstance(background, pd.DataFrame)
        else pd.DataFrame(background, columns=X_df.columns)
    )
    if list(background_df.columns) != list(X_df.columns):
        raise ValueError("background must have the same columns as X.")
    if len(X_df) == 0 or len(background_df) == 0:
        raise ValueError("X and background must each have at least one row.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    n_rows, n_features = X_df.shape
    rng = np.random.default_rng(random_state)
    orders = np.stack(
        [
            [rng.permutation(n_features) for _ in range(n_permutations)]
            for _ in range(n_rows)
        ]
    )
    # rank[i, p, j]: position of feature j in ordering p of row i.
    rank = np.argsort(orders, axis=-1)
    sizes = np.arange(1, n_features)[:, None]
    # Shown features after adding the first k features of an ordering, and
    # after removing them, for k in 1..n_features-1. All shown and all hidden
    # are shared by every ordering, so they are evaluated once per row.
    added = rank[:, :, None, :] < sizes
    removed = rank[:, :, None, :] >= sizes
    masks = np.concatenate(
        [
            np.zeros((n_rows, 1, n_features), dtype=bool),
            np.ones((n_rows, 1, n_features), dtype=bool),
            np.concatenate([added, removed], axis=2).reshape(n_rows, -1, n_features),
        ],
        axis=1,
    )
    n_masks = masks.shape[1]
    n_background = len(background_df)

    # Masked rows are ordered (explained row, mask, background row).
    masked = pd.DataFrame(
        {
            column: pd.Series(
                np.where(
                    np.repeat(masks[:, :, j].ravel(), n_background),
                    np.repeat(X_df[column].to_numpy(), n_masks * n_background),
                    np.tile(background_df[column].to_numpy(), n_rows * n_masks),
                )
            ).astype(X_df[column].dtype)
            for j, column in enumerate(X_df.columns)
        }
    )

    step = batch_size or len(masked)
    predictions = np.concatenate(
        [
            np.asarray(predict_fn(masked.iloc[start : start + step]))
            for start in range(0, len(masked), step)
        ]
    )
    if len(predictions) != len(masked):
        raise ValueError(
            f"predict_fn returned {len(predictions)} predictions for "
            f"{len(masked)} rows."
        )
    single_output = predictions.ndim == 1
    predictions = predictions.reshape(len(masked), -1)
    n_outputs = predictions.shape[1]

    mean = predictions.reshape(n_rows, n_masks, n_background, n_outputs).mean(axis=2)
    hidden, shown = mean[:, 0], mean[:, 1]
    steps = mean[:, 2:].reshape(n_rows, n_permutations, 2, n_features - 1, n_outputs)
    per_ordering = (n_rows, n_permutations, 1, n_outputs)
    adding = np.concatenate(
        [
            np.broadcast_to(hidden[:, None, None], per_ordering),
            steps[:, :, 0],
            np.broadcast_to(shown[:, None, None], per_ordering),
        ],
        axis=2,
    )
    removing = np.concatenate(
        [
            np.broadcast_to(shown[:, None, None], per_ordering),
            steps[:, :, 1],
            np.broadcast_to(hidden[:, None, None], per_ordering),
        ],
        axis=2,
    )
    # Entry k is the change from the k-th feature of the ordering, both ways.
    gains = np.diff(adding, axis=2) - np.diff(removing, axis=2)
    by_feature = np.take_along_axis(gains, rank[..., None], axis=2)
    values = by_feature.sum(axis=1) / (2 * n_permutations)

    if single_output:
        values, hidden = values[..., 0], hidden[..., 0]
    return ShapExplanation(
        values=values,
        base_values=hidden,
        data=X_df,
        feature_names=[str(c) for c in X_df.columns],
    )
