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
    """

    values: np.ndarray
    base_values: np.ndarray
    data: pd.DataFrame

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
            feature_names=[str(c) for c in self.data.columns],
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
    for every row. This function predicts the masked rows of many explained
    rows in each call of `predict_fn`.

    Each set of shown features is predicted once per explained row, and the
    all-hidden set once in total. So the number of masked rows is at most
    `len(background) * (1 + len(X) * (1 + 2 * n_permutations * (n_features - 1)))`,
    and fewer when orderings share sets, which is common with few features.

    Args:
        predict_fn: Maps a DataFrame with the columns of `X` to predictions of
            shape (n,) or (n, n_outputs), for example `model.predict_proba`.
        X: The rows to explain.
        background: Rows that supply the values of hidden features, with the
            same columns and dtypes as `X`. 20 to 100 rows sampled from the
            training set is usual.
        n_permutations: Feature orderings sampled per explained row. Each one
            is also evaluated in reverse.
        batch_size: Most masked rows per call of `predict_fn`, and at least
            `len(background)`. A call holds only its own masked rows in
            memory. A remote model limits the rows per request, so large
            inputs need a batch size under that limit. None sends all masked
            rows in one call.
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
    # Masked columns are cast to the dtype of X, which would silently turn
    # unknown categories into NaN and truncate floats into integer columns.
    mismatched = [
        column
        for column in X_df.columns
        if background_df[column].dtype != X_df[column].dtype
    ]
    if mismatched:
        raise ValueError(
            f"background must have the same dtypes as X; columns {mismatched} differ."
        )
    if len(X_df) == 0 or len(background_df) == 0:
        raise ValueError("X and background must each have at least one row.")
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")
    n_background = len(background_df)
    if batch_size is not None and batch_size < n_background:
        raise ValueError("batch_size must be at least len(background).")

    n_rows, n_features = X_df.shape
    rng = np.random.default_rng(random_state)
    # rank[i, p, j]: position of feature j in ordering p of row i. Removing
    # features in one order passes the same coalitions as adding them in the
    # reverse order, so the reversed orderings are the antithetic half.
    rank = rng.random((n_rows, n_permutations, n_features)).argsort(axis=-1)
    rank = np.concatenate([rank, n_features - 1 - rank], axis=1)
    n_orderings = 2 * n_permutations
    # All hidden and all shown are shared by every ordering, so they are
    # evaluated once per row, then the n_features - 1 steps of each ordering.
    steps = rank[:, :, None, :] < np.arange(1, n_features)[:, None]
    masks = np.concatenate(
        [
            np.zeros((n_rows, 1, n_features), dtype=bool),
            np.ones((n_rows, 1, n_features), dtype=bool),
            steps.reshape(n_rows, -1, n_features),
        ],
        axis=1,
    )
    n_masks = masks.shape[1]

    # Predict each (explained row, mask) pair once. All hidden gives the
    # background rows for every explained row, so it is a single pair.
    shown_masks = masks[:, 1:].reshape(-1, n_features)
    shown_rows = np.repeat(np.arange(n_rows), n_masks - 1)
    keys = np.concatenate(
        [
            shown_rows[:, None].astype(">i8").view(np.uint8),
            np.packbits(shown_masks, axis=1),
        ],
        axis=1,
    )
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    pair_rows = np.concatenate([[0], shown_rows[first]])
    pair_masks = np.concatenate(
        [np.zeros((1, n_features), dtype=bool), shown_masks[first]]
    )
    pair_of_mask = np.concatenate(
        [
            np.zeros((n_rows, 1), dtype=int),
            1 + inverse.reshape(n_rows, n_masks - 1),
        ],
        axis=1,
    )

    pairs_per_call = (batch_size or len(pair_masks) * n_background) // n_background
    means = []
    single_output = False
    for start in range(0, len(pair_masks), pairs_per_call):
        stop = min(start + pairs_per_call, len(pair_masks))
        rows = _masked_rows(
            X_df, background_df, pair_masks[start:stop], pair_rows[start:stop]
        )
        predictions = np.asarray(predict_fn(rows))
        if len(predictions) != len(rows):
            raise ValueError(
                f"predict_fn returned {len(predictions)} predictions for "
                f"{len(rows)} rows."
            )
        single_output = predictions.ndim == 1
        means.append(predictions.reshape(stop - start, n_background, -1).mean(axis=1))
    mean = np.concatenate(means)[pair_of_mask]

    hidden, shown = mean[:, 0], mean[:, 1]
    ends = (n_rows, n_orderings, 1, mean.shape[-1])
    path = np.concatenate(
        [
            np.broadcast_to(hidden[:, None, None], ends),
            mean[:, 2:].reshape(n_rows, n_orderings, n_features - 1, -1),
            np.broadcast_to(shown[:, None, None], ends),
        ],
        axis=2,
    )
    # Entry k of the diff is the gain from the k-th feature of the ordering.
    values = np.take_along_axis(np.diff(path, axis=2), rank[..., None], axis=2).mean(
        axis=1
    )

    if single_output:
        values, hidden = values[..., 0], hidden[..., 0]
    return ShapExplanation(values=values, base_values=hidden, data=X_df)


def _masked_rows(
    X: pd.DataFrame, background: pd.DataFrame, masks: np.ndarray, row_index: np.ndarray
) -> pd.DataFrame:
    """Rows ordered (mask, background row): shown features from X, others from background."""
    n_background = len(background)
    shown = np.repeat(masks, n_background, axis=0)
    source = np.repeat(row_index, n_background)
    return pd.DataFrame(
        {
            column: pd.Series(
                np.where(
                    shown[:, j],
                    X[column].to_numpy()[source],
                    np.tile(background[column].to_numpy(), len(masks)),
                )
            ).astype(X[column].dtype)
            for j, column in enumerate(X.columns)
        }
    )
