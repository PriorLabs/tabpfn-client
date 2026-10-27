"""Permutation SHAP values computed with few prediction requests."""

#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from tabpfn_client.client import ServiceClient
from tabpfn_client.estimator import (
    TabPFNClassifier,
    TabPFNRegressor,
    _limit_for_model_path,
    _test_row_limit,
)

# Most rows per call. The API charges more tokens per predicted row as a request
# grows, and much smaller requests pay the minimum charge per request. See
# `estimate_cost`. 100 float features take 40 MB per batch.
DEFAULT_BATCH_SIZE = 50_000
# The API rejects a predict of more rows against a cached fit.
CACHED_FIT_BATCH_SIZE = 10_000


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
    *,
    background: pd.DataFrame | np.ndarray,
    n_background: int | None = 100,
    n_permutations: int = 2,
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
    `n_background * (1 + len(X) * (1 + 2 * n_permutations * (n_features - 1)))`,
    and fewer when orderings share sets, which is common with few features.

    Args:
        predict_fn: Maps a DataFrame with the columns of `X` to predictions of
            shape (n,) or (n, n_outputs), for example `model.predict_proba`.
        X: The rows to explain.
        background: Rows to sample the values of hidden features from, usually
            the training data. It must have the same columns and dtypes as `X`.
        n_background: Rows sampled from `background` without replacement. The
            cost grows linearly with it. None, or a value of at least
            `len(background)`, uses every row.
        n_permutations: Feature orderings sampled per explained row. Each one
            is also evaluated in reverse. The cost grows linearly with it. For
            the same cost, more background rows reduce the error more than
            more orderings, up to about 100 background rows.
        batch_size: Most masked rows per call of `predict_fn`, and at least
            `n_background`. Memory holds one batch of masked rows, plus a
            part that grows with `len(X) * n_permutations * n_features`. None
            uses `DEFAULT_BATCH_SIZE`, or fewer rows when `predict_fn` is a
            method of a fitted `TabPFNClassifier` or `TabPFNRegressor` and the
            API accepts fewer in one request. A model fitted with
            `fit_mode="fit_with_cache"` gets `CACHED_FIT_BATCH_SIZE`. The byte
            limit is estimated from the average row size, so very uneven text
            columns can still exceed it.
        random_state: Seed for the sampled background rows and feature
            orderings.

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
    if not X_df.columns.is_unique:
        raise ValueError("X must not have duplicate column names.")
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
    if len(X_df) == 0 or len(background_df) == 0 or X_df.shape[1] == 0:
        raise ValueError(
            "X and background must each have at least one row and one column."
        )
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")
    if n_background is not None and n_background < 1:
        raise ValueError("n_background must be at least 1.")

    rng = np.random.default_rng(random_state)
    if n_background is not None and n_background < len(background_df):
        background_df = background_df.iloc[
            rng.choice(len(background_df), n_background, replace=False)
        ]
    match getattr(predict_fn, "__self__", None):
        case (
            TabPFNClassifier(fit_mode="fit_with_cache")
            | TabPFNRegressor(fit_mode="fit_with_cache")
        ):
            warnings.warn(
                "The model uses fit_mode='fit_with_cache'. The API predicts at "
                f"most {CACHED_FIT_BATCH_SIZE} rows per request against a cached "
                "fit, so permutation_shap needs more requests and runs slower. "
                "Fit the model with the default fit_mode to explain it.",
                stacklevel=2,
            )
    n_background = len(background_df)
    if batch_size is None:
        batch_size = _default_batch_size(predict_fn, X_df, background_df)
        if batch_size < n_background:
            raise ValueError(
                f"The API accepts {batch_size} rows per request, fewer than the "
                f"{n_background} background rows. Set a smaller n_background."
            )
    elif batch_size < n_background:
        raise ValueError("batch_size must be at least n_background.")

    n_rows, n_features = X_df.shape
    n_orderings = 2 * n_permutations
    # Masks of a row: all hidden, all shown, then the n_features - 1 steps of
    # each ordering. See `_masks` for the numbering.
    n_masks = 2 + n_orderings * (n_features - 1)
    # The bool masks of one chunk of rows take at most 1/8 of the memory of one
    # batch of masked float rows.
    rows_per_chunk = max(1, batch_size // n_masks)
    chunks = [
        (start, min(start + rows_per_chunk, n_rows))
        for start in range(0, n_rows, rows_per_chunk)
    ]

    rank = np.empty(
        (n_rows, n_orderings, n_features), dtype=np.min_scalar_type(n_features - 1)
    )
    # Predict each (explained row, mask) pair once. All hidden gives the
    # background rows for every explained row, so it is the single pair 0. A
    # pair is stored as row * n_masks + mask, and its mask is rebuilt from
    # `rank` when its batch is predicted.
    pairs = [np.zeros(1, dtype=np.int64)]
    pair_of_mask = np.zeros(
        (n_rows, n_masks), dtype=np.min_scalar_type(n_rows * n_masks)
    )
    n_pairs = 1
    for start, stop in chunks:
        # rank[i, p, j]: position of feature j in ordering p of row i. Removing
        # features in one order passes the same coalitions as adding them in
        # the reverse order, so the reversed orderings are the antithetic half.
        chunk_rank = rng.random((stop - start, n_permutations, n_features)).argsort(
            axis=-1
        )
        rank[start:stop] = np.concatenate(
            [chunk_rank, n_features - 1 - chunk_rank], axis=1
        )
        flat = (
            np.arange(start, stop)[:, None] * n_masks + np.arange(1, n_masks)
        ).ravel()
        keys = np.concatenate(
            [
                (flat // n_masks)[:, None].astype(">i8").view(np.uint8),
                np.packbits(_masks(rank, flat, n_masks), axis=1),
            ],
            axis=1,
        )
        _, first, inverse = np.unique(
            keys, axis=0, return_index=True, return_inverse=True
        )
        pairs.append(flat[first])
        pair_of_mask[start:stop, 1:] = n_pairs + inverse.reshape(stop - start, -1)
        n_pairs += len(first)
    pair_ids = np.concatenate(pairs)

    pairs_per_call = batch_size // n_background
    means = []
    single_output = False
    for start in range(0, n_pairs, pairs_per_call):
        batch = pair_ids[start : start + pairs_per_call]
        rows = _masked_rows(
            X_df, background_df, _masks(rank, batch, n_masks), batch // n_masks
        )
        predictions = np.asarray(predict_fn(rows))
        if len(predictions) != len(rows):
            raise ValueError(
                f"predict_fn returned {len(predictions)} predictions for "
                f"{len(rows)} rows."
            )
        single_output = predictions.ndim == 1
        means.append(predictions.reshape(len(batch), n_background, -1).mean(axis=1))
    mean = np.concatenate(means)

    values = np.concatenate(
        [
            _values(mean[pair_of_mask[start:stop]], rank[start:stop])
            for start, stop in chunks
        ]
    )
    base_values = np.tile(mean[0], (n_rows, 1))
    if single_output:
        values, base_values = values[..., 0], base_values[..., 0]
    return ShapExplanation(values=values, base_values=base_values, data=X_df)


def _masks(rank: np.ndarray, pairs: np.ndarray, n_masks: int) -> np.ndarray:
    """Shown features of each pair `row * n_masks + mask`, shape (len(pairs), n_features).

    Mask 0 hides every feature and mask 1 shows every feature. Mask
    `2 + p * (n_features - 1) + s` shows the first `s + 1` features of ordering p.
    """
    rows, mask = np.divmod(pairs, n_masks)
    ordering, step = np.divmod(np.maximum(mask - 2, 0), max(rank.shape[-1] - 1, 1))
    shown = rank[rows, ordering] <= step[:, None]
    shown[mask == 0] = False
    shown[mask == 1] = True
    return shown


def _values(mean: np.ndarray, rank: np.ndarray) -> np.ndarray:
    """SHAP values from `mean`, the mean prediction of each (row, mask, output)."""
    n_rows, n_orderings, n_features = rank.shape
    ends = (n_rows, n_orderings, 1, mean.shape[-1])
    path = np.concatenate(
        [
            np.broadcast_to(mean[:, None, None, 0], ends),
            mean[:, 2:].reshape(n_rows, n_orderings, n_features - 1, mean.shape[-1]),
            np.broadcast_to(mean[:, None, None, 1], ends),
        ],
        axis=2,
    )
    # Entry k of the diff is the gain from the k-th feature of the ordering.
    return np.take_along_axis(np.diff(path, axis=2), rank[..., None], axis=2).mean(
        axis=1
    )


def _default_batch_size(
    predict_fn: Callable[[pd.DataFrame], Any],
    X: pd.DataFrame,
    background: pd.DataFrame,
) -> int:
    """The default, or fewer if one request of a client estimator accepts fewer."""
    match getattr(predict_fn, "__self__", None):
        case TabPFNClassifier() | TabPFNRegressor() as model:
            default = (
                CACHED_FIT_BATCH_SIZE
                if model.fit_mode == "fit_with_cache"
                else DEFAULT_BATCH_SIZE
            )
            api_settings = ServiceClient.get_settings()
            tabpfn_config = model._get_tabpfn_config()
            limit = _limit_for_model_path(tabpfn_config.model_path)
            if api_settings is None or limit is None:
                return default
            n_features = X.shape[1]
            # A masked row takes each value from X or background, so take the
            # larger average per column. The size check also counts the index.
            bytes_per_row = np.maximum(
                X.memory_usage(deep=True, index=False) / len(X),
                background.memory_usage(deep=True, index=False) / len(background),
            ).sum()
            max_bytes = (
                api_settings.dataset_max_size_bytes - pd.RangeIndex(0).memory_usage()
            )
            return min(
                default,
                _test_row_limit(
                    limit, model._n_train_rows, tabpfn_config.inference_config
                ),
                limit.test_set_max_cells // n_features,
                int(max_bytes // bytes_per_row),
            )
        case _:
            return DEFAULT_BATCH_SIZE


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
