"""Plot the predicted target distribution of a TabPFN regressor for one sample."""

#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from scipy.ndimage import uniform_filter1d

_STAT_STYLES = {
    "mean": ("#d62728", "-"),
    "median": ("#2ca02c", "--"),
    "mode": ("#ff7f0e", ":"),
}


def _validate_args(
    prediction: Mapping[str, Any],
    sample_idx: int,
    statistics: Sequence[str],
    quantile_interval: tuple[float, float] | None,
    zoom_quantile: float | None,
    smooth: float,
) -> None:
    if not {"logits", "borders"} <= prediction.keys():
        raise ValueError(
            'prediction must be the output of predict(..., output_type="full").'
        )
    unknown = [name for name in statistics if name not in _STAT_STYLES]
    if unknown:
        raise ValueError(
            f"Unknown statistics {unknown}; choose from {list(_STAT_STYLES)}."
        )
    if quantile_interval is not None:
        lo_q, hi_q = quantile_interval
        if not 0 <= lo_q < hi_q <= 1:
            raise ValueError(
                "quantile_interval must be (low, high) with 0 <= low < high <= 1."
            )
    if zoom_quantile is not None and not 0 < zoom_quantile <= 1:
        raise ValueError("zoom_quantile must be in (0, 1].")
    if smooth < 0:
        raise ValueError("smooth must be non-negative.")
    n_samples = prediction["logits"].shape[0]
    if not 0 <= sample_idx < n_samples:
        raise ValueError(
            f"sample_idx {sample_idx} is out of range for {n_samples} sample(s)."
        )


def _softmax(values: np.ndarray) -> np.ndarray:
    """Compute stable softmax probabilities for NumPy logits."""
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _icdf(probabilities: np.ndarray, borders: np.ndarray, q: float) -> float:
    """Return the quantile of the piecewise-uniform bar distribution."""
    if q <= 0:
        return float(borders[0])
    if q >= 1:
        return float(borders[-1])

    cumulative = probabilities.cumsum()
    index = min(int(np.searchsorted(cumulative, q)), len(probabilities) - 1)
    probability_before = cumulative[index - 1] if index else 0.0
    fraction_in_bucket = (q - probability_before) / probabilities[index]
    return float(
        borders[index] + (borders[index + 1] - borders[index]) * fraction_in_bucket
    )


def plot_regression_distribution(
    prediction: Mapping[str, Any],
    *,
    sample_idx: int = 0,
    statistics: Sequence[str] = ("mean", "median", "mode"),
    quantile_interval: tuple[float, float] | None = (0.1, 0.9),
    zoom_quantile: float | None = 0.99,
    smooth: float = 0.005,
    ax: Axes | None = None,
    color: str = "#1f77b4",
) -> Axes:
    """Plot the predicted target distribution for a single sample.

    Parameters
    ----------
    prediction : mapping
        Output of ``regressor.predict(X, output_type="full")``. It may hold
        several samples; pick the one to plot with ``sample_idx``.
    sample_idx : int, default=0
        Index of the sample to plot within ``prediction``.
    statistics : sequence of str, default=("mean", "median", "mode")
        Point statistics to mark with a vertical line.
    quantile_interval : tuple of float or None, default=(0.1, 0.9)
        Central interval to shade. Pass ``None`` to disable.
    zoom_quantile : float or None, default=0.99
        Fraction of probability mass to keep in view, centred on the median.
        Pass ``None`` to show the full support.
    smooth : float, default=0.005
        Width of the display-only moving average over the density, as a
        fraction of the number of bars. Pass ``0`` to show the raw bar density.
    ax : matplotlib.axes.Axes or None, default=None
        Existing axes to draw on. A new figure is created if omitted.
    color : str, default="#1f77b4"
        Base colour of the density curve.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    _validate_args(
        prediction, sample_idx, statistics, quantile_interval, zoom_quantile, smooth
    )

    logits = np.asarray(prediction["logits"][sample_idx], dtype=float)
    borders = np.asarray(prediction["borders"], dtype=float)
    widths = np.diff(borders)
    centers = borders[:-1] + widths / 2
    probabilities = _softmax(logits)
    density = probabilities / widths
    if smooth:
        density = uniform_filter1d(density, max(1, round(smooth * len(density))))

    if ax is None:
        _, plot_ax = plt.subplots(figsize=(8, 4.5))
    else:
        plot_ax = ax

    plot_ax.fill_between(centers, density, color=color, alpha=0.18, lw=0)
    plot_ax.plot(centers, density, color=color, lw=1.8)

    legend_handles = []
    if quantile_interval is not None:
        lo, hi = (_icdf(probabilities, borders, q) for q in quantile_interval)
        band = (centers >= lo) & (centers <= hi)
        pct = round((quantile_interval[1] - quantile_interval[0]) * 100)
        plot_ax.fill_between(centers[band], density[band], color=color, alpha=0.3, lw=0)
        legend_handles.append(
            Patch(facecolor=color, alpha=0.5, lw=0, label=f"{pct}% interval")
        )

    for name in statistics:
        value = float(np.atleast_1d(prediction[name])[sample_idx])
        line_color, line_style = _STAT_STYLES[name]
        legend_handles.append(
            plot_ax.axvline(
                value,
                color=line_color,
                ls=line_style,
                lw=1.6,
                label=f"{name} = {value:.3g}",
            )
        )

    if zoom_quantile is not None:
        tail = (1 - zoom_quantile) / 2
        plot_ax.set_xlim(
            _icdf(probabilities, borders, tail),
            _icdf(probabilities, borders, 1 - tail),
        )

    visible = density[
        (centers >= plot_ax.get_xlim()[0]) & (centers <= plot_ax.get_xlim()[1])
    ]
    plot_ax.set_ylim(0, visible.max() * 1.1 if visible.size else None)
    plot_ax.margins(x=0)
    plot_ax.set_xlabel("Predicted target")
    plot_ax.set_ylabel("Probability density")
    plot_ax.set_title("TabPFN predicted distribution")
    plot_ax.spines["top"].set_visible(False)
    plot_ax.spines["right"].set_visible(False)
    plot_ax.legend(handles=legend_handles, fontsize=9)
    return plot_ax
