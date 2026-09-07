"""Plot the predicted target distribution of a TabPFN regressor for one sample."""

#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_STAT_STYLES = {
    "mean": ("#d62728", "-"),
    "median": ("#2ca02c", "--"),
    "mode": ("#ff7f0e", ":"),
}


def _validated_arrays(
    prediction: Mapping[str, Any],
    sample_idx: int,
    statistics: Sequence[str],
    quantile_interval: tuple[float, float] | None,
    zoom_quantile: float | None,
    smooth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Check the arguments and return the logits and borders as float arrays."""
    if not {"logits", "borders"} <= prediction.keys():
        raise ValueError(
            'prediction must be the output of predict(..., output_type="full").'
        )
    unknown = [name for name in statistics if name not in _STAT_STYLES]
    if unknown:
        raise ValueError(
            f"Unknown statistics {unknown}; choose from {list(_STAT_STYLES)}."
        )
    missing = [name for name in statistics if name not in prediction]
    if missing:
        raise ValueError(
            f"prediction does not contain the requested statistics {missing}."
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

    logits = np.atleast_2d(np.asarray(prediction["logits"], dtype=float))
    if logits.ndim != 2:
        raise ValueError(
            "prediction['logits'] must be 2-D (n_samples, n_bars); got shape "
            f"{np.shape(prediction['logits'])}."
        )
    borders = np.asarray(prediction["borders"], dtype=float)
    if borders.ndim != 1 or borders.size != logits.shape[1] + 1:
        raise ValueError(
            f"prediction['borders'] must be 1-D with {logits.shape[1] + 1} entries "
            f"for {logits.shape[1]} bars; got shape {borders.shape}."
        )
    n_samples = logits.shape[0]
    if not 0 <= sample_idx < n_samples:
        raise ValueError(
            f"sample_idx {sample_idx} is out of range for {n_samples} sample(s)."
        )
    return logits, borders


def _softmax(values: np.ndarray) -> np.ndarray:
    """Compute stable softmax probabilities for NumPy logits."""
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _boxcar(values: np.ndarray, window: int) -> np.ndarray:
    """Return the running sum of ``values`` over a centred window of ``window``."""
    return np.convolve(values, np.ones(window), mode="same")


def _icdf(probabilities: np.ndarray, borders: np.ndarray, q: float) -> float:
    """Return the quantile of the piecewise-uniform bar distribution."""
    if q <= 0:
        return float(borders[0])
    if q >= 1:
        return float(borders[-1])

    cumulative = probabilities.cumsum()
    last = int(np.flatnonzero(probabilities > 0)[-1])
    index = min(int(np.searchsorted(cumulative, q)), last)
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
        Point statistics to mark with a vertical line. Each one must be present
        in ``prediction``.
    quantile_interval : tuple of float or None, default=(0.1, 0.9)
        Central interval to shade. Pass ``None`` to disable.
    zoom_quantile : float or None, default=0.99
        Fraction of probability mass to keep in view, centred on the median.
        Pass ``None`` to show the full support.
    smooth : float, default=0.005
        Width of the display-only moving average over the density, as a
        fraction of the number of bars. Pass ``0`` to show the raw bar density.
    ax : matplotlib.axes.Axes or None, default=None
        Existing axes to draw on. A new figure is created if omitted. When the
        axes already holds a curve, the limits, labels and legend of that curve
        are preserved so several distributions can be overlaid.
    color : str, default="#1f77b4"
        Base colour of the density curve.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """
    logits, borders = _validated_arrays(
        prediction, sample_idx, statistics, quantile_interval, zoom_quantile, smooth
    )

    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.patches import Patch  # noqa: PLC0415
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. "
            'Install it with `pip install "tabpfn-client[viz]"`.'
        ) from err

    widths = np.diff(borders)
    centers = borders[:-1] + widths / 2
    probabilities = _softmax(logits[sample_idx])
    density = probabilities / widths
    window = max(1, round(smooth * len(density))) if smooth else 1
    if window > 1:
        # Smooth mass and width with the same kernel so the ratio stays a
        # density on non-uniform bars and the edge taper cancels out.
        density = _boxcar(probabilities, window) / _boxcar(widths, window)

    if ax is None:
        _, plot_ax = plt.subplots(figsize=(8, 4.5))
        overlay = False
    else:
        plot_ax = ax
        overlay = bool(ax.lines or ax.collections)
    previous_xlim = plot_ax.get_xlim() if overlay else None
    previous_ylim = plot_ax.get_ylim() if overlay else None
    previous_legend = plot_ax.get_legend() if overlay else None
    previous_handles = previous_legend.legend_handles if previous_legend else []

    plot_ax.fill_between(centers, density, color=color, alpha=0.18, lw=0)
    plot_ax.plot(centers, density, color=color, lw=1.8)

    legend_handles = []
    if quantile_interval is not None:
        lo, hi = (_icdf(probabilities, borders, q) for q in quantile_interval)
        inside = (centers > lo) & (centers < hi)
        band_x = np.concatenate(([lo], centers[inside], [hi]))
        pct = round((quantile_interval[1] - quantile_interval[0]) * 100)
        plot_ax.fill_between(
            band_x,
            np.interp(band_x, centers, density),
            color=color,
            alpha=0.3,
            lw=0,
        )
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
        low = _icdf(probabilities, borders, tail)
        high = _icdf(probabilities, borders, 1 - tail)
        if previous_xlim is not None:
            low = min(low, previous_xlim[0])
            high = max(high, previous_xlim[1])
        plot_ax.set_xlim(low, high)

    left, right = plot_ax.get_xlim()
    visible = density[(centers >= left) & (centers <= right)]
    top = (visible.max() if visible.size else density.max()) * 1.1
    if previous_ylim is not None:
        top = max(top, previous_ylim[1])
    plot_ax.set_ylim(0, top)
    plot_ax.margins(x=0)
    if not overlay:
        plot_ax.set_xlabel("Predicted target")
        plot_ax.set_ylabel("Probability density")
        plot_ax.set_title("TabPFN predicted distribution")
    plot_ax.spines["top"].set_visible(False)
    plot_ax.spines["right"].set_visible(False)
    plot_ax.legend(handles=[*previous_handles, *legend_handles], fontsize=9)
    return plot_ax
