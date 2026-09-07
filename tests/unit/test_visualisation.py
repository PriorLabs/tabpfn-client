#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import builtins
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from tabpfn_client.visualisation import plot_regression_distribution
from tabpfn_client.visualisation.regression_distribution import _icdf


@pytest.fixture
def full_prediction() -> dict[str, object]:
    return {
        "logits": np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]]),
        "borders": np.array([0.0, 1.0, 2.0, 4.0]),
        "mean": np.array([1.2, 2.6]),
        "median": np.array([1.5, 3.0]),
        "mode": np.array([1.5, 3.0]),
    }


@pytest.fixture
def agg_pyplot() -> Any:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def test_plot_regression_distribution_renders_client_numpy_output(
    full_prediction: dict[str, object], agg_pyplot: Any
) -> None:
    ax = plot_regression_distribution(full_prediction, sample_idx=1, smooth=0)

    assert ax.get_xlabel() == "Predicted target"
    assert ax.get_ylabel() == "Probability density"
    assert ax.get_title() == "TabPFN predicted distribution"
    assert len(ax.lines) == 4
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "80% interval",
        "mean = 2.6",
        "median = 3",
        "mode = 3",
    ]


def test_plot_regression_distribution_smooths_over_non_uniform_bars(
    full_prediction: dict[str, object], agg_pyplot: Any
) -> None:
    ax = plot_regression_distribution(full_prediction, smooth=1.0)

    smoothed = np.asarray(ax.lines[0].get_ydata())
    assert np.all(np.isfinite(smoothed))
    assert np.all(smoothed > 0)


def test_plot_regression_distribution_accepts_single_sample_1d_logits(
    agg_pyplot: Any,
) -> None:
    prediction = {
        "logits": np.array([0.0, 1.0, -1.0]),
        "borders": np.array([0.0, 1.0, 2.0, 4.0]),
        "mean": 1.2,
    }

    ax = plot_regression_distribution(prediction, statistics=("mean",))

    assert ax.get_xlim()[0] < ax.get_xlim()[1]


def test_plot_regression_distribution_accepts_list_input(agg_pyplot: Any) -> None:
    prediction = {
        "logits": [[0.0, 1.0, -1.0]],
        "borders": [0.0, 1.0, 2.0, 4.0],
        "mean": [1.2],
    }

    ax = plot_regression_distribution(prediction, statistics=("mean",))

    assert len(ax.lines) == 2


def test_plot_regression_distribution_preserves_a_reused_axes(
    full_prediction: dict[str, object], agg_pyplot: Any
) -> None:
    first = plot_regression_distribution(full_prediction, sample_idx=0)
    first_xlim, first_ylim = first.get_xlim(), first.get_ylim()

    second = plot_regression_distribution(full_prediction, sample_idx=1, ax=first)

    assert second is first
    assert second.get_title() == "TabPFN predicted distribution"
    assert second.get_xlim()[0] <= first_xlim[0]
    assert second.get_xlim()[1] >= first_xlim[1]
    assert second.get_ylim()[1] >= first_ylim[1]
    legend = second.get_legend()
    assert legend is not None
    legend_labels = [text.get_text() for text in legend.get_texts()]
    assert legend_labels.count("80% interval") == 2
    assert "mean = 1.2" in legend_labels
    assert "mean = 2.6" in legend_labels


def test_plot_regression_distribution_handles_masked_tail_bars(
    agg_pyplot: Any,
) -> None:
    n_bars = 200
    logits = np.full(n_bars, -1.0)
    logits[n_bars // 2 :] = -np.inf
    prediction = {
        "logits": logits[None, :],
        "borders": np.linspace(0.0, 10.0, n_bars + 1),
        "mean": np.array([2.5]),
    }

    ax = plot_regression_distribution(prediction, statistics=("mean",))

    assert np.all(np.isfinite(ax.get_xlim()))
    assert np.all(np.isfinite(ax.get_ylim()))


def test_icdf_is_finite_beyond_the_accumulated_mass() -> None:
    probabilities = np.array([0.5, 0.5, 0.0, 0.0])
    borders = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    assert _icdf(probabilities, borders, 1 - 1e-16) == pytest.approx(2.0)


def test_plot_regression_distribution_shades_the_exact_quantile_edges(
    agg_pyplot: Any,
) -> None:
    prediction = {
        "logits": np.zeros((1, 4)),
        "borders": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "mean": np.array([2.0]),
    }

    ax = plot_regression_distribution(
        prediction,
        statistics=("mean",),
        quantile_interval=(0.25, 0.75),
        smooth=0,
    )

    band_x = np.asarray(ax.collections[-1].get_paths()[0].vertices)[:, 0]
    assert band_x.min() == pytest.approx(1.0)
    assert band_x.max() == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"statistics": ("variance",)}, "Unknown statistics"),
        ({"quantile_interval": (0.9, 0.1)}, "quantile_interval"),
        ({"zoom_quantile": 0.0}, "zoom_quantile"),
        ({"smooth": -0.1}, "smooth must be non-negative"),
        ({"sample_idx": 2}, "sample_idx 2 is out of range"),
    ],
)
def test_plot_regression_distribution_validates_arguments(
    full_prediction: dict[str, object], kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        plot_regression_distribution(full_prediction, **kwargs)


def test_plot_regression_distribution_reports_missing_statistics(
    full_prediction: dict[str, object],
) -> None:
    del full_prediction["mode"]

    with pytest.raises(ValueError, match=r"requested statistics \['mode'\]"):
        plot_regression_distribution(full_prediction)


def test_plot_regression_distribution_rejects_mismatched_borders() -> None:
    prediction = {
        "logits": np.zeros((1, 3)),
        "borders": np.array([0.0, 1.0, 2.0]),
        "mean": np.array([1.0]),
    }

    with pytest.raises(ValueError, match=r"borders'\] must be 1-D with 4 entries"):
        plot_regression_distribution(prediction, statistics=("mean",))


def test_plot_regression_distribution_rejects_3d_logits() -> None:
    prediction = {
        "logits": np.zeros((1, 2, 3)),
        "borders": np.array([0.0, 1.0, 2.0, 3.0]),
        "mean": np.array([1.0]),
    }

    with pytest.raises(ValueError, match="must be 2-D"):
        plot_regression_distribution(prediction, statistics=("mean",))


def test_plot_regression_distribution_requires_full_output() -> None:
    with pytest.raises(ValueError, match='output_type="full"'):
        plot_regression_distribution({"mean": np.array([1.0])})


def test_plot_regression_distribution_explains_optional_dependencies(
    full_prediction: dict[str, object],
) -> None:
    original_import = builtins.__import__

    def import_without_viz(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.partition(".")[0] == "matplotlib":
            raise ModuleNotFoundError(f"No module named {name}")
        return original_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=import_without_viz),
        pytest.raises(ModuleNotFoundError, match=r"tabpfn-client\[viz\]"),
    ):
        plot_regression_distribution(full_prediction)
