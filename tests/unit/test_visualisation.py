#  Copyright (c) Prior Labs GmbH 2026.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import builtins
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from tabpfn_client.visualisation import plot_regression_distribution


@pytest.fixture
def full_prediction() -> dict[str, object]:
    return {
        "logits": np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]]),
        "borders": np.array([0.0, 1.0, 2.0, 4.0]),
        "mean": np.array([1.2, 2.6]),
        "median": np.array([1.5, 3.0]),
        "mode": np.array([1.5, 3.0]),
    }


def test_plot_regression_distribution_renders_client_numpy_output(
    full_prediction: dict[str, object],
) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    pytest.importorskip("scipy")
    matplotlib.use("Agg")

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


def test_plot_regression_distribution_requires_full_output() -> None:
    with pytest.raises(ValueError, match='output_type="full"'):
        plot_regression_distribution({"mean": np.array([1.0])})


def test_plot_regression_distribution_explains_optional_dependencies(
    full_prediction: dict[str, object],
) -> None:
    original_import = builtins.__import__

    def import_without_viz(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.partition(".")[0] in {"matplotlib", "scipy"}:
            raise ModuleNotFoundError(f"No module named {name}")
        return original_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=import_without_viz),
        pytest.raises(ModuleNotFoundError, match=r"tabpfn-client\[viz\]"),
    ):
        plot_regression_distribution(full_prediction)
