"""Draws the reliability figures of the calibration study."""

import importlib
from pathlib import Path
from typing import Any

import pandas as pd

matplotlib: Any = importlib.import_module("matplotlib")
matplotlib.use("Agg")
pyplot: Any = importlib.import_module("matplotlib.pyplot")

FIGURE_SIZE = (6.0, 6.0)
RESOLUTION = 150
MARKERS: list[str] = ["o", "s", "^"]


def save_reliability_figure(
    tables: dict[str, pd.DataFrame],
    title: str,
    path: Path,
) -> None:
    """Plots the observed booking rate against the mean prediction per bin."""
    figure, axes = pyplot.subplots(figsize=FIGURE_SIZE)
    low = min(float(table["mean_prediction"].min()) for table in tables.values())
    high = max(float(table["mean_prediction"].max()) for table in tables.values())
    axes.plot([low, high], [low, high], color="grey", linestyle="--", label="perfect")
    for marker, (name, table) in zip(MARKERS, tables.items()):
        axes.plot(
            table["mean_prediction"],
            table["observed_rate"],
            marker=marker,
            linestyle="-",
            label=name,
        )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("mean predicted probability")
    axes.set_ylabel("observed booking rate")
    axes.set_title(title)
    axes.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=RESOLUTION, bbox_inches="tight")
    pyplot.close(figure)
