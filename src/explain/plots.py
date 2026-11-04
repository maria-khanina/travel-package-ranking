"""Draws the Shapley dependence figure."""

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.explain.shap_values import ShapResult

matplotlib: Any = importlib.import_module("matplotlib")
matplotlib.use("Agg")
pyplot: Any = importlib.import_module("matplotlib.pyplot")

PANEL_COLUMNS = 4
PANEL_SIZE = (4.0, 3.2)
RESOLUTION = 130
POINT_SIZE = 3.0
TRANSPARENCY = 0.35
CATEGORY_LIMIT = 10
LABEL_SIZE = 6
TITLE_SIZE = 8
SAMPLE_SEED = 0


def draw_numeric(axes: Any, values: pd.Series, shap: np.ndarray) -> None:
    """Plots the contribution of a numeric feature against its value."""
    axes.scatter(values, shap, s=POINT_SIZE, alpha=TRANSPARENCY)


def draw_categorical(axes: Any, values: pd.Series, shap: np.ndarray) -> None:
    """Plots the mean contribution of the most frequent categories."""
    frame = pd.DataFrame({"value": values.astype(str).to_numpy(), "shap": shap})
    top = frame["value"].value_counts().head(CATEGORY_LIMIT).index
    means = frame[frame["value"].isin(top)].groupby("value")["shap"].mean()
    axes.bar(range(len(means)), means.to_numpy())
    axes.set_xticks(range(len(means)))
    axes.set_xticklabels(list(means.index), rotation=90, fontsize=LABEL_SIZE)


def save_dependence_figure(
    result: ShapResult,
    features: list[str],
    points: int,
    path: Path,
) -> None:
    """Draws, for each given feature, its Shapley value against its value."""
    rows = (len(features) + PANEL_COLUMNS - 1) // PANEL_COLUMNS
    size = (PANEL_SIZE[0] * PANEL_COLUMNS, PANEL_SIZE[1] * rows)
    figure, grid = pyplot.subplots(rows, PANEL_COLUMNS, figsize=size, squeeze=False)
    generator = np.random.default_rng(SAMPLE_SEED)
    sample = generator.choice(len(result.frame), points, replace=False)
    for axes, name in zip(grid.reshape(-1), features):
        shap = result.contributions[sample, result.columns.index(name)]
        values = result.frame[name].iloc[sample]
        if name in result.categorical:
            draw_categorical(axes, values, shap)
        else:
            draw_numeric(axes, values, shap)
        axes.set_title(name, fontsize=TITLE_SIZE)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=RESOLUTION)
    pyplot.close(figure)
