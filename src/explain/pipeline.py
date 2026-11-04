"""Explains the booking model with Shapley values and saves the tables."""

import pandas as pd

from src.explain.plots import save_dependence_figure
from src.explain.shap_values import compute_shap
from src.explain.summary import group_table, importance_table, price_effect_table
from src.settings.loader import Settings
from src.storage import results_directory

PRICE_EFFECT_FILE = "shap_price_effect.csv"
PRICE_BINS = 20


def run_explanation(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes Shapley values and writes the importance tables and the figure."""
    options = settings.modeling.explain
    result = compute_shap(settings)
    importance = importance_table(result, settings)
    groups = group_table(importance)
    output = results_directory(settings)
    output.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output / options.importance_file, index=False)
    groups.to_csv(output / options.group_file, index=False)
    price = price_effect_table(result, PRICE_BINS)
    price.to_csv(output / PRICE_EFFECT_FILE, index=False)
    folder = output / settings.modeling.calibration.figure_folder
    top = importance["feature"].head(options.top_features).tolist()
    save_dependence_figure(
        result, top, options.dependence_points, folder / options.figure_name
    )
    return importance, groups
