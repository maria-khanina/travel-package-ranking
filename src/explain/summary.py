"""Summarizes Shapley values by feature, by feature group and for price."""

import numpy as np
import pandas as pd

from src.explain.shap_values import ShapResult
from src.settings.loader import Settings
from src.world.helpers import read_floats
from src.world.models import FloatArray


def feature_group(name: str, settings: Settings) -> str:
    """Assigns a feature to a named group by exact name or by prefix."""
    options = settings.modeling.explain
    for group, names in options.exact_groups.items():
        if name in names:
            return group
    for prefix, group in options.prefix_groups.items():
        if name.startswith(prefix):
            return group
    return "other"


def split_session_parts(
    contributions: FloatArray,
    pool_size: int,
) -> tuple[FloatArray, FloatArray]:
    """Splits each contribution into its session mean and the deviation from it."""
    shaped = contributions.reshape(-1, pool_size, contributions.shape[1])
    between = shaped.mean(axis=1, keepdims=True)
    return np.broadcast_to(between, shaped.shape), shaped - between


def within_variance_share(contributions: FloatArray, pool_size: int) -> FloatArray:
    """Gives, per feature, the share of the variance that lies inside sessions."""
    shaped = contributions.reshape(-1, pool_size, contributions.shape[1])
    inside = shaped.var(axis=1).mean(axis=0)
    total = contributions.var(axis=0)
    return np.where(total > 0, inside / np.where(total > 0, total, 1.0), 0.0)


def importance_table(result: ShapResult, settings: Settings) -> pd.DataFrame:
    """Ranks the features by the mean absolute Shapley value and splits it."""
    options = settings.modeling.explain
    pool = settings.pool.size
    between, within = split_session_parts(result.contributions, pool)
    total = np.abs(result.contributions).mean(axis=0)
    table = pd.DataFrame({"feature": result.columns, "mean_abs_shap": total})
    table["between_sessions"] = np.abs(between).mean(axis=(0, 1))
    table["within_sessions"] = np.abs(within).mean(axis=(0, 1))
    table["within_variance_share"] = within_variance_share(result.contributions, pool)
    table["share"] = table["mean_abs_shap"] / table["mean_abs_shap"].sum()
    table["ranking_share"] = table["within_sessions"] / table["within_sessions"].sum()
    table["group"] = [feature_group(name, settings) for name in table["feature"]]
    table["no_effect_in_generator"] = table["feature"].isin(options.no_effect_columns)
    return table.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def group_table(importance: pd.DataFrame) -> pd.DataFrame:
    """Adds up the importance of the features of every group."""
    grouped = importance.groupby("group").agg(
        features=("feature", "size"),
        mean_abs_shap=("mean_abs_shap", "sum"),
        share=("share", "sum"),
        ranking_share=("ranking_share", "sum"),
    )
    return grouped.sort_values("share", ascending=False).reset_index()


def price_effect_table(result: ShapResult, bins: int) -> pd.DataFrame:
    """Shows how the log odds of booking move with the relative price."""
    ratio = read_floats(result.frame, "price_ratio_vs_competitor")
    index = result.columns.index("price_ratio_vs_competitor")
    frame = pd.DataFrame(
        {
            "price_ratio": ratio,
            "own_contribution": result.contributions[:, index],
            "total_contribution": result.contributions.sum(axis=1),
        }
    )
    frame["bin"] = pd.qcut(frame["price_ratio"], bins, duplicates="drop")
    grouped = frame.groupby("bin", observed=True).mean(numeric_only=True)
    return grouped.reset_index(drop=True)
