"""Turns per-session revenue estimates of pricing policies into report tables."""

import numpy as np
import pandas as pd

from src.margin.policies import Policy
from src.margin.sample import MarginSample
from src.margin.truth_tables import TrueTables
from src.modeling.truth_metrics import difference_parts, ratio_interval
from src.settings.loader import Settings
from src.world.helpers import read_floats
from src.world.models import FloatArray

Estimates = dict[str, dict[str, FloatArray]]


def interval_columns(
    kind: str,
    values: FloatArray,
    settings: Settings,
) -> dict[str, float]:
    """Gives the mean of one estimate with its bootstrap interval, or blanks."""
    if np.isnan(values).any():
        return {kind: np.nan, f"{kind}_lower": np.nan, f"{kind}_upper": np.nan}
    ones = np.ones(len(values))
    bootstrap = settings.modeling.bootstrap
    mean, low, high = ratio_interval((values, ones), bootstrap)
    return {kind: mean, f"{kind}_lower": low, f"{kind}_upper": high}


def summarize(estimates: Estimates, settings: Settings) -> pd.DataFrame:
    """Reports the mean revenue per session of every estimate with intervals."""
    rows: list[dict[str, float | str]] = []
    for name, kinds in estimates.items():
        row: dict[str, float | str] = {"policy": name}
        for kind, values in kinds.items():
            row.update(interval_columns(kind, values, settings))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_summary(
    estimates: Estimates,
    baselines: list[str],
    settings: Settings,
) -> pd.DataFrame:
    """Compares every policy with the baselines on the same resampled sessions."""
    bootstrap = settings.modeling.bootstrap
    rows: list[dict[str, float | str]] = []
    for baseline in baselines:
        for name, kinds in estimates.items():
            ones = np.ones(len(kinds["true"]))
            parts = difference_parts(
                (kinds["true"], ones), (estimates[baseline]["true"], ones)
            )
            mean, low, high = ratio_interval(parts, bootstrap)
            rows.append(
                {
                    "baseline": baseline,
                    "policy": name,
                    "difference": mean,
                    "lower": low,
                    "upper": high,
                }
            )
    return pd.DataFrame(rows)


def price_ratio_at(
    sample: MarginSample,
    margin: float,
    pool: int,
) -> FloatArray:
    """Computes the price relative to competitors at a given margin."""
    net_cost = read_floats(sample.rows, "net_cost").reshape(-1, pool)
    competitor = read_floats(sample.rows, "competitor_retail_price")
    return net_cost * (1.0 + margin) / competitor.reshape(-1, pool)


def price_bins(
    ratio: FloatArray,
    predicted: FloatArray,
    true: FloatArray,
    margin: float,
    count: int,
) -> list[dict[str, float]]:
    """Compares predicted and true probabilities in bins of relative price."""
    edges = np.quantile(ratio, np.linspace(0.0, 1.0, count + 1))
    index = np.searchsorted(edges[1:-1], ratio, side="right")
    rows: list[dict[str, float]] = []
    for number in range(count):
        chosen = index == number
        rows.append(
            {
                "margin": margin,
                "bin": float(number),
                "mean_price_ratio": float(ratio[chosen].mean()),
                "mean_predicted": float(predicted[chosen].mean()),
                "mean_true": float(true[chosen].mean()),
            }
        )
    return rows


def response_tables(
    probability: FloatArray,
    tables: TrueTables,
    sample: MarginSample,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compares the predicted and the true price response at every margin."""
    count = settings.modeling.margin.bins
    totals: list[dict[str, float]] = []
    bins: list[dict[str, float]] = []
    for position, margin in enumerate(settings.run.margin_grid):
        predicted = probability[:, :, position]
        true = tables.single_probability[:, :, position]
        ratio = price_ratio_at(sample, margin, settings.pool.size)
        totals.append(
            {
                "margin": margin,
                "mean_predicted": float(predicted.mean()),
                "mean_true": float(true.mean()),
                "mean_price_ratio": float(ratio.mean()),
            }
        )
        bins += price_bins(ratio, predicted, true, margin, count)
    return pd.DataFrame(totals), pd.DataFrame(bins)


def choices_table(policy: Policy, settings: Settings) -> pd.DataFrame:
    """Counts how often each margin of the grid is chosen by a policy."""
    grid = settings.run.margin_grid
    counts = np.bincount(policy.margin_index.reshape(-1), minlength=len(grid))
    return pd.DataFrame({"margin": grid, "share": counts / counts.sum()})
