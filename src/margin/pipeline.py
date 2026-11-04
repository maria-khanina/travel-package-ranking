"""Compares pricing policies by their estimated and true expected revenue."""

import numpy as np
import pandas as pd

from src.margin.estimators import (
    LoggedData,
    logged_estimate,
    model_estimate,
    true_estimate,
)
from src.margin.policies import (
    CONVERSION,
    LAB_ORACLE,
    MARGIN_ONLY,
    OPTIMIZED,
    PROFIT,
    RANDOM,
    Policy,
    fixed_margin_policy,
    logged_margin_policy,
    logged_probability,
    optimized_margins_policy,
    optimized_policy,
    profit_grid,
    random_policy,
)
from src.margin.predict import predict_grid
from src.margin.sample import MarginSample, load_margin_sample
from src.margin.truth_tables import TrueTables, build_true_tables
from src.modeling.truth_metrics import difference_parts, ratio_interval
from src.settings.loader import Settings
from src.storage import results_directory
from src.world.helpers import read_floats
from src.world.models import FloatArray

Estimates = dict[str, dict[str, FloatArray]]
LOGGED_MARGIN_POLICIES: list[str] = [RANDOM, CONVERSION, PROFIT]
CHOICE_FILE = "margin_choices.csv"


def logged_data(sample: MarginSample, tables: TrueTables, pool: int) -> LoggedData:
    """Collects the logged quantities needed by the estimators."""
    shape = (-1, pool)
    return LoggedData(
        net_cost=read_floats(sample.rows, "net_cost").reshape(shape),
        booked=read_floats(sample.rows, "is_booked").reshape(shape),
        propensity=read_floats(sample.rows, "margin_propensity").reshape(shape),
        logged_index=tables.logged_index,
    )


def fixed_policies(
    probability: FloatArray,
    data: LoggedData,
    settings: Settings,
) -> dict[str, Policy]:
    """Builds the policies that charge one margin to every package."""
    grid = settings.run.margin_grid
    size = settings.modeling.margin.slate_size
    profit = profit_grid(data.net_cost, grid)
    policies: dict[str, Policy] = {}
    for position, margin in enumerate(grid):
        value = probability[:, :, position] * profit[:, :, position]
        policies[f"fixed_margin_{margin:.2f}"] = fixed_margin_policy(
            value, position, size
        )
    return policies


def build_policies(
    probability: FloatArray,
    tables: TrueTables,
    data: LoggedData,
    settings: Settings,
) -> dict[str, Policy]:
    """Builds every compared policy from model probabilities."""
    options = settings.modeling.margin
    grid = settings.run.margin_grid
    size = options.slate_size
    generator = np.random.default_rng(options.seed)
    shown = logged_probability(probability, data.logged_index)
    profit = data.net_cost * np.array(grid)[data.logged_index]
    conversion = logged_margin_policy(shown, data.logged_index, size)
    single = tables.single_probability
    policies = {
        RANDOM: random_policy(data.logged_index, size, generator),
        CONVERSION: conversion,
        PROFIT: logged_margin_policy(shown * profit, data.logged_index, size),
        MARGIN_ONLY: optimized_margins_policy(
            probability, data.net_cost, grid, conversion.slate
        ),
        OPTIMIZED: optimized_policy(probability, data.net_cost, grid, size),
        LAB_ORACLE: optimized_policy(single, data.net_cost, grid, size),
    }
    if options.fixed_margin_policies:
        policies.update(fixed_policies(probability, data, settings))
    return policies


def estimate_all(
    policies: dict[str, Policy],
    probability: FloatArray,
    tables: TrueTables,
    data: LoggedData,
    settings: Settings,
) -> Estimates:
    """Computes the three revenue estimates per session for every policy."""
    estimates: Estimates = {}
    for name, policy in policies.items():
        weighted = logged_estimate(data, policy, settings)
        if name in LOGGED_MARGIN_POLICIES:
            weighted = np.full(len(weighted), np.nan)
        estimates[name] = {
            "true": true_estimate(tables, data, policy, settings),
            "model": model_estimate(probability, data, policy, settings),
            "logged": weighted,
        }
    return estimates


def summarize(estimates: Estimates, settings: Settings) -> pd.DataFrame:
    """Reports the mean revenue per session of every estimate with intervals."""
    bootstrap = settings.modeling.bootstrap
    rows: list[dict[str, float | str]] = []
    for name, kinds in estimates.items():
        row: dict[str, float | str] = {"policy": name}
        for kind, values in kinds.items():
            if np.isnan(values).any():
                missing = {kind: np.nan, f"{kind}_lower": np.nan}
                row.update(missing | {f"{kind}_upper": np.nan})
                continue
            ones = np.ones(len(values))
            mean, low, high = ratio_interval((values, ones), bootstrap)
            row.update({kind: mean, f"{kind}_lower": low, f"{kind}_upper": high})
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


def run_margin(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds policies, estimates their revenue and saves the comparison."""
    options = settings.modeling.margin
    sample = load_margin_sample(settings)
    probability = predict_grid(sample, settings)
    tables = build_true_tables(sample, settings)
    data = logged_data(sample, tables, settings.pool.size)
    policies = build_policies(probability, tables, data, settings)
    estimates = estimate_all(policies, probability, tables, data, settings)
    table = summarize(estimates, settings)
    paired = paired_summary(estimates, [RANDOM, PROFIT], settings)
    totals, bins = response_tables(probability, tables, sample, settings)
    output = results_directory(settings)
    choices = choices_table(policies[OPTIMIZED], settings)
    choices.to_csv(output / CHOICE_FILE, index=False)
    table.to_csv(output / options.policy_file, index=False)
    paired.to_csv(output / options.paired_file, index=False)
    totals.to_csv(output / options.response_file, index=False)
    bins.to_csv(output / options.response_price_file, index=False)
    return table, paired
