"""Compares pricing policies by their estimated and true expected revenue."""

from dataclasses import dataclass
from pathlib import Path

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
from src.margin.reports import (
    Estimates,
    choices_table,
    paired_summary,
    response_tables,
    summarize,
)
from src.margin.repricing import repricing_policies
from src.margin.sample import MarginSample, load_margin_sample
from src.margin.truth_tables import TrueTables, build_true_tables
from src.settings.loader import Settings
from src.storage import results_directory
from src.world.helpers import read_floats
from src.world.models import FloatArray

MARGIN_FOLDER = "margin"
CHOICE_FILE = "margin_choices.csv"


@dataclass
class MarginRun:
    sample: MarginSample
    probability: FloatArray
    tables: TrueTables
    data: LoggedData


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
    """Builds the slate policies that charge one margin to every package."""
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


def slate_policies(run: MarginRun, settings: Settings) -> dict[str, Policy]:
    """Builds the policies that show a short slate of packages."""
    options = settings.modeling.margin
    grid = settings.run.margin_grid
    size = options.slate_size
    generator = np.random.default_rng(options.seed)
    data = run.data
    shown = logged_probability(run.probability, data.logged_index)
    profit = data.net_cost * np.array(grid)[data.logged_index]
    conversion = logged_margin_policy(shown, data.logged_index, size)
    single = run.tables.single_probability
    return {
        RANDOM: random_policy(data.logged_index, size, generator),
        CONVERSION: conversion,
        PROFIT: logged_margin_policy(shown * profit, data.logged_index, size),
        MARGIN_ONLY: optimized_margins_policy(
            run.probability, data.net_cost, grid, conversion.slate
        ),
        OPTIMIZED: optimized_policy(run.probability, data.net_cost, grid, size),
        LAB_ORACLE: optimized_policy(single, data.net_cost, grid, size),
    }


def build_policies(
    run: MarginRun,
    settings: Settings,
) -> tuple[dict[str, Policy], list[str]]:
    """Builds every policy and names those the logs can grade."""
    policies = slate_policies(run, settings)
    if settings.modeling.margin.fixed_margin_policies:
        policies.update(fixed_policies(run.probability, run.data, settings))
    grid = settings.run.margin_grid
    repricing = repricing_policies(run.probability, run.data.net_cost, grid)
    policies.update(repricing)
    return policies, list(repricing)


def estimate_policy(
    policy: Policy,
    graded_by_logs: bool,
    run: MarginRun,
    settings: Settings,
) -> dict[str, FloatArray]:
    """Computes the true, the model and, where valid, the logged revenue."""
    true = true_estimate(run.tables, run.data, policy, settings)
    logged = np.full(len(true), np.nan)
    if graded_by_logs:
        logged = logged_estimate(run.data, policy, settings)
    model = model_estimate(run.probability, run.data, policy, settings)
    return {"true": true, "model": model, "logged": logged}


def estimate_all(
    policies: dict[str, Policy],
    logged_names: list[str],
    run: MarginRun,
    settings: Settings,
) -> Estimates:
    """Computes the revenue estimates per session for every policy."""
    estimates: Estimates = {}
    for name, policy in policies.items():
        graded_by_logs = name in logged_names
        estimates[name] = estimate_policy(policy, graded_by_logs, run, settings)
    return estimates


def prepare_run(settings: Settings) -> MarginRun:
    """Loads the sample, predicts every margin and builds the true tables."""
    sample = load_margin_sample(settings)
    probability = predict_grid(sample, settings)
    tables = build_true_tables(sample, settings)
    data = logged_data(sample, tables, settings.pool.size)
    return MarginRun(sample, probability, tables, data)


def margin_output_folder(settings: Settings) -> Path:
    """Returns the folder for the margin results of the evaluated model."""
    experiment = settings.modeling.margin.experiment
    folder = results_directory(settings) / MARGIN_FOLDER / experiment
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_tables(tables: dict[str, pd.DataFrame], folder: Path) -> None:
    """Writes every report table under its file name."""
    for name, table in tables.items():
        table.to_csv(folder / name, index=False)


def run_margin(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds policies, estimates their revenue and saves the comparison."""
    options = settings.modeling.margin
    run = prepare_run(settings)
    policies, logged_names = build_policies(run, settings)
    estimates = estimate_all(policies, logged_names, run, settings)
    table = summarize(estimates, settings)
    paired = paired_summary(estimates, [RANDOM, PROFIT], settings)
    totals, bins = response_tables(run.probability, run.tables, run.sample, settings)
    reports: dict[str, pd.DataFrame] = {
        options.policy_file: table,
        options.paired_file: paired,
        options.response_file: totals,
        options.response_price_file: bins,
        CHOICE_FILE: choices_table(policies[OPTIMIZED], settings),
    }
    save_tables(reports, margin_output_folder(settings))
    return table, paired
