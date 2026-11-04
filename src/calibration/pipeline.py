"""Studies, fits and saves the calibration of the booking probability model."""

import json

import numpy as np
import pandas as pd

from src.calibration.calibrators import fit_isotonic, fit_platt
from src.calibration.data import CalibrationData, load_calibration_data
from src.calibration.plots import save_reliability_figure
from src.calibration.reliability import expected_calibration_error, reliability_table
from src.calibration.summary import (
    calibration_measures,
    group_summary,
    session_reliability,
    session_values,
)
from src.settings.loader import Settings
from src.storage import (
    TRUE_FILE,
    dataset_directory,
    models_directory,
    results_directory,
)
from src.world.helpers import read_floats
from src.world.models import FloatArray

Variants = dict[str, FloatArray]


def cross_fitted_variants(data: CalibrationData, settings: Settings) -> Variants:
    """Calibrates each fold with curves fitted on the other folds only."""
    options = settings.modeling.calibration
    platt = np.empty_like(data.predictions)
    isotonic = np.empty_like(data.predictions)
    for fold in np.unique(data.fold):
        held = data.fold == fold
        train, labels = data.predictions[~held], data.labels[~held]
        target = data.predictions[held]
        platt[held] = fit_platt(train, labels, options).apply(target)
        isotonic[held] = fit_isotonic(train, labels).apply(target)
    return {"raw": data.predictions, "platt": platt, "isotonic": isotonic}


def comparison_row(
    name: str,
    predictions: FloatArray,
    data: CalibrationData,
    settings: Settings,
) -> dict[str, float | str]:
    """Measures the calibration of one variant on row and on session level."""
    options = settings.modeling.calibration
    pool = settings.modeling.split.pool_size
    measures = calibration_measures(predictions, data.labels, options)
    sessions = session_reliability(predictions, data.labels, pool, options)
    sums, booked = session_values(predictions, data.labels, pool)
    measures["session_calibration_error"] = expected_calibration_error(sessions)
    measures["mean_session_sum"] = float(sums.mean())
    measures["observed_conversion"] = float(booked.mean())
    return {"variant": name, **measures}


def comparison_table(
    data: CalibrationData,
    variants: Variants,
    settings: Settings,
) -> pd.DataFrame:
    """Compares the raw and the calibrated predictions."""
    rows: list[dict[str, float | str]] = []
    for name, predictions in variants.items():
        rows.append(comparison_row(name, predictions, data, settings))
    return pd.DataFrame(rows)


def reliability_tables(
    data: CalibrationData,
    variants: Variants,
    settings: Settings,
) -> dict[str, pd.DataFrame]:
    """Builds the row-level reliability table of every variant."""
    options = settings.modeling.calibration
    tables: dict[str, pd.DataFrame] = {}
    for name, values in variants.items():
        tables[name] = reliability_table(
            values, data.labels, options.bins, options.confidence_z
        )
    return tables


def session_tables(
    data: CalibrationData,
    variants: Variants,
    settings: Settings,
) -> dict[str, pd.DataFrame]:
    """Builds the session-level reliability table of every variant."""
    options = settings.modeling.calibration
    pool = settings.modeling.split.pool_size
    tables: dict[str, pd.DataFrame] = {}
    for name, values in variants.items():
        tables[name] = session_reliability(values, data.labels, pool, options)
    return tables


def save_final_calibrators(data: CalibrationData, settings: Settings) -> None:
    """Fits both calibrators on all development predictions and saves them."""
    options = settings.modeling.calibration
    folder = models_directory(settings) / options.model_folder
    folder.mkdir(parents=True, exist_ok=True)
    platt = fit_platt(data.predictions, data.labels, options)
    isotonic = fit_isotonic(data.predictions, data.labels)
    text = json.dumps({"intercept": platt.intercept, "slope": platt.slope})
    (folder / f"{options.experiment}_platt.json").write_text(text, encoding="utf-8")
    np.savez(
        folder / f"{options.experiment}_isotonic.npz",
        thresholds=isotonic.thresholds,
        values=isotonic.values,
    )


def grade_against_truth(
    data: CalibrationData,
    variants: Variants,
    settings: Settings,
) -> pd.DataFrame:
    """Grades each variant against the true probabilities; nothing is fitted on them."""
    truth = pd.read_parquet(dataset_directory(settings) / TRUE_FILE)
    true_probability = read_floats(truth, "hid_true_probability")[data.rows]
    rows: list[dict[str, float | str]] = []
    for name, predictions in variants.items():
        error = float(np.abs(predictions - true_probability).mean())
        rows.append(
            {
                "variant": name,
                "mean_absolute_error": error,
                "mean_prediction": float(predictions.mean()),
                "mean_true_probability": float(true_probability.mean()),
            }
        )
    return pd.DataFrame(rows)


def save_figures(
    data: CalibrationData,
    variants: Variants,
    settings: Settings,
) -> None:
    """Draws the row-level and the session-level reliability figures."""
    options = settings.modeling.calibration
    folder = results_directory(settings) / options.figure_folder
    rows = reliability_tables(data, variants, settings)
    sessions = session_tables(data, variants, settings)
    save_reliability_figure(rows, "Rows", folder / "reliability_rows.png")
    save_reliability_figure(sessions, "Sessions", folder / "reliability_sessions.png")


def group_tables(
    data: CalibrationData,
    variants: Variants,
    settings: Settings,
) -> pd.DataFrame:
    """Reports the calibration of row groups for the raw and the isotonic variant."""
    options = settings.modeling.calibration
    tables: list[pd.DataFrame] = []
    for name in ("raw", "isotonic"):
        summary = group_summary(data, variants[name], options)
        tables.append(summary.assign(variant=name))
    return pd.concat(tables)


def save_tables(
    tables: dict[str, pd.DataFrame],
    settings: Settings,
) -> None:
    """Writes the calibration tables to the results folder."""
    output = results_directory(settings)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)


def run_calibration(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the calibration study and saves its tables, figures and calibrators."""
    data = load_calibration_data(settings)
    variants = cross_fitted_variants(data, settings)
    comparison = comparison_table(data, variants, settings)
    groups = group_tables(data, variants, settings)
    grading = grade_against_truth(data, variants, settings)
    names = ["calibration_comparison", "calibration_groups"]
    save_tables(dict(zip(names, (comparison, groups))), settings)
    save_tables({"calibration_grading_vs_truth": grading}, settings)
    save_figures(data, variants, settings)
    save_final_calibrators(data, settings)
    return comparison, groups
