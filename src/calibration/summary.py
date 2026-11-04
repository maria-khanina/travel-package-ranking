"""Summarizes calibration overall, by session and for groups of rows."""

import numpy as np
import pandas as pd

from src.calibration.data import CalibrationData
from src.calibration.reliability import (
    brier_score,
    expected_calibration_error,
    fit_logistic,
    reliability_table,
)
from src.modeling.evaluation import compute_log_loss
from src.settings.modeling_models import CalibrationSettings
from src.world.models import BoolArray, FloatArray


def calibration_measures(
    predictions: FloatArray,
    labels: BoolArray,
    settings: CalibrationSettings,
) -> dict[str, float]:
    """Computes the standard calibration measures of one set of predictions."""
    table = reliability_table(predictions, labels, settings.bins, settings.confidence_z)
    intercept, slope = fit_logistic(
        predictions,
        labels,
        settings.newton_iterations,
        settings.newton_tolerance,
        settings.probability_floor,
    )
    return {
        "rows": float(len(predictions)),
        "bookings": float(labels.sum()),
        "mean_prediction": float(predictions.mean()),
        "observed_rate": float(labels.mean()),
        "expected_calibration_error": expected_calibration_error(table),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "brier_score": brier_score(predictions, labels),
        "log_loss": compute_log_loss(labels, predictions),
    }


def session_values(
    predictions: FloatArray,
    labels: BoolArray,
    pool_size: int,
) -> tuple[FloatArray, BoolArray]:
    """Gives, per session, the sum of predictions and whether anything was booked."""
    sums = predictions.reshape(-1, pool_size).sum(axis=1)
    return sums, labels.reshape(-1, pool_size).any(axis=1)


def session_reliability(
    predictions: FloatArray,
    labels: BoolArray,
    pool_size: int,
    settings: CalibrationSettings,
) -> pd.DataFrame:
    """Compares the summed prediction of a session with its booking rate."""
    sums, booked = session_values(predictions, labels, pool_size)
    return reliability_table(sums, booked, settings.session_bins, settings.confidence_z)


def group_masks(
    data: CalibrationData,
    settings: CalibrationSettings,
) -> dict[str, BoolArray]:
    """Builds the row groups in which calibration is checked separately."""
    low, high = np.quantile(data.price_ratio, settings.price_quantiles)
    return {
        "all_rows": np.ones(len(data.labels), dtype=np.bool_),
        "known_hotels": ~data.cold,
        "unknown_hotels": data.cold,
        "with_children": data.children > 0,
        "without_children": data.children == 0,
        "cheaper_than_competitors": data.price_ratio <= low,
        "similar_to_competitors": (data.price_ratio > low) & (data.price_ratio <= high),
        "pricier_than_competitors": data.price_ratio > high,
    }


def group_summary(
    data: CalibrationData,
    predictions: FloatArray,
    settings: CalibrationSettings,
) -> pd.DataFrame:
    """Reports the calibration measures of every row group."""
    rows: list[dict[str, float | str]] = []
    for name, mask in group_masks(data, settings).items():
        measures = calibration_measures(predictions[mask], data.labels[mask], settings)
        rows.append({"group": name, **measures})
    return pd.DataFrame(rows)
