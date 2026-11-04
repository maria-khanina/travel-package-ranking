"""Measures how well predicted probabilities match the observed booking rates."""

import numpy as np
import pandas as pd

from src.world.models import BoolArray, FloatArray, IntArray

SINGLE_PARAMETER_SIZE = 2


def logit(probabilities: FloatArray, floor: float) -> FloatArray:
    """Turns probabilities into log odds, keeping them away from 0 and 1."""
    clipped = np.clip(probabilities, floor, 1.0 - floor)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: FloatArray) -> FloatArray:
    """Turns log odds back into probabilities."""
    return 1.0 / (1.0 + np.exp(-values))


def quantile_edges(predictions: FloatArray, bins: int) -> FloatArray:
    """Finds bin edges that put about the same number of rows in every bin."""
    edges = np.quantile(predictions, np.linspace(0.0, 1.0, bins + 1))
    return np.unique(edges)


def assign_bins(predictions: FloatArray, edges: FloatArray) -> IntArray:
    """Gives every prediction the number of its bin."""
    return np.searchsorted(edges[1:-1], predictions, side="right")


def wilson_interval(
    positives: FloatArray,
    counts: FloatArray,
    z: float,
) -> tuple[FloatArray, FloatArray]:
    """Computes the Wilson confidence interval of a share for each bin."""
    share = positives / counts
    scale = 1.0 + z**2 / counts
    center = (share + z**2 / (2.0 * counts)) / scale
    spread = z * np.sqrt(share * (1.0 - share) / counts + z**2 / (4.0 * counts**2))
    return center - spread / scale, center + spread / scale


def reliability_table(
    predictions: FloatArray,
    labels: BoolArray,
    bins: int,
    z: float,
) -> pd.DataFrame:
    """Compares the mean prediction with the observed share in every bin."""
    edges = quantile_edges(predictions, bins)
    frame = pd.DataFrame(
        {
            "bin": assign_bins(predictions, edges),
            "prediction": predictions,
            "booked": labels.astype(np.float64),
        }
    )
    grouped = frame.groupby("bin").agg(
        count=("booked", "size"),
        positives=("booked", "sum"),
        mean_prediction=("prediction", "mean"),
    )
    counts = grouped["count"].to_numpy(dtype=np.float64)
    positives = grouped["positives"].to_numpy(dtype=np.float64)
    low, high = wilson_interval(positives, counts, z)
    table = grouped.reset_index()
    table["observed_rate"] = positives / counts
    table["lower"], table["upper"] = low, high
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Averages the gap between prediction and observed share, weighted by size."""
    gap = (table["mean_prediction"] - table["observed_rate"]).abs()
    return float((gap * table["count"]).sum() / table["count"].sum())


def fit_logistic(
    predictions: FloatArray,
    labels: BoolArray,
    iterations: int,
    tolerance: float,
    floor: float,
) -> tuple[float, float]:
    """Fits label ~ intercept + slope * log odds of the prediction by Newton steps."""
    design = np.stack([np.ones(len(predictions)), logit(predictions, floor)], axis=1)
    target = labels.astype(np.float64)
    weights = np.array([0.0, 1.0])
    for _ in range(iterations):
        fitted = sigmoid(design @ weights)
        gradient = design.T @ (target - fitted)
        curvature = (design * (fitted * (1.0 - fitted))[:, None]).T @ design
        step = np.linalg.solve(curvature, gradient)
        weights = weights + step
        if float(np.abs(step).max()) < tolerance:
            break
    return float(weights[0]), float(weights[1])


def brier_score(predictions: FloatArray, labels: BoolArray) -> float:
    """Computes the mean squared error of the predicted probabilities."""
    return float(np.mean((predictions - labels.astype(np.float64)) ** 2))
