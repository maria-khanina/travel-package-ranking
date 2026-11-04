"""Calibrators that map predicted probabilities to better calibrated ones."""

import importlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.calibration.reliability import fit_logistic, logit, sigmoid
from src.settings.modeling_models import CalibrationSettings
from src.world.models import BoolArray, FloatArray

isotonic_module: Any = importlib.import_module("sklearn.isotonic")


@dataclass
class PlattCalibrator:
    intercept: float
    slope: float
    floor: float

    def apply(self, predictions: FloatArray) -> FloatArray:
        """Maps predictions through the fitted logistic curve."""
        return sigmoid(self.intercept + self.slope * logit(predictions, self.floor))


@dataclass
class IsotonicCalibrator:
    thresholds: FloatArray
    values: FloatArray

    def apply(self, predictions: FloatArray) -> FloatArray:
        """Maps predictions through the fitted non-decreasing step curve."""
        return np.interp(predictions, self.thresholds, self.values)


def fit_platt(
    predictions: FloatArray,
    labels: BoolArray,
    settings: CalibrationSettings,
) -> PlattCalibrator:
    """Fits a logistic calibration curve."""
    intercept, slope = fit_logistic(
        predictions,
        labels,
        settings.newton_iterations,
        settings.newton_tolerance,
        settings.probability_floor,
    )
    return PlattCalibrator(intercept, slope, settings.probability_floor)


def fit_isotonic(predictions: FloatArray, labels: BoolArray) -> IsotonicCalibrator:
    """Fits a non-decreasing calibration curve by pooling adjacent violators."""
    model = isotonic_module.IsotonicRegression(
        out_of_bounds="clip", y_min=0.0, y_max=1.0
    )
    model.fit(predictions, labels.astype(np.float64))
    thresholds = np.asarray(model.X_thresholds_, dtype=np.float64)
    values = np.asarray(model.y_thresholds_, dtype=np.float64)
    return IsotonicCalibrator(thresholds, values)
