"""Checks of the calibration tools."""

import numpy as np

from src.calibration.calibrators import fit_isotonic, fit_platt
from src.calibration.reliability import (
    expected_calibration_error,
    fit_logistic,
    reliability_table,
    sigmoid,
    wilson_interval,
)
from src.settings.loader import Settings

SAMPLE = 200000
SHIFT = 0.8
TOLERANCE = 0.05
ECE_LIMIT = 0.002


def make_overconfident(count: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(0)
    true_log_odds = generator.normal(-4.0, 1.0, count)
    labels = generator.random(count) < sigmoid(true_log_odds)
    return sigmoid(SHIFT * true_log_odds - 0.5), labels


def test_logistic_fit_recovers_a_known_curve(settings: Settings) -> None:
    options = settings.modeling.calibration
    generator = np.random.default_rng(1)
    predictions = sigmoid(generator.normal(-3.0, 1.0, SAMPLE))
    labels = generator.random(SAMPLE) < predictions
    intercept, slope = fit_logistic(
        predictions,
        labels,
        options.newton_iterations,
        options.newton_tolerance,
        options.probability_floor,
    )
    assert abs(intercept) < TOLERANCE
    assert abs(slope - 1.0) < TOLERANCE


def test_calibrators_reduce_the_calibration_error(settings: Settings) -> None:
    options = settings.modeling.calibration
    predictions, labels = make_overconfident(SAMPLE)
    before = expected_calibration_error(
        reliability_table(predictions, labels, options.bins, options.confidence_z)
    )
    platt = fit_platt(predictions, labels, options).apply(predictions)
    isotonic = fit_isotonic(predictions, labels).apply(predictions)
    for fixed in (platt, isotonic):
        table = reliability_table(fixed, labels, options.bins, options.confidence_z)
        assert expected_calibration_error(table) < before
        assert expected_calibration_error(table) < ECE_LIMIT


def test_wilson_interval_contains_the_share(settings: Settings) -> None:
    positives = np.array([5.0, 50.0, 500.0])
    counts = np.array([100.0, 1000.0, 10000.0])
    z = settings.modeling.calibration.confidence_z
    low, high = wilson_interval(positives, counts, z)
    share = positives / counts
    assert (low < share).all()
    assert (share < high).all()
