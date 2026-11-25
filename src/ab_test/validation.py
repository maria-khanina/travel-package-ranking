"""Checks the experiment method on many replications against the known truth."""

import numpy as np

from src.world.models import FloatArray


def rejection_rate(p_values: FloatArray, alpha: float) -> float:
    """Measures the share of replications that reject the null hypothesis."""
    return np.count_nonzero(p_values < alpha) / len(p_values)


def coverage_rate(lower: FloatArray, upper: FloatArray, truth: float) -> float:
    """Measures the share of intervals that contain the true effect."""
    covered = (lower <= truth) & (truth <= upper)
    return np.count_nonzero(covered) / len(covered)
