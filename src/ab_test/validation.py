"""Checks the experiment method on many replications against the known truth."""

from src.world.models import FloatArray


def rejection_rate(p_values: FloatArray, alpha: float) -> float:
    """Measures the share of replications that reject the null hypothesis."""
    raise NotImplementedError


def coverage_rate(lower: FloatArray, upper: FloatArray, truth: float) -> float:
    """Measures the share of intervals that contain the true effect."""
    raise NotImplementedError
