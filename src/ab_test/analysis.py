"""Compares the metric of two experiment arms and reduces its variance."""

from dataclasses import dataclass

from src.world.models import FloatArray


@dataclass
class ArmComparison:
    control_mean: float
    treatment_mean: float
    difference: float
    standard_error: float
    lower: float
    upper: float
    p_value: float
    relative_lift: float


@dataclass
class CupedResult:
    adjusted: FloatArray
    theta: float
    variance_reduction: float


def compare_arms(
    control: FloatArray,
    treatment: FloatArray,
    alpha: float,
) -> ArmComparison:
    """Tests the difference of the arm means with the Welch standard error."""
    raise NotImplementedError


def cuped_adjust(metric: FloatArray, covariate: FloatArray) -> CupedResult:
    """Removes from the metric the part explained by a pre-assignment covariate."""
    raise NotImplementedError
