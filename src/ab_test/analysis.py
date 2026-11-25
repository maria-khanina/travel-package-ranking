"""Compares the metric of two experiment arms and reduces its variance."""

from dataclasses import dataclass
import math
from statistics import NormalDist

from src.world.models import FloatArray

MINIMUM_ARM_SESSIONS = 2


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


def welch_standard_error(control: FloatArray, treatment: FloatArray) -> float:
    """Computes the standard error of a difference of means with unequal variances."""
    control_part = float(control.var(ddof=1)) / len(control)
    treatment_part = float(treatment.var(ddof=1)) / len(treatment)
    return math.sqrt(control_part + treatment_part)


def two_sided_p_value(difference: float, standard_error: float) -> float:
    """Computes the normal two-sided p-value of a difference."""
    if standard_error == 0.0:
        return 1.0 if difference == 0.0 else 0.0
    tail = 1.0 - NormalDist().cdf(abs(difference) / standard_error)
    return 2.0 * tail


def check_arm_sizes(control: FloatArray, treatment: FloatArray) -> None:
    """Refuses arms too small for a variance estimate."""
    if min(len(control), len(treatment)) < MINIMUM_ARM_SESSIONS:
        raise ValueError(
            f"each arm needs at least {MINIMUM_ARM_SESSIONS} sessions, got "
            f"{len(control)} in control and {len(treatment)} in treatment"
        )


def relative_lift(difference: float, control_mean: float) -> float:
    """Expresses the difference as a share of the control mean, if it is not zero."""
    if control_mean == 0.0:
        return math.nan
    return difference / control_mean


def compare_arms(
    control: FloatArray,
    treatment: FloatArray,
    alpha: float,
) -> ArmComparison:
    """Tests the difference of the arm means with the Welch standard error."""
    check_arm_sizes(control, treatment)
    control_mean = float(control.mean())
    treatment_mean = float(treatment.mean())
    difference = treatment_mean - control_mean
    error = welch_standard_error(control, treatment)
    half_width = NormalDist().inv_cdf(1.0 - alpha / 2.0) * error
    return ArmComparison(
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        difference=difference,
        standard_error=error,
        lower=difference - half_width,
        upper=difference + half_width,
        p_value=two_sided_p_value(difference, error),
        relative_lift=relative_lift(difference, control_mean),
    )


def cuped_adjust(metric: FloatArray, covariate: FloatArray) -> CupedResult:
    """Removes from the metric the part explained by a pre-assignment covariate."""
    centered = covariate - covariate.mean()
    variance = float((centered * centered).mean())
    theta = 0.0
    if variance > 0.0:
        theta = float((centered * (metric - metric.mean())).mean()) / variance
    adjusted = metric - theta * centered
    reduction = 1.0 - float(adjusted.var()) / float(metric.var())
    return CupedResult(adjusted, theta, reduction)
