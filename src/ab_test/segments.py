"""Reports the experiment effect within each traveler segment."""

import math

import pandas as pd

from src.ab_test.analysis import compare_arms
from src.world.models import BoolArray, FloatArray, TextArray

ESTIMATE_COLUMNS = [
    "control_mean",
    "treatment_mean",
    "difference",
    "lower",
    "upper",
    "p_value",
]


def segment_row(
    metric: FloatArray,
    treated: BoolArray,
    inside: BoolArray,
    alpha: float,
) -> dict[str, float]:
    """Compares the arms among the sessions of one segment."""
    control = metric[inside & ~treated]
    comparison = compare_arms(control, metric[inside & treated], alpha)
    return {
        "sessions": float(inside.sum()),
        "control_mean": comparison.control_mean,
        "treatment_mean": comparison.treatment_mean,
        "difference": comparison.difference,
        "lower": comparison.lower,
        "upper": comparison.upper,
        "p_value": comparison.p_value,
    }


def unestimated_row(inside: BoolArray) -> dict[str, float]:
    """Reports a segment too small to estimate, keeping only its size."""
    row: dict[str, float] = {"sessions": float(inside.sum())}
    for column in ESTIMATE_COLUMNS:
        row[column] = math.nan
    return row


def is_estimable(
    treated: BoolArray,
    inside: BoolArray,
    minimum_sessions: int,
) -> bool:
    """Tells whether both arms of a segment reach the minimum size."""
    treated_count = int((inside & treated).sum())
    control_count = int((inside & ~treated).sum())
    return min(treated_count, control_count) >= minimum_sessions


def segment_effects(
    metric: FloatArray,
    treated: BoolArray,
    segments: TextArray,
    alpha: float,
    minimum_sessions: int,
) -> pd.DataFrame:
    """Compares the arms separately inside every segment."""
    names: list[str] = sorted(str(label) for label in set(segments.tolist()))
    rows: list[dict[str, float]] = []
    for name in names:
        inside = segments == name
        if is_estimable(treated, inside, minimum_sessions):
            rows.append(segment_row(metric, treated, inside, alpha))
        else:
            rows.append(unestimated_row(inside))
    table = pd.DataFrame(rows)
    table.insert(0, "segment", names)
    return table
