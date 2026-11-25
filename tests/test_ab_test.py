"""Checks of the simulated A/B test: assignment, outcomes, statistics, segments."""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from tests.ab_test_check_settings import (
    ALPHA,
    ARM_SESSIONS,
    BINOMIAL_ERRORS,
    CLOSE_TOLERANCE,
    CONFIDENCE,
    CONTROL_VALUES,
    CUPED_NOISE,
    CUPED_OFFSET,
    CUPED_SESSIONS,
    CUPED_SLOPE,
    DOMINANCE_SESSIONS,
    DOMINANT_UTILITY,
    ESTIMABLE_SEGMENT,
    EXPECTED_COVERAGE,
    EXPECTED_REJECTION,
    EXPECTED_SESSIONS,
    FIRST_SEED,
    FREQUENCY_NOISE_SIGMA,
    FREQUENCY_OUTSIDE,
    FREQUENCY_SESSIONS,
    FREQUENCY_UTILITY,
    LOWER_BOUNDS,
    MINIMUM_REDUCTION,
    MINIMUM_SEGMENT_SESSIONS,
    NEGLIGIBLE_UTILITY,
    NORMAL_SHIFT,
    ODD_SESSION_COUNT,
    POWER,
    P_VALUES,
    RATE_TOLERANCE,
    REFERENCE_DRAWS,
    REPLICATIONS,
    SECOND_SEED,
    SEGMENT_LABELS,
    SEGMENT_METRIC,
    SEGMENT_TREATED,
    SHOWN_PROFIT,
    SINGLE_SESSION,
    SIZE_GROWTH,
    SMALL_SEGMENT,
    THETA_TOLERANCE,
    TREATMENT_VALUES,
    TRUE_EFFECT,
    UNIT_DEVIATION,
    UNIT_EFFECT,
    UPPER_BOUNDS,
    ZERO_CONTROL,
)

from src.ab_test.analysis import ArmComparison, compare_arms, cuped_adjust
from src.ab_test.assignment import assign_sessions
from src.ab_test.outcomes import SessionOutcomes, draw_outcomes
from src.ab_test.power import required_sessions
from src.ab_test.segments import segment_effects
from src.ab_test.validation import coverage_rate, rejection_rate
from src.world.models import FloatArray


def close(actual: float, expected: float) -> bool:
    """Tells whether two numbers agree up to rounding."""
    return math.isclose(
        actual,
        expected,
        rel_tol=CLOSE_TOLERANCE,
        abs_tol=CLOSE_TOLERANCE,
    )


def repeated_rows(values: list[float], sessions: int) -> FloatArray:
    """Repeats one row of package values for every session."""
    row = np.array(values, dtype=np.float64)
    return np.tile(row, (sessions, 1))


def draw_dominated(package_utility: float, outside_utility: float) -> SessionOutcomes:
    """Draws sessions where either the first package or leaving clearly wins."""
    utility: list[float] = [package_utility, NEGLIGIBLE_UTILITY, NEGLIGIBLE_UTILITY]
    return draw_outcomes(
        repeated_rows(utility, DOMINANCE_SESSIONS),
        np.full(DOMINANCE_SESSIONS, outside_utility),
        repeated_rows(SHOWN_PROFIT, DOMINANCE_SESSIONS),
        UNIT_DEVIATION,
        np.random.default_rng(FIRST_SEED),
    )


def reference_probabilities() -> FloatArray:
    """Averages the logit booking probabilities over the normal taste noise."""
    generator = np.random.default_rng(SECOND_SEED)
    utility = np.array(FREQUENCY_UTILITY, dtype=np.float64)
    shape = (REFERENCE_DRAWS, len(FREQUENCY_UTILITY))
    weight = np.exp(utility + FREQUENCY_NOISE_SIGMA * generator.normal(size=shape))
    total = math.exp(FREQUENCY_OUTSIDE) + weight.sum(axis=1, keepdims=True)
    return (weight / total).mean(axis=0)


def two_sided_quantile() -> float:
    """Returns the normal quantile of a two-sided interval at the test level."""
    return NormalDist().inv_cdf(1.0 - ALPHA / 2.0)


def normal_comparisons(shift: float) -> list[ArmComparison]:
    """Runs many tests on normal arms whose means differ by the given shift."""
    generator = np.random.default_rng(FIRST_SEED)
    comparisons: list[ArmComparison] = []
    for _ in range(REPLICATIONS):
        control = generator.normal(size=ARM_SESSIONS)
        treatment = generator.normal(loc=shift, size=ARM_SESSIONS)
        comparisons.append(compare_arms(control, treatment, ALPHA))
    return comparisons


def welch_error(control: FloatArray, treatment: FloatArray) -> float:
    """Computes by hand the Welch standard error of a difference of means."""
    control_part = control.var(ddof=1) / len(control)
    treatment_part = treatment.var(ddof=1) / len(treatment)
    return math.sqrt(control_part + treatment_part)


def hand_difference(
    metric: FloatArray,
    treated: FloatArray,
    inside: FloatArray,
) -> float:
    """Computes the difference of arm means inside one segment."""
    treated_mean = metric[(inside == 1.0) & (treated == 1.0)].mean()
    control_mean = metric[(inside == 1.0) & (treated == 0.0)].mean()
    return float(treated_mean - control_mean)


def segment_table() -> pd.DataFrame:
    """Builds the segment report of the small hand-made experiment."""
    metric = np.array(SEGMENT_METRIC)
    treated = np.array(SEGMENT_TREATED)
    labels = np.array(SEGMENT_LABELS, dtype=np.object_)
    return segment_effects(metric, treated, labels, ALPHA, MINIMUM_SEGMENT_SESSIONS)


def segment_value(table: pd.DataFrame, segment: str, column: str) -> float:
    """Reads one number of the segment report."""
    row = table[table["segment"] == segment]
    return float(row[column].to_numpy(dtype=np.float64)[0])


def test_assignment_treats_exactly_half() -> None:
    treated = assign_sessions(ODD_SESSION_COUNT, np.random.default_rng(FIRST_SEED))
    assert treated.dtype == np.bool_
    assert len(treated) == ODD_SESSION_COUNT
    assert int(treated.sum()) == ODD_SESSION_COUNT // 2


def test_assignment_differs_between_seeds() -> None:
    first = assign_sessions(ODD_SESSION_COUNT, np.random.default_rng(FIRST_SEED))
    second = assign_sessions(ODD_SESSION_COUNT, np.random.default_rng(SECOND_SEED))
    assert not np.array_equal(first, second)


def test_dominant_package_is_always_booked() -> None:
    outcomes = draw_dominated(DOMINANT_UTILITY, NEGLIGIBLE_UTILITY)
    assert np.all(outcomes.booked == 1.0)
    assert np.all(outcomes.profit == SHOWN_PROFIT[0])


def test_dominant_outside_option_books_nothing() -> None:
    outcomes = draw_dominated(NEGLIGIBLE_UTILITY, DOMINANT_UTILITY)
    assert np.all(outcomes.booked == 0.0)
    assert np.all(outcomes.profit == 0.0)


def test_booking_frequencies_match_the_choice_model() -> None:
    outcomes = draw_outcomes(
        repeated_rows(FREQUENCY_UTILITY, FREQUENCY_SESSIONS),
        np.full(FREQUENCY_SESSIONS, FREQUENCY_OUTSIDE),
        repeated_rows(SHOWN_PROFIT, FREQUENCY_SESSIONS),
        FREQUENCY_NOISE_SIGMA,
        np.random.default_rng(FIRST_SEED),
    )
    expected = reference_probabilities()
    for position, profit in enumerate(SHOWN_PROFIT):
        share = np.count_nonzero(outcomes.profit == profit) / FREQUENCY_SESSIONS
        chance = float(expected[position])
        error = math.sqrt(chance * (1.0 - chance) / FREQUENCY_SESSIONS)
        assert abs(share - chance) < BINOMIAL_ERRORS * error
    assert np.array_equal(outcomes.booked == 1.0, outcomes.profit > 0.0)


def test_comparison_matches_hand_computation() -> None:
    control = np.array(CONTROL_VALUES)
    treatment = np.array(TREATMENT_VALUES)
    comparison = compare_arms(control, treatment, ALPHA)
    difference = float(treatment.mean() - control.mean())
    error = welch_error(control, treatment)
    assert close(comparison.control_mean, float(control.mean()))
    assert close(comparison.treatment_mean, float(treatment.mean()))
    assert close(comparison.difference, difference)
    assert close(comparison.standard_error, error)


def test_comparison_interval_p_value_and_lift() -> None:
    control = np.array(CONTROL_VALUES)
    treatment = np.array(TREATMENT_VALUES)
    comparison = compare_arms(control, treatment, ALPHA)
    difference = float(treatment.mean() - control.mean())
    error = welch_error(control, treatment)
    tail = 1.0 - NormalDist().cdf(abs(difference) / error)
    assert close(comparison.lower, difference - two_sided_quantile() * error)
    assert close(comparison.upper, difference + two_sided_quantile() * error)
    assert close(comparison.p_value, 2.0 * tail)
    assert close(comparison.relative_lift, difference / float(control.mean()))


def test_identical_arms_show_no_difference() -> None:
    values = np.array(CONTROL_VALUES)
    comparison = compare_arms(values, values.copy(), ALPHA)
    assert close(comparison.difference, 0.0)
    assert close(comparison.p_value, 1.0)


def test_cuped_removes_the_explained_variance() -> None:
    generator = np.random.default_rng(FIRST_SEED)
    covariate = generator.normal(size=CUPED_SESSIONS)
    noise = CUPED_NOISE * generator.normal(size=CUPED_SESSIONS)
    metric = CUPED_OFFSET + CUPED_SLOPE * covariate + noise
    result = cuped_adjust(metric, covariate)
    assert abs(result.theta - CUPED_SLOPE) < THETA_TOLERANCE
    assert result.variance_reduction > MINIMUM_REDUCTION
    assert close(float(result.adjusted.mean()), float(metric.mean()))


def test_constant_covariate_changes_nothing() -> None:
    metric = np.array(CONTROL_VALUES)
    result = cuped_adjust(metric, np.full(len(metric), CUPED_OFFSET))
    assert result.theta == 0.0
    assert close(result.variance_reduction, 0.0)
    assert np.allclose(result.adjusted, metric)


def test_required_sessions_follow_the_formula() -> None:
    sessions = required_sessions(UNIT_DEVIATION, UNIT_EFFECT, ALPHA, POWER)
    assert sessions == EXPECTED_SESSIONS


def test_half_the_effect_needs_four_times_the_sessions() -> None:
    full = required_sessions(UNIT_DEVIATION, UNIT_EFFECT, ALPHA, POWER)
    half = required_sessions(UNIT_DEVIATION, UNIT_EFFECT / 2.0, ALPHA, POWER)
    assert abs(half - SIZE_GROWTH * full) <= SIZE_GROWTH


def test_rejection_rate_counts_small_p_values() -> None:
    rate = rejection_rate(np.array(P_VALUES), ALPHA)
    assert close(rate, EXPECTED_REJECTION)


def test_coverage_rate_counts_intervals_with_the_truth() -> None:
    rate = coverage_rate(np.array(LOWER_BOUNDS), np.array(UPPER_BOUNDS), TRUE_EFFECT)
    assert close(rate, EXPECTED_COVERAGE)


def test_aa_test_rejects_at_the_nominal_level() -> None:
    comparisons = normal_comparisons(0.0)
    p_values = np.array([comparison.p_value for comparison in comparisons])
    assert abs(rejection_rate(p_values, ALPHA) - ALPHA) < RATE_TOLERANCE


def test_intervals_cover_the_true_shift() -> None:
    comparisons = normal_comparisons(NORMAL_SHIFT)
    lower = np.array([comparison.lower for comparison in comparisons])
    upper = np.array([comparison.upper for comparison in comparisons])
    coverage = coverage_rate(lower, upper, NORMAL_SHIFT)
    assert abs(coverage - CONFIDENCE) < RATE_TOLERANCE


def test_segment_difference_matches_hand_computation() -> None:
    table = segment_table()
    metric = np.array(SEGMENT_METRIC)
    treated = np.array(SEGMENT_TREATED, dtype=np.float64)
    labels = np.array(SEGMENT_LABELS, dtype=np.object_)
    inside = (labels == ESTIMABLE_SEGMENT).astype(np.float64)
    difference = hand_difference(metric, treated, inside)
    assert close(segment_value(table, ESTIMABLE_SEGMENT, "difference"), difference)
    lower = segment_value(table, ESTIMABLE_SEGMENT, "lower")
    assert lower < difference < segment_value(table, ESTIMABLE_SEGMENT, "upper")


def test_small_segment_is_reported_without_estimates() -> None:
    table = segment_table()
    sessions = SEGMENT_LABELS.count(SMALL_SEGMENT)
    assert int(segment_value(table, SMALL_SEGMENT, "sessions")) == sessions
    for column in ["difference", "lower", "upper", "p_value"]:
        assert math.isnan(segment_value(table, SMALL_SEGMENT, column))


def test_arm_of_one_session_is_rejected() -> None:
    with pytest.raises(ValueError):
        compare_arms(np.array(SINGLE_SESSION), np.array(CONTROL_VALUES), ALPHA)


def test_relative_lift_is_undefined_for_a_zero_control_mean() -> None:
    comparison = compare_arms(np.array(ZERO_CONTROL), np.array(CONTROL_VALUES), ALPHA)
    assert close(comparison.difference, float(np.mean(CONTROL_VALUES)))
    assert math.isnan(comparison.relative_lift)
    assert comparison.lower < comparison.difference < comparison.upper


def test_segment_sizes_add_up_to_the_total() -> None:
    table = segment_table()
    assert int(table["sessions"].sum()) == len(SEGMENT_LABELS)
    assert {"sessions", "difference", "lower", "upper", "p_value"} <= set(table.columns)
