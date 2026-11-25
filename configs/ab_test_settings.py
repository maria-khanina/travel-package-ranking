"""Settings of the simulated A/B tests of pricing policies.

Each test compares a control policy with a treatment policy of the margin step by
their names. The small test pits the best single margin against the optimized
margin and slate, the large test the conversion ranker at logged margins against
the same optimized policy.
"""

from src.margin.policies import CONVERSION, OPTIMIZED
from src.settings.ab_test_models import (
    AbTestSettings,
    PolicyTestRecord,
    SegmentSettings,
)


AB_TEST = AbTestSettings(
    tests=[
        PolicyTestRecord(
            name="single_margin_against_optimized",
            control="fixed_margin_0.20",
            treatment=OPTIMIZED,
        ),
        PolicyTestRecord(
            name="conversion_ranker_against_optimized",
            control=CONVERSION,
            treatment=OPTIMIZED,
        ),
    ],
    alpha=0.05,
    power=0.8,
    replications=1000,
    bootstrap_resamples=2000,
    seed=71,
    segments=SegmentSettings(
        device_column="user_device",
        children_column="n_children",
        family_label="with children",
        adults_label="adults only",
    ),
    results_folder="ab_test",
    comparison_file="ab_test_comparison.csv",
    power_file="ab_test_power.csv",
    validation_file="ab_test_validation.csv",
    segment_file="ab_test_segments.csv",
)
