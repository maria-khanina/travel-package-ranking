# A/B test: implementation plan

Design: `docs/02_ab_test.md`. Each step ends with green tests, clean pycodestyle and
strict pyright, a look at the diff and a commit. The tests of all steps are written in
step 0 against the interfaces below; from step 1 on, `docs/.implementing` freezes
them.

## Interfaces fixed by the tests

| Module | Function | Contract |
|---|---|---|
| `src/ab_test/assignment.py` | `assign_sessions(count: int, generator: np.random.Generator) -> BoolArray` | random split, treated count is `count // 2` |
| `src/ab_test/outcomes.py` | `draw_outcomes(shown_utility: FloatArray, outside: FloatArray, profit: FloatArray, noise_sigma: float, generator) -> SessionOutcomes` | `SessionOutcomes(booked: FloatArray, profit: FloatArray)`, one value per session; at most one booking; profit is that of the booked package or 0 |
| `src/ab_test/analysis.py` | `compare_arms(control: FloatArray, treatment: FloatArray, alpha: float) -> ArmComparison` | means, difference, Welch standard error, normal interval, two-sided p-value, relative lift |
| `src/ab_test/analysis.py` | `cuped_adjust(metric: FloatArray, covariate: FloatArray) -> CupedResult` | `adjusted`, `theta = cov / var`, `variance_reduction = 1 - var(adjusted) / var(metric)`; the mean is kept |
| `src/ab_test/power.py` | `required_sessions(standard_deviation: float, effect: float, alpha: float, power: float) -> int` | sessions per arm, `ceil(2 (z_{1-alpha/2} + z_power)^2 sd^2 / effect^2)` |
| `src/ab_test/validation.py` | `rejection_rate(p_values: FloatArray, alpha: float) -> float`, `coverage_rate(lower: FloatArray, upper: FloatArray, truth: float) -> float` | shares over replications |
| `src/ab_test/segments.py` | `segment_effects(metric: FloatArray, treated: BoolArray, segments: TextArray, alpha: float, minimum_sessions: int) -> pd.DataFrame` | one row per segment: segment, sessions, means, difference, lower, upper, p-value; estimates are NaN when the smaller arm of the segment is below `minimum_sessions` |

`compare_arms` raises `ValueError` when an arm has fewer than 2 sessions and returns
a NaN relative lift when the control mean is zero.

## Step 0. Tests and settings

- **Goal:** the specification as tests, and the settings the code will read.
- **Files:** `tests/test_ab_test.py`, `tests/ab_test_check_settings.py` (inputs and
  tolerances), `configs/ab_test_settings.py`, `src/settings/ab_test_models.py`,
  `src/settings/loader.py` (new field `ab_test`), and the modules of `src/ab_test/`
  with the interfaces above whose bodies raise `NotImplementedError`, so that the
  tests type-check and fail only on the missing implementation.
- **Tests (synthetic data with known answers):**
  - assignment: exact half treated, different seeds give different splits;
  - outcomes: a dominant package is always booked, a dominant outside option books
    nothing, profit equals the booked package's profit, booking frequencies over many
    draws match the noise-averaged logit probabilities within a binomial tolerance;
  - analysis: the difference and the Welch error equal hand computation; identical
    arms give p-value 1; CUPED on `y = 2x + noise` gives theta near 2, a large
    variance reduction and the same mean; a constant covariate gives no reduction;
  - power: `sd = 1, effect = 0.1, alpha = 0.05, power = 0.8` gives 1,570; halving
    the effect quadruples the size;
  - validation: rates on hand-made arrays; an A/A test on normal data over 2,000
    replications rejects 5% ± 1.5 points;
  - segments: per-segment differences equal those computed by hand, segment sizes
    add up to the total.
- **Success:** the tests run and fail only on the missing implementation.

## Step 1. Baseline: one run of each test

- **Goal:** a working `ab-test` command that runs each of the two tests once on the
  20,000 sessions and writes the table.
- **Files:** `src/ab_test/assignment.py`, `outcomes.py`, `analysis.py` (without CUPED),
  `pipeline.py`, `src/cli.py`.
- **Tests:** assignment, outcomes, analysis except CUPED.
- **Measure:** conversion and profit per arm, difference with interval and p-value
  for both tests, next to the true effect from `true_estimate`.
- **Success:** the true effect lies in the interval of the large test; the means are
  close to the README margin table (61.4, 82.8, 85.2) within the interval.
- **Out of the step:** power, CUPED, replications, segments.

## Step 2. Power analysis

- **Goal:** the planned sessions per arm for both effects from a pilot A/A run.
- **Files:** `src/ab_test/power.py`, `pipeline.py`.
- **Tests:** power.
- **Measure:** sessions per arm for +3% and +39%.
- **Success:** the numbers are written to the results table; the A/B runs of step 1
  use the planned size (drawn with replacement where it exceeds 20,000).

## Step 3. CUPED

- **Goal:** the variance reduction from the model's predicted profit under the
  control policy, and the planned size with CUPED.
- **Files:** `analysis.py`, `power.py` use, `pipeline.py`.
- **Tests:** CUPED.
- **Measure:** variance reduction, sessions saved, intervals with and without CUPED.
- **Success:** a positive variance reduction; the estimated difference stays within
  its interval of the unadjusted one.

## Step 4. Validation against the truth

- **Goal:** A/A false positive rate, coverage and empirical power on 1,000
  replications per test, with and without CUPED.
- **Files:** `src/ab_test/validation.py`, `pipeline.py`.
- **Tests:** validation.
- **Success:** the acceptance bands of the design (false positives 3.5% to 6.5%,
  coverage 93.5% to 96.5%, power within 5 points of 80% at the planned size). A
  miss is a result: it is recorded with its likely cause as a hypothesis and checked.

## Step 5. Segments

- **Goal:** the effect of each test by device and by party.
- **Files:** `src/ab_test/segments.py`, `pipeline.py`.
- **Tests:** segments.
- **Success:** the segment table is written; segment sizes add up to the total.

## Step 6. README

- **Goal:** a README section with the planned sizes, both tests, CUPED, segments and
  validation, and the run command.
- **Success:** every number in the section comes from the written result tables.
