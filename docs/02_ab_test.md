# A/B test of pricing policies on the simulated market

## Brief

**Task.** The margin step grades pricing policies by their expected profit under the
true choice model, which a real travel site never knows. A real site decides by an
online experiment: it splits live traffic at random between the current policy and
the new one, observes bookings, and judges the difference with its noise. This
feature runs such an experiment on the simulator, so the project shows the whole
path from an offline estimate to a launch decision, and checks the experiment
method itself against the known truth, which a live test cannot do.

**Goals.**

1. Plan the test before running it: the sample size needed to detect a given lift
   with 80% power at a 5% significance level (power analysis).
2. Run the test: random assignment of sessions, observed bookings, conversion and
   profit per session per arm, the difference with a 95% interval and a p-value.
3. Reduce variance with CUPED and report how many sessions it saves.
4. Report the effect by traveler segment.
5. Validate the method on many replications: the false positive rate of an A/A
   test, the coverage of the true effect by the 95% intervals, and the empirical
   power against the planned one.

**Agreed with the user (2026-11-25).**

- Two tests: a small effect, the best single margin of 20% for all packages against
  the optimized margin and slate (82.8 against 85.2 per session, about +3%), and a
  large effect, ranking by booking probability at logged margins against the
  optimized policy (61.4 against 85.2, about +39%).
- Analysis: the validation against the truth, effects by segment, and CUPED.
- Base scenario only (`data/`); the confounded scenario is out of scope.

**Out of scope.** Sequential testing and early stopping, multi-armed bandits,
interference between sessions, the confounded scenario, novelty effects over time.

## Design

### Unit, population and outcome

- **Unit of randomization:** the search session. Sessions are independent in the
  simulator, so there is no interference between arms.
- **Population:** the 20,000 development sessions of the margin sample (the closed
  test split stays closed). For sample sizes above the sample, sessions are drawn
  with replacement; each draw is a new traveler with the same visible context,
  because the booking noise is drawn afresh.
- **Outcome of one session under a policy:** the policy shows its slate of 3
  packages at its margins; the traveler books one of them or none by the true
  choice model: noise-free utility from `TrueTables`, a fresh normal taste noise
  (`noise_sigma`) per shown package and a fresh Gumbel draw per option including
  leaving, the highest score wins. This is the same random utility model that
  generated the logs and that `true_estimate` averages over.
- **Metrics per session:** booked (0 or 1) and profit (margin times net cost of the
  booked package, 0 without a booking). Primary metric: profit per session;
  secondary: conversion.
- **True effect:** the difference of the mean `true_estimate` of the two policies
  over the population; it is known exactly up to the noise averaging and is used
  only for validation.

### Analysis

- **Difference of means** with the Welch standard error, a normal 95% interval and a
  two-sided p-value; the relative lift is the difference over the control mean.
- **CUPED.** The covariate must not depend on the assignment. There is no
  pre-experiment period in a session, so the covariate is the model's predicted
  profit of the session under the control policy (`model_estimate`), known before
  the session is assigned (the variant of CUPED with a model prediction, also called
  CUPAC). The adjusted metric is `y - theta * (x - mean(x))` with
  `theta = cov(x, y) / var(x)` pooled over both arms; the variance reduction is
  `1 - var(adjusted) / var(y)`.
- **Power analysis.** From a pilot (a separate A/A run of the control policy) the
  standard deviation of profit per session gives the required sessions per arm,
  `n = 2 (z_{1-alpha/2} + z_{power})^2 sigma^2 / delta^2`, for the minimum
  detectable effects of the two tests; with CUPED, sigma is that of the adjusted
  metric.
- **Segments.** The same difference and interval within each segment: device
  (iOS, Android, Desktop) and party (with children or adults only). The segment
  intervals are reported without a multiple-testing claim; a note says so.
- **Degenerate cases** (added 2026-11-25). An arm with fewer than 2 sessions has no
  variance estimate, so the comparison of two arms raises an error instead of
  returning a silent NaN: in the main test it can only be a bug of the caller. A
  zero control mean leaves the relative lift undefined (NaN), while the absolute
  difference and its interval stay valid. A segment whose smaller arm is below a
  configured minimum keeps its row with the session count and empty estimates, so
  the report shows that it exists but cannot be judged.

### Validation against the truth

On R replications (new assignment and new booking draws each time) at a given sample
size:

- **A/A false positive rate:** both arms run the control policy; the share of
  p-values below 0.05 should be about 5%.
- **Coverage:** the share of 95% intervals of the A/B difference that contain the
  true effect should be about 95%.
- **Empirical power:** the share of significant A/B results, compared with the
  planned power at the size from the power analysis, with and without CUPED.

### Components

| File | Role |
|---|---|
| `configs/ab_test.py` | the two tests (control and treatment policy names), alpha, power, sample-size grid, replications, seeds, segment columns, output file names |
| `src/settings/ab_test_models.py` | Pydantic models of these settings, attached to `Settings` |
| `src/ab_test/outcomes.py` | draws the booking and profit of each session under a policy |
| `src/ab_test/analysis.py` | difference of means, Welch interval and p-value, CUPED adjustment |
| `src/ab_test/power.py` | required sample size from a pilot |
| `src/ab_test/validation.py` | A/A false positive rate, coverage and empirical power over replications |
| `src/ab_test/segments.py` | the effect per segment |
| `src/ab_test/pipeline.py` | prepares the margin run, builds the policies, runs everything, writes CSV tables to `results/ab_test/<experiment>/` |
| `src/cli.py` | new command `ab-test` |
| `tests/test_ab_test.py` | tests of the statistics on synthetic data with known answers, and of the outcome draw |

The pipeline reuses `prepare_run`, `slate_policies`, `fixed_policies`,
`true_estimate` and `model_estimate` from the margin step without changing them.

### Decisions and rejected options

- **Booking draws instead of expected values:** a test on expected profit would have
  no outcome noise and would make the power analysis meaningless.
- **Welch normal interval instead of the bootstrap** inside replications: with
  thousands of sessions per arm the normal interval is accurate, and the bootstrap
  over thousands of replications would be slow; the main reported test also gets a
  bootstrap interval for comparison.
- **Model prediction as the CUPED covariate:** a session has no history before the
  test; a prediction made before assignment keeps the estimate unbiased.

### Risks

- Profit per session is heavy-tailed (most sessions earn 0), so the normal interval
  may undercover at small sizes; the coverage check measures it.
- The required size for +3% may exceed the 20,000 sessions; sampling with
  replacement covers it, and the README states that the population is the sample.

### Acceptance

- Tests green, pycodestyle and strict pyright clean.
- A/A false positive rate within 3.5% to 6.5% and coverage within 93.5% to 96.5% on
  1,000 replications (a binomial band of about ±1.5 points around the target).
- Empirical power at the planned size within 5 points of 80%.
- README section with the planned sizes, the result of both tests, the CUPED
  saving, the segment table and the validation numbers.
