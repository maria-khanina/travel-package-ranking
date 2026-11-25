# A/B test: progress

## Step 0. Tests and settings

Done 2026-11-25. 18 tests in `tests/test_ab_test.py`; all fail with
`NotImplementedError` from the interface stubs; the other 61 tests pass.
pycodestyle and strict pyright are clean on the new and changed files (pyright
reports 3 errors in `src/labels/`, which this step did not touch).

## Step 1. Baseline: one run of each test

In progress. While `docs/.implementing` is present, red tests block the end of a
reply, and the tests of steps 2 to 5 were written in step 0; so all statistical
functions of `src/ab_test/` (assignment, outcomes, comparison, CUPED, power,
rejection and coverage rates, segments) are implemented in this step, and steps 2
to 5 wire them into the pipeline and run them on the data.

The segment test of step 0 had a control arm of one session, whose variance is
undefined (NaN interval). With the user's agreement the flag was removed and the
degenerate cases were specified and tested (see the design): an arm of one session
raises an error, a zero control mean gives a NaN relative lift, a segment below the
minimum is reported with empty estimates. 21 tests pass.

## Step 2. Power analysis

Not started.

## Step 3. CUPED

Not started.

## Step 4. Validation against the truth

Not started.

## Step 5. Segments

Not started.

## Step 6. README

Not started.
