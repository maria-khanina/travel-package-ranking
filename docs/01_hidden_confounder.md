# Hidden confounder in the pricing logs

## Goal

Show the causal limitation of learning a price effect from logged data, and a fix.
The historical pricing rule raises margins when a hidden demand shock is present
(for example, a local event the pricing team knew about); the shock also makes
travelers book more. A model trained on these logs mistakes the shock for a weak
price effect, so its margin policy overprices. Training with the logged margin
propensities removes the bias, and inverse propensity scoring (IPS) on the logs
gives an independent check before launch.

## Finding that comes first: why IPS disagreed with the truth

In the logs the traveler chose among all 30 candidates of a search, while the
compared policies show a slate of 3. Weighting by 1/propensity corrects the choice
of margin, not the change of the choice set, so IPS of a 3-package slate is biased
down (72.6 against a true 85.2 for the optimized policy). Checked on the current
data with policies that keep all 30 packages and change only margins:

| Policy (all 30 shown) | True | Model | IPS |
|---|---|---|---|
| one margin of 15% | 164.0 | 161.7 | 162.2 ± 13.9 |
| one margin of 20% | 168.7 | 164.0 | 160.3 ± 16.1 |
| margin per package optimized by the model | 174.0 | 178.7 | 172.5 ± 17.6 |

Fix: the margin step also reports policies that reprice all candidates (where IPS is
valid), and leaves IPS empty for slate policies, with the reason in the README.

## Design

1. **Settings** `configs/confounder.py`: share of (session, destination) pairs with
   a shock, utility shift of the shock, margin shift of the logging rule, seed,
   data directory of the scenario (`data_confounded/`, git-ignored).
2. **Scenario** (`src/world/confounder.py`), applied in `simulate` when the global
   CLI flag `--confounded` is set: draws a hidden column `hid_demand_shock` per
   (session, destination) with its own random generator (world and texts stay
   unchanged), then redraws the logged margins and propensities with the logging
   rule's target raised by the margin shift where the shock is present. The
   propensities stay those the pricing system really used, so they include the shock.
3. **Utility**: `compute_utilities` adds `utility_shift * hid_demand_shock`; the
   column is all zeros in the base scenario, so the base results do not change.
4. **Propensity-weighted training**: an experiment option that passes stabilized
   weights `p(margin) / p(margin | row)` to CatBoost, so that in the weighted data the
   margin is independent of the shock.
5. **Margin step**: the all-candidate policies above, plus the price response check
   for both models.

The shock is drawn per session and destination, so no model feature (city,
departure month, days to departure) can reconstruct it.

## Runs

Base data are reused through a link to the existing `data/`; the scenario writes to
`data_confounded/` after copying `world/`, `texts/` and the cached text vectors.
`simulate --confounded`, `features --confounded`, `train --confounded --only
cosines_and_labels cosines_and_labels_weighted`, `margin --confounded`. Expected
machine time is about 20 minutes.

## Acceptance

- Base scenario: all existing tests pass and `margin_policies.csv` is unchanged
  except for the new rows and the emptied slate IPS column.
- Confounded scenario, unweighted model: the predicted booking rate at high margins
  is above the truth, and the true profit of its optimized policy is below the
  model's estimate.
- Weighted model: the price response is close to the truth, and its optimized policy
  earns more true profit than the unweighted one.
- IPS of all-candidate policies agrees with the truth within its interval in both
  scenarios, and flags the unweighted model's estimate as too high.
- New tests: the shock changes only rows where it is drawn, the redrawn propensities
  sum to one per row, the weights have mean about one.

## Out of scope

Slate-level IPS (it would need logs with randomized slates), other confounders,
rerunning the 14 feature and loss setups.
