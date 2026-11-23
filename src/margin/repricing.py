"""Builds policies that keep every candidate shown and change only the margins.

In the logs the traveler chose among all candidates of a search. Inverse propensity
scoring corrects the choice of margin but not a change of the choice set, so it
can only grade policies that show the same candidates as the logs. These policies
reprice all candidates: one margin for all of them, or the margin each model
expects to be most profitable for each package.
"""

import numpy as np

from src.margin.policies import Policy, profit_grid
from src.world.models import FloatArray, IntArray

ALL_CANDIDATES_PREFIX = "all_candidates_"


def all_candidates(sessions: int, pool: int) -> IntArray:
    """Lists every candidate of every session as the shown set."""
    return np.tile(np.arange(pool, dtype=np.int64), (sessions, 1))


def repricing_policies(
    probability: FloatArray,
    net_cost: FloatArray,
    grid: list[float],
) -> dict[str, Policy]:
    """Builds the fixed-margin and the model-optimized repricing policies."""
    sessions, pool, _ = probability.shape
    shown = all_candidates(sessions, pool)
    policies: dict[str, Policy] = {}
    for position, margin in enumerate(grid):
        margins = np.full(shown.shape, position, dtype=np.int64)
        policies[f"{ALL_CANDIDATES_PREFIX}fixed_{margin:.2f}"] = Policy(shown, margins)
    best = (probability * profit_grid(net_cost, grid)).argmax(axis=2)
    policies[f"{ALL_CANDIDATES_PREFIX}optimized"] = Policy(shown, best)
    return policies
