"""Computes stabilized inverse propensity weights of the logged margins.

Weighting every row by the overall share of its margin divided by the chance the
historical rule chose that margin for this row makes the margin independent of
everything the rule looked at, the hidden demand shock included. A model trained
on the weighted rows learns the effect of the margin itself rather than the
demand that came with it.
"""

import numpy as np

from src.world.models import FloatArray


def compute_propensity_weights(
    margins: FloatArray,
    propensity: FloatArray,
) -> FloatArray:
    """Divides the overall share of each logged margin by its row propensity."""
    _, inverse, counts = np.unique(margins, return_inverse=True, return_counts=True)
    share = counts / len(margins)
    return share[inverse] / propensity
