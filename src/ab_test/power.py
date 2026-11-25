"""Plans the number of sessions an experiment needs to detect an effect."""

import math
from statistics import NormalDist


def required_sessions(
    standard_deviation: float,
    effect: float,
    alpha: float,
    power: float,
) -> int:
    """Computes the sessions per arm for a two-sided test of a mean difference."""
    normal = NormalDist()
    quantiles = normal.inv_cdf(1.0 - alpha / 2.0) + normal.inv_cdf(power)
    ratio = standard_deviation / effect
    return math.ceil(2.0 * quantiles**2 * ratio**2)
