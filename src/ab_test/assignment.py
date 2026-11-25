"""Splits the sessions of an experiment at random between control and treatment."""

import numpy as np

from src.world.models import BoolArray


def assign_sessions(count: int, generator: np.random.Generator) -> BoolArray:
    """Marks a random half of the sessions as treated."""
    treated = np.zeros(count, dtype=np.bool_)
    treated[generator.permutation(count)[: count // 2]] = True
    return treated
