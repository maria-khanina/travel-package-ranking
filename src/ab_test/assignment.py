"""Splits the sessions of an experiment at random between control and treatment."""

import numpy as np

from src.world.models import BoolArray


def assign_sessions(count: int, generator: np.random.Generator) -> BoolArray:
    """Marks a random half of the sessions as treated."""
    raise NotImplementedError
