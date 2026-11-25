"""Draws what each traveler of an experiment books under the shown policy."""

from dataclasses import dataclass

import numpy as np

from src.world.models import FloatArray


@dataclass
class SessionOutcomes:
    booked: FloatArray
    profit: FloatArray


def draw_outcomes(
    shown_utility: FloatArray,
    outside: FloatArray,
    profit: FloatArray,
    noise_sigma: float,
    generator: np.random.Generator,
) -> SessionOutcomes:
    """Draws one booking or none per session by the true random utility model."""
    raise NotImplementedError
