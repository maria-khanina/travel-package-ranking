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
    shape = shown_utility.shape
    taste = noise_sigma * generator.normal(size=shape)
    scores = shown_utility + taste + generator.gumbel(size=shape)
    leaving = outside + generator.gumbel(size=outside.shape)
    best = scores.argmax(axis=1)
    booked = scores.max(axis=1) > leaving
    chosen = np.take_along_axis(profit, best[:, None], axis=1)[:, 0]
    return SessionOutcomes(
        booked=booked.astype(np.float64),
        profit=np.where(booked, chosen, 0.0),
    )
