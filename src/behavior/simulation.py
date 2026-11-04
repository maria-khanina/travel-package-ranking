"""Simulates which package each traveler books and calibrates conversion."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.behavior.utility import compute_outside_utilities, compute_utilities
from src.settings.loader import Settings
from src.world.helpers import read_floats
from src.world.models import FloatArray, IntArray


@dataclass
class BookingOutcome:
    is_booked: IntArray
    probabilities: FloatArray


def compute_choice_probabilities(
    outside_utilities: FloatArray,
    utilities: FloatArray,
) -> FloatArray:
    """Turns utilities into booking probabilities, outside option first."""
    scores = np.concatenate([outside_utilities.reshape(-1, 1), utilities], axis=1)
    scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    return weights / weights.sum(axis=1, keepdims=True)


def pick_booked_packages(scores: FloatArray) -> IntArray:
    """Marks the package with the best noisy score unless leaving wins."""
    picked = scores.argmax(axis=1)
    booked = np.zeros((scores.shape[0], scores.shape[1] - 1), dtype=np.int64)
    sold = np.flatnonzero(picked > 0)
    booked[sold, picked[sold] - 1] = 1
    return booked.reshape(-1)


def simulate_bookings(
    rows: pd.DataFrame,
    settings: Settings,
    outside_constant: float,
    margin: FloatArray,
) -> BookingOutcome:
    """Draws the booked package of every session at the given margins."""
    pool_size = settings.pool.size
    sessions = rows.iloc[::pool_size]
    shape = (len(sessions), pool_size)
    utilities = compute_utilities(rows, settings, margin, True).reshape(shape)
    outside = compute_outside_utilities(sessions, settings, outside_constant)
    package_noise = read_floats(rows, "hid_gumbel").reshape(shape)
    outside_noise = read_floats(sessions, "hid_gumbel_out")
    scores = np.concatenate(
        [(outside + outside_noise).reshape(-1, 1), utilities + package_noise],
        axis=1,
    )
    return BookingOutcome(
        is_booked=pick_booked_packages(scores),
        probabilities=compute_choice_probabilities(outside, utilities),
    )


def measure_conversion(
    rows: pd.DataFrame,
    settings: Settings,
    outside_constant: float,
) -> float:
    """Measures the share of sessions that end with a booking."""
    margin = read_floats(rows, "margin_percentage")
    outcome = simulate_bookings(rows, settings, outside_constant, margin)
    return float(outcome.is_booked.sum() / (len(rows) // settings.pool.size))


def calibrate_outside_constant(
    rows: pd.DataFrame,
    settings: Settings,
    target_conversion: float,
) -> float:
    """Finds by bisection the constant that yields the target conversion."""
    choice = settings.choice
    lower = choice.calibration_lower
    upper = choice.calibration_upper
    for _ in range(choice.calibration_iterations):
        middle = (lower + upper) / 2.0
        if measure_conversion(rows, settings, middle) > target_conversion:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2.0
