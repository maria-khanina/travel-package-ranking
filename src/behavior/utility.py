"""Combines all utility components into the value of each package."""

import pandas as pd

from src.behavior.price_terms import compute_price_term
from src.behavior.taste_terms import compute_taste_term
from src.behavior.terms import (
    compute_fit_term,
    compute_flexibility_term,
    compute_flight_term,
    compute_quality_term,
)
from src.settings.loader import Settings
from src.world.helpers import read_floats, read_integers
from src.world.models import FloatArray


def compute_utilities(
    rows: pd.DataFrame,
    settings: Settings,
    margin: FloatArray,
    include_noise: bool,
) -> FloatArray:
    """Computes the utility of every package at the given margins."""
    utility = compute_fit_term(rows, settings) + compute_quality_term(rows, settings)
    utility = utility + compute_taste_term(rows, settings)
    utility = utility + compute_flexibility_term(rows, settings)
    utility = utility + compute_flight_term(rows, settings)
    utility = utility + compute_price_term(rows, margin, settings.choice)
    if include_noise:
        noise = read_floats(rows, "hid_eps_std")
        utility = utility + settings.choice.noise_sigma * noise
    return utility


def compute_outside_shifts(sessions: pd.DataFrame, settings: Settings) -> FloatArray:
    """Sums the context effects that make leaving without booking likelier."""
    choice = settings.choice
    days = read_integers(sessions, "days_to_departure")
    night = read_integers(sessions, "search_hour") < choice.outside_night_limit_hour
    far = days > choice.outside_far_days
    urgent = days < choice.outside_last_minute_days
    weekend = read_integers(sessions, "search_dow") >= choice.outside_weekend_start
    shifts = choice.outside_night_shift * night + choice.outside_far_shift * far
    shifts = shifts + choice.outside_last_minute_shift * urgent
    return shifts + choice.outside_weekend_shift * weekend


def compute_outside_utilities(
    sessions: pd.DataFrame,
    settings: Settings,
    outside_constant: float,
) -> FloatArray:
    """Computes the utility of leaving without booking for every session."""
    choice = settings.choice
    device_shift = sessions["user_device"].map(choice.outside_device_shift)
    noise = read_floats(sessions, "hid_out_noise_std")
    noise_term = choice.outside_noise_standard_deviation * noise
    shifts = compute_outside_shifts(sessions, settings)
    return outside_constant + device_shift.to_numpy(dtype=float) + shifts + noise_term
