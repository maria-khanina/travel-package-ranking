"""Non-price components of the utility a traveler gets from a package."""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.world.helpers import read_floats, read_integers
from src.world.models import FloatArray


def compute_fit_term(rows: pd.DataFrame, settings: Settings) -> FloatArray:
    """Rewards packages whose aspects match the hidden wishes, with saturation."""
    choice = settings.choice
    fit = read_floats(rows, "hid_fit")
    return choice.fit_amplitude * np.tanh(choice.fit_slope * (fit - choice.fit_center))


def compute_quality_term(rows: pd.DataFrame, settings: Settings) -> FloatArray:
    """Rewards stars and guest rating, plus the hidden hotel charm."""
    choice = settings.choice
    luxury_wish = np.maximum(read_floats(rows, "hid_luxury_wish"), 0.0)
    star_gap = read_integers(rows, "stars") - choice.star_reference
    rating_gap = read_floats(rows, "rating") - choice.rating_reference
    stars = choice.star_coefficient * (1.0 + luxury_wish) * star_gap
    rating = choice.rating_coefficient * rating_gap
    return stars + rating + read_floats(rows, "hid_hotel_effect")


def compute_flexibility_term(rows: pd.DataFrame, settings: Settings) -> FloatArray:
    """Rewards free cancellation, more so for distant departures."""
    choice = settings.choice
    days = read_integers(rows, "days_to_departure")
    far = days > choice.flexibility_far_days
    free = rows["free_cancellation"].to_numpy(dtype=np.float64)
    return choice.flexibility_coefficient * free * (1.0 + far)


def compute_flight_term(rows: pd.DataFrame, settings: Settings) -> FloatArray:
    """Penalizes stops and long flights, more so for families and short trips."""
    choice = settings.choice
    kids = read_integers(rows, "n_children") > 0
    short = read_integers(rows, "nights") <= choice.short_trip_nights
    stops = read_floats(rows, "number_of_stops")
    kids_factor = 1.0 + choice.kids_stops_factor * kids
    short_factor = 1.0 + choice.short_trip_stops_factor * short
    hours = read_floats(rows, "flight_duration_hours")
    excess = np.maximum(0.0, hours - choice.duration_free_hours)
    penalty = choice.stops_coefficient * stops * kids_factor * short_factor
    return -penalty - choice.duration_coefficient * excess
