"""Samples the timing, length and party composition of every trip."""

import numpy as np

from src.settings.models import SessionSettings
from src.world.columns import HOURS_PER_DAY
from src.world.models import FloatArray, IntArray


def circular_bump(hours: IntArray, center: float, width: float) -> FloatArray:
    """Evaluates a daily-cycle bump that wraps around midnight."""
    gap = np.abs(hours - center)
    distance = np.minimum(gap, HOURS_PER_DAY - gap)
    return np.exp(-(distance**2) / (2 * width**2))


def compute_hour_probabilities(sessions: SessionSettings) -> FloatArray:
    """Gives the probability of a search happening in each hour."""
    hours = np.arange(HOURS_PER_DAY)
    evening = circular_bump(hours, sessions.evening_center, sessions.evening_width)
    noon = circular_bump(hours, sessions.noon_center, sessions.noon_width)
    weights = sessions.hour_floor + evening + sessions.noon_weight * noon
    return weights / weights.sum()


def sample_search_times(
    random_generator: np.random.Generator,
    sessions: SessionSettings,
    count: int,
) -> tuple[IntArray, IntArray]:
    """Draws the hour of day and the weekday of every search."""
    hour = random_generator.choice(
        HOURS_PER_DAY,
        size=count,
        p=compute_hour_probabilities(sessions),
    )
    weekday_weights = np.array(sessions.weekday_weights)
    weekday = random_generator.choice(
        len(weekday_weights),
        size=count,
        p=weekday_weights / weekday_weights.sum(),
    )
    return hour, weekday


def sample_departure_days(
    random_generator: np.random.Generator,
    sessions: SessionSettings,
    count: int,
) -> IntArray:
    """Draws how many days ahead the trip starts."""
    component = random_generator.choice(
        len(sessions.departure_component_probabilities),
        size=count,
        p=sessions.departure_component_probabilities,
    )
    lower = np.array(sessions.departure_lower_days)[component]
    upper = np.array(sessions.departure_upper_days)[component]
    return random_generator.integers(lower, upper + 1)


def sample_nights(
    random_generator: np.random.Generator,
    sessions: SessionSettings,
    count: int,
) -> IntArray:
    """Draws the number of nights of every trip."""
    return random_generator.choice(
        sessions.nights_values,
        size=count,
        p=sessions.nights_probabilities,
    )


def sample_party(
    random_generator: np.random.Generator,
    sessions: SessionSettings,
    count: int,
) -> tuple[IntArray, IntArray]:
    """Draws the number of adults and children travelling together."""
    adults = random_generator.choice(
        sessions.adults_values,
        size=count,
        p=sessions.adults_probabilities,
    )
    kids_probability = np.where(
        adults >= sessions.couple_adults,
        sessions.kids_probability_with_partner,
        sessions.kids_probability_single,
    )
    has_kids = random_generator.random(count) < kids_probability
    kid_counts = random_generator.choice(
        sessions.kid_count_values,
        size=count,
        p=sessions.kid_count_probabilities,
    )
    return adults, np.where(has_kids, kid_counts, 0)


def compute_departure_month(
    search_day: IntArray,
    days_ahead: IntArray,
    month_lengths: list[int],
) -> IntArray:
    """Converts a search day and a lead time into the departure month."""
    month_ends = np.cumsum(np.array(month_lengths, dtype=np.int64))
    departure_day = (search_day - 1 + days_ahead) % month_ends[-1] + 1
    passed = departure_day.reshape(-1, 1) > month_ends.reshape(1, -1)
    return passed.sum(axis=1, dtype=np.int64) + 1
