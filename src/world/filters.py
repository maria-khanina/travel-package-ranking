"""Samples the hard filters that travelers apply before seeing packages."""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.settings.models import SessionSettings
from src.world.columns import destination_columns
from src.world.helpers import normalize_rows, read_texts, sample_categorical
from src.world.models import BoolArray, FloatArray, TextArray


def compute_country_scores(
    wishes: FloatArray,
    destinations: pd.DataFrame,
    countries: list[str],
    aspect_names: list[str],
) -> FloatArray:
    """Scores how well the best destination of each country fits the wishes."""
    levels = destinations[destination_columns(aspect_names)].to_numpy(dtype=np.float64)
    totals = np.abs(wishes).sum(axis=1, keepdims=True)
    fit = (wishes @ levels.T) / totals
    destination_countries = read_texts(destinations, "country")
    columns: list[FloatArray] = []
    for country in countries:
        columns.append(fit[:, destination_countries == country].max(axis=1))
    return np.stack(columns, axis=1)


def sample_country_filter(
    random_generator: np.random.Generator,
    wishes: FloatArray,
    destinations: pd.DataFrame,
    settings: Settings,
) -> TextArray:
    """Draws the country each traveler would restrict the search to."""
    countries = sorted(set(read_texts(destinations, "country").tolist()))
    scores = compute_country_scores(
        wishes,
        destinations,
        countries,
        settings.catalogue.aspects.names,
    )
    temperature = settings.sessions.country_filter_temperature
    weights = np.exp(temperature * (scores - scores.max(axis=1, keepdims=True)))
    picked = sample_categorical(random_generator, normalize_rows(weights))
    return np.array(countries, dtype=np.object_)[picked]


def draw_keep_mask(
    random_generator: np.random.Generator,
    count: int,
    probability: float,
) -> BoolArray:
    """Decides which sessions apply a filter."""
    return random_generator.random(count) < probability


def keep_text_values(values: TextArray, keep: BoolArray) -> TextArray:
    """Blanks the text values of sessions that do not apply the filter."""
    result = values.copy()
    result[~keep] = None
    return result


def keep_numeric_values(values: FloatArray, keep: BoolArray) -> FloatArray:
    """Blanks the numeric values of sessions that do not apply the filter."""
    return np.where(keep, values, np.nan)


def draw_filter_masks(
    random_generator: np.random.Generator,
    count: int,
    sessions: SessionSettings,
) -> tuple[BoolArray, BoolArray, BoolArray, BoolArray]:
    """Decides which sessions apply the country, star, stop and board filters."""
    country = draw_keep_mask(
        random_generator,
        count,
        sessions.country_filter_probability,
    )
    stars = draw_keep_mask(
        random_generator,
        count,
        sessions.minimum_stars_filter_probability,
    )
    stops = draw_keep_mask(
        random_generator,
        count,
        sessions.maximum_stops_filter_probability,
    )
    board = draw_keep_mask(
        random_generator,
        count,
        sessions.board_filter_probability,
    )
    return country, stars, stops, board


def sample_filters(
    random_generator: np.random.Generator,
    wishes: FloatArray,
    destinations: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Draws the optional country, star, stop and board filters."""
    sessions = settings.sessions
    count = len(wishes)
    stars = random_generator.choice(
        sessions.minimum_stars_values,
        size=count,
        p=sessions.minimum_stars_probabilities,
    )
    stops = random_generator.choice(
        sessions.maximum_stops_values,
        size=count,
        p=sessions.maximum_stops_probabilities,
    )
    boards = random_generator.choice(
        np.array(sessions.board_filter_values, dtype=np.object_),
        size=count,
        p=sessions.board_filter_probabilities,
    )
    countries = sample_country_filter(random_generator, wishes, destinations, settings)
    country_keep, stars_keep, stops_keep, board_keep = draw_filter_masks(
        random_generator,
        count,
        sessions,
    )
    return pd.DataFrame(
        {
            "req_country": keep_text_values(countries, country_keep),
            "req_min_stars": keep_numeric_values(stars.astype(np.float64), stars_keep),
            "req_max_stops": keep_numeric_values(stops.astype(np.float64), stops_keep),
            "req_board": keep_text_values(boards, board_keep),
        }
    )
