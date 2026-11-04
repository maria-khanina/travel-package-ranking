"""Samples where simulated travelers fly from and which device they use."""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.world.helpers import read_floats, read_integers, read_texts
from src.world.models import FloatArray, IntArray, TextArray


def build_hub_table(settings: Settings) -> pd.DataFrame:
    """Lists every departure hub together with its country and weight."""
    rows: list[dict[str, float | str]] = []
    for country, origin in settings.origins.items():
        for hub in origin.hubs:
            rows.append(
                {
                    "country": country,
                    "hub": hub.name,
                    "latitude": hub.latitude,
                    "longitude": hub.longitude,
                    "weight": hub.weight,
                }
            )
    hubs = pd.DataFrame(rows)
    hubs.insert(0, "hub_id", np.arange(len(hubs)))
    return hubs


def sample_countries(
    random_generator: np.random.Generator,
    settings: Settings,
    count: int,
) -> TextArray:
    """Draws the departure country of every session."""
    weights: list[float] = []
    for origin in settings.origins.values():
        weights.append(origin.weight)
    probabilities = np.array(weights) / sum(weights)
    names = np.array(list(settings.origins), dtype=np.object_)
    picked = random_generator.choice(len(names), size=count, p=probabilities)
    return names[picked]


def sample_hubs(
    random_generator: np.random.Generator,
    countries: TextArray,
    hubs: pd.DataFrame,
) -> IntArray:
    """Draws a departure hub inside the country of every session."""
    hub_identifiers = read_integers(hubs, "hub_id")
    weights = read_floats(hubs, "weight")
    hub_countries = read_texts(hubs, "country")
    picked_hubs = np.zeros(len(countries), dtype=np.int64)
    for country in np.unique(countries):
        rows = np.flatnonzero(countries == country)
        in_country = hub_countries == country
        probabilities = weights[in_country] / weights[in_country].sum()
        picked_hubs[rows] = random_generator.choice(
            hub_identifiers[in_country],
            size=rows.size,
            p=probabilities,
        )
    return picked_hubs


def sample_devices(
    random_generator: np.random.Generator,
    settings: Settings,
    count: int,
) -> TextArray:
    """Draws the device used in every session."""
    sessions = settings.sessions
    names = np.array(sessions.device_names, dtype=np.object_)
    picked = random_generator.choice(
        len(names),
        size=count,
        p=sessions.device_probabilities,
    )
    return names[picked]


def read_country_attribute(
    countries: TextArray,
    settings: Settings,
    attribute: str,
) -> FloatArray:
    """Looks up a numeric origin attribute for every session country."""
    values: list[float] = []
    for country in countries:
        values.append(float(getattr(settings.origins[str(country)], attribute)))
    return np.array(values)
