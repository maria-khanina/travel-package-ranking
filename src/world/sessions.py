"""Builds the table of simulated search sessions with hidden traveler traits."""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.world.columns import wish_columns
from src.world.filters import sample_filters
from src.world.helpers import read_integers, read_texts
from src.world.models import FloatArray, IntArray, TextArray
from src.world.origins import (
    read_country_attribute,
    sample_countries,
    sample_devices,
    sample_hubs,
)
from src.world.party import (
    compute_departure_month,
    sample_departure_days,
    sample_nights,
    sample_party,
    sample_search_times,
)
from src.world.wishes import (
    build_context_flags,
    sample_archetypes,
    sample_budget_multiplier,
    sample_wishes,
)


def build_context_table(
    random_generator: np.random.Generator,
    hubs: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Samples the observable context of every session."""
    sessions = settings.sessions
    count = settings.run.session_count
    countries = sample_countries(random_generator, settings, count)
    hour, weekday = sample_search_times(random_generator, sessions, count)
    days_ahead = sample_departure_days(random_generator, sessions, count)
    adults, children = sample_party(random_generator, sessions, count)
    year_days = sum(sessions.month_lengths)
    search_day = random_generator.integers(1, year_days + 1, size=count)
    return pd.DataFrame(
        {
            "session_id": np.arange(count),
            "user_country": countries,
            "hub_id": sample_hubs(random_generator, countries, hubs),
            "user_device": sample_devices(random_generator, settings, count),
            "search_hour": hour,
            "search_dow": weekday,
            "days_to_departure": days_ahead,
            "nights": sample_nights(random_generator, sessions, count),
            "n_adults": adults,
            "n_children": children,
            "departure_month": compute_departure_month(
                search_day,
                days_ahead,
                sessions.month_lengths,
            ),
        }
    )


def compute_alpha_shift(
    devices: TextArray,
    countries: TextArray,
    settings: Settings,
) -> FloatArray:
    """Combines device and country effects on price sensitivity."""
    device_shift: list[float] = []
    for device in devices:
        device_shift.append(settings.sessions.device_alpha_shift[str(device)])
    country_shift = read_country_attribute(countries, settings, "alpha")
    return np.array(device_shift) + country_shift


def sample_format_labels(
    random_generator: np.random.Generator,
    settings: Settings,
    count: int,
) -> TextArray:
    """Draws the message format each traveler will use."""
    options = settings.text_styles.intent_formats
    labels = np.array([option.label for option in options], dtype=np.object_)
    weights = np.array([option.probability for option in options])
    picked = random_generator.choice(len(labels), size=count, p=weights / weights.sum())
    return labels[picked]


def build_hidden_table(
    random_generator: np.random.Generator,
    context: pd.DataFrame,
    archetype_index: IntArray,
    wishes: FloatArray,
    settings: Settings,
) -> pd.DataFrame:
    """Samples the hidden traits of every session that drive its choices."""
    count = len(context)
    countries = read_texts(context, "user_country")
    names = np.array(list(settings.archetypes), dtype=np.object_)[archetype_index]
    budget = sample_budget_multiplier(random_generator, names, settings)
    country_budget = read_country_attribute(countries, settings, "budget")
    hidden = pd.DataFrame(
        {
            "hid_archetype": names,
            "hid_z": random_generator.normal(size=count),
            "hid_alpha_shift": compute_alpha_shift(
                read_texts(context, "user_device"),
                countries,
                settings,
            ),
            "hid_budget_mult": budget,
            "hid_budget_scale": country_budget * budget,
            "hid_text_style": sample_format_labels(random_generator, settings, count),
            "hid_out_noise_std": random_generator.normal(size=count),
            "hid_gumbel_out": random_generator.gumbel(size=count),
        }
    )
    columns = wish_columns(settings.catalogue.aspects.names)
    return pd.concat([hidden, pd.DataFrame(wishes, columns=columns)], axis=1)


def build_session_table(
    random_generator: np.random.Generator,
    destinations: pd.DataFrame,
    hubs: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Builds the full session table with filters and hidden traits."""
    context = build_context_table(random_generator, hubs, settings)
    children = read_integers(context, "n_children")
    flags = build_context_flags(
        children,
        read_integers(context, "n_adults"),
        read_integers(context, "departure_month"),
        settings,
    )
    archetype_index = sample_archetypes(random_generator, flags, settings)
    wishes = sample_wishes(random_generator, archetype_index, children, settings)
    filters = sample_filters(random_generator, wishes, destinations, settings)
    hidden = build_hidden_table(
        random_generator,
        context,
        archetype_index,
        wishes,
        settings,
    )
    return pd.concat([context, filters, hidden], axis=1)
