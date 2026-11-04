"""Samples traveler archetypes, hidden wish weights and budget levels."""

import numpy as np

from src.settings.loader import Settings
from src.settings.records import ArchetypeRecord
from src.world.helpers import build_aspect_vector, normalize_rows, sample_categorical
from src.world.models import BoolArray, FloatArray, IntArray, TextArray


def build_context_flags(
    children: IntArray,
    adults: IntArray,
    month: IntArray,
    settings: Settings,
) -> dict[str, BoolArray]:
    """Describes each session by the flags that archetype multipliers use."""
    kids = children > 0
    winter = np.isin(month, settings.seasons.snow_months)
    return {
        "kids": kids,
        "no_kids": ~kids,
        "couple": (adults == settings.sessions.couple_adults) & ~kids,
        "group": adults >= settings.sessions.group_minimum_adults,
        "winter": winter,
        "off_winter": ~winter,
        "summer": np.isin(month, settings.seasons.summer_months),
    }


def compute_archetype_weight(
    record: ArchetypeRecord,
    flags: dict[str, BoolArray],
    count: int,
) -> FloatArray:
    """Multiplies the base weight of an archetype by every active flag."""
    weight = np.ones(count)
    for flag, factor in record.multipliers.items():
        weight = np.where(flags[flag], weight * factor, weight)
    return weight


def compute_archetype_probabilities(
    flags: dict[str, BoolArray],
    settings: Settings,
) -> FloatArray:
    """Gives every session a probability for each traveler archetype."""
    count = len(flags["kids"])
    columns: list[FloatArray] = []
    for record in settings.archetypes.values():
        columns.append(compute_archetype_weight(record, flags, count))
    return normalize_rows(np.stack(columns, axis=1))


def sample_archetypes(
    random_generator: np.random.Generator,
    flags: dict[str, BoolArray],
    settings: Settings,
) -> IntArray:
    """Draws the archetype index of every session."""
    probabilities = compute_archetype_probabilities(flags, settings)
    return sample_categorical(random_generator, probabilities)


def build_wish_means(settings: Settings) -> FloatArray:
    """Stacks the mean wish vector of every archetype."""
    names = settings.catalogue.aspects.names
    rows: list[FloatArray] = []
    for record in settings.archetypes.values():
        rows.append(build_aspect_vector(record.wishes, names))
    return np.stack(rows)


def add_extra_wishes(
    random_generator: np.random.Generator,
    wishes: FloatArray,
    settings: Settings,
) -> None:
    """Gives some travelers one additional random wish."""
    sessions = settings.sessions
    count = int(wishes.shape[0])
    aspect_count = int(wishes.shape[1])
    aspect = random_generator.integers(0, aspect_count, size=count, dtype=np.int64)
    draws = np.asarray(random_generator.random(count), dtype=np.float64)
    wants_extra = draws < sessions.extra_wish_probability
    unset = wishes[np.arange(count), aspect] == 0
    rows = np.flatnonzero(wants_extra & unset)
    wishes[rows, aspect[rows]] = random_generator.uniform(
        sessions.extra_wish_low,
        sessions.extra_wish_high,
        size=rows.size,
    )


def add_family_wishes(
    random_generator: np.random.Generator,
    wishes: FloatArray,
    children: IntArray,
    settings: Settings,
) -> None:
    """Makes travelers with children care about family friendliness."""
    sessions = settings.sessions
    aspects = settings.catalogue.aspects
    rows = np.flatnonzero(children > 0)
    family = aspects.names.index(aspects.family)
    boost = random_generator.uniform(
        sessions.kid_family_wish_low,
        sessions.kid_family_wish_high,
        size=rows.size,
    )
    wishes[rows, family] = np.maximum(wishes[rows, family], boost)


def restore_empty_wishes(wishes: FloatArray, means: FloatArray) -> None:
    """Gives travelers without any wish the mean wishes of their archetype."""
    empty = np.flatnonzero(np.abs(wishes).sum(axis=1) == 0)
    wishes[empty] = means[empty]


def sample_wishes(
    random_generator: np.random.Generator,
    archetype_index: IntArray,
    children: IntArray,
    settings: Settings,
) -> FloatArray:
    """Draws the hidden wish weight of every traveler for every aspect."""
    sessions = settings.sessions
    means = build_wish_means(settings)[archetype_index]
    spread = sessions.wish_noise_standard_deviation
    noise = random_generator.normal(0.0, spread, means.shape)
    wishes = np.clip(means + noise, -1.0, 1.0)
    wishes[np.abs(wishes) < sessions.wish_sparsify_threshold] = 0.0
    add_extra_wishes(random_generator, wishes, settings)
    add_family_wishes(random_generator, wishes, children, settings)
    restore_empty_wishes(wishes, means)
    return wishes


def sample_budget_multiplier(
    random_generator: np.random.Generator,
    archetype_names: TextArray,
    settings: Settings,
) -> FloatArray:
    """Draws how much more or less than average a traveler can spend."""
    shifts: list[float] = []
    for name in archetype_names:
        shifts.append(settings.archetypes[str(name)].budget_shift)
    spread = settings.sessions.budget_noise_standard_deviation
    noise = random_generator.normal(0.0, spread, len(shifts))
    return np.exp(np.array(shifts) + noise)
