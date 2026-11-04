"""Samples the hidden tastes of every session for concrete package features."""

import numpy as np
import pandas as pd

from src.settings.amenity_models import TasteRecord
from src.settings.loader import Settings
from src.world.columns import TASTE_PREFIX
from src.world.helpers import read_integers, read_texts
from src.world.models import BoolArray, FloatArray
from src.world.wishes import build_context_flags


def compute_taste_means(
    record: TasteRecord,
    archetypes: list[str],
    kids: BoolArray,
    couples: BoolArray,
) -> FloatArray:
    """Combines the base, archetype and party effects on one taste weight."""
    shifts = np.array([record.archetype_shifts.get(name, 0.0) for name in archetypes])
    party = record.kids_shift * kids + record.couple_shift * couples
    return np.asarray(record.mean + shifts + party, dtype=np.float64)


def sample_tastes(
    random_generator: np.random.Generator,
    sessions: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Draws the weight each session puts on every concrete feature."""
    flags = build_context_flags(
        read_integers(sessions, "n_children"),
        read_integers(sessions, "n_adults"),
        read_integers(sessions, "departure_month"),
        settings,
    )
    archetypes = [str(name) for name in read_texts(sessions, "hid_archetype")]
    columns: dict[str, FloatArray] = {}
    for record in settings.amenities.tastes:
        means = compute_taste_means(record, archetypes, flags["kids"], flags["couple"])
        noise = random_generator.normal(0.0, record.standard_deviation, len(sessions))
        columns[TASTE_PREFIX + record.name] = means + noise
    return pd.DataFrame(columns, index=sessions.index)
