"""Utility from the match between traveler tastes and package features."""

import numpy as np
import pandas as pd

from src.settings.amenity_models import CharacteristicRecord
from src.settings.loader import Settings
from src.world.columns import TASTE_PREFIX
from src.world.helpers import read_floats, read_texts
from src.world.models import FloatArray


def encode_feature(rows: pd.DataFrame, record: CharacteristicRecord) -> FloatArray:
    """Turns the package feature behind a taste into a number from 0 to 1."""
    if record.category is not None:
        matches = read_texts(rows, record.column) == record.category
        return np.asarray(matches, dtype=np.float64)
    span = record.high - record.low
    scaled = np.clip((read_floats(rows, record.column) - record.low) / span, 0.0, 1.0)
    return 1.0 - scaled if record.invert else scaled


def compute_taste_term(rows: pd.DataFrame, settings: Settings) -> FloatArray:
    """Sums the taste weight times the feature value over all tastes."""
    total = np.zeros(len(rows))
    for record in settings.amenities.tastes:
        weight = read_floats(rows, TASTE_PREFIX + record.name)
        total = total + weight * encode_feature(rows, record)
    return total
