"""Combines the query-label similarities with the labels a package has."""

import numpy as np
import pandas as pd

from src.behavior.taste_terms import encode_feature
from src.features.extras import LABEL_PREFIX
from src.settings.feature_models import FeatureSettings
from src.world.helpers import read_floats
from src.world.models import VectorArray

LABEL_MATCH = "label_match"


def standardize(values: np.ndarray) -> np.ndarray:
    """Centers values and divides them by their spread."""
    return (values - values.mean()) / values.std()


def build_label_match(
    table: pd.DataFrame,
    dataset: pd.DataFrame,
    settings: FeatureSettings,
) -> VectorArray:
    """Sums, over labels, how much the query talks about labels the package has."""
    total = np.zeros(len(table), dtype=np.float64)
    for name, record in settings.label_characteristics.items():
        similarity = read_floats(table, f"{LABEL_PREFIX}{name}")
        total += standardize(similarity) * encode_feature(dataset, record)
    return np.asarray(total, dtype=np.float32)
