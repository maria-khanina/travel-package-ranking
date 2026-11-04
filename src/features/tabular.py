"""Builds the tabular model inputs from the public dataset columns."""

import numpy as np
import pandas as pd

from src.settings.feature_models import FeatureColumns
from src.world.helpers import read_floats
from src.world.models import FloatArray


def compute_travelers(dataset: pd.DataFrame) -> FloatArray:
    """Counts the people travelling together."""
    return read_floats(dataset, "n_adults") + read_floats(dataset, "n_children")


def compute_price_features(
    dataset: pd.DataFrame,
    margin: FloatArray,
) -> dict[str, FloatArray]:
    """Computes the features that change when the margin changes."""
    net_cost = read_floats(dataset, "net_cost")
    selling = net_cost * (1.0 + margin)
    nights = read_floats(dataset, "nights")
    competitor = read_floats(dataset, "competitor_retail_price")
    return {
        "price_ratio_vs_competitor": selling / competitor,
        "selling_price": selling,
        "margin_dollars": net_cost * margin,
        "price_per_person_night": selling / (compute_travelers(dataset) * nights),
    }


def compute_derived_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """Computes the price and query features that need several columns."""
    margin = read_floats(dataset, "margin_percentage")
    prices = compute_price_features(dataset, margin)
    prices["intent_word_count"] = read_floats(dataset, "intent_word_count")
    return pd.DataFrame(prices, index=dataset.index)


def build_tabular_features(
    dataset: pd.DataFrame,
    columns: FeatureColumns,
) -> pd.DataFrame:
    """Selects the plain columns and appends the derived ones."""
    plain = dataset[columns.categorical + columns.numeric].copy()
    for column in plain.select_dtypes(include="bool").columns:
        plain[column] = plain[column].astype(np.int64)
    derived = compute_derived_features(dataset)
    return pd.concat([plain, derived[columns.derived]], axis=1)
