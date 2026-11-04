"""Loads the saved dataset columns and distinct texts needed for features."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.catalog import TextCatalog, build_text_catalog, locate
from src.settings.feature_models import FeatureSettings
from src.storage import (
    DATASET_FILE,
    HOTEL_TEXT_FILE,
    INTENT_TEXT_FILE,
    ROOM_TEXT_FILE,
    TRUTH_FILE,
)
from src.world.helpers import read_integers
from src.world.models import FloatArray

DEPENDENCY_COLUMNS: list[str] = ["hotel_id", "competitor_retail_price"]
TRUTH_COLUMNS: list[str] = ["session_id", "hid_fit", "hid_intent_register"]


@dataclass
class FeatureInputs:
    dataset: pd.DataFrame
    catalog: TextCatalog


def dataset_columns(settings: FeatureSettings) -> list[str]:
    """Lists the dataset columns that the feature stage reads."""
    columns = settings.columns
    names = settings.id_columns + columns.categorical + columns.numeric
    names = names + DEPENDENCY_COLUMNS
    return list(dict.fromkeys(names))


def count_words(texts: list[str]) -> FloatArray:
    """Counts the words of every text."""
    return np.array([len(text.split()) for text in texts], dtype=np.float64)


def load_feature_inputs(directory: Path, settings: FeatureSettings) -> FeatureInputs:
    """Reads the needed dataset columns and builds the distinct-text catalog."""
    catalog = build_text_catalog(
        pd.read_parquet(directory / INTENT_TEXT_FILE),
        pd.read_parquet(directory / HOTEL_TEXT_FILE),
        pd.read_parquet(directory / ROOM_TEXT_FILE),
    )
    columns = dataset_columns(settings)
    dataset = pd.read_parquet(directory / DATASET_FILE, columns=columns)
    sessions = locate(catalog.intent_ids, read_integers(dataset, "session_id"))
    words = count_words(catalog.intent_texts)
    dataset["intent_word_count"] = words[sessions]
    return FeatureInputs(dataset=dataset, catalog=catalog)


def load_truth_for_scores(directory: Path) -> pd.DataFrame:
    """Reads the hidden columns used to score the text features."""
    return pd.read_parquet(directory / TRUTH_FILE, columns=TRUTH_COLUMNS)
