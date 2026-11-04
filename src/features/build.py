"""Builds and saves the complete feature table of the dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.inputs import FeatureInputs
from src.features.label_match import LABEL_MATCH, build_label_match
from src.features.relative import build_relative_features
from src.features.tabular import build_tabular_features
from src.features.text_features import build_text_features
from src.settings.feature_models import FeatureSettings

FEATURE_FILE = "features.parquet"


def shrink_floats(table: pd.DataFrame) -> pd.DataFrame:
    """Stores floating point columns with single precision to save memory."""
    for column in table.select_dtypes(include="float64").columns:
        table[column] = table[column].astype(np.float32)
    return table


def build_feature_table(
    inputs: FeatureInputs,
    settings: FeatureSettings,
    directory: Path,
) -> pd.DataFrame:
    """Joins ids, tabular features and text features into one table."""
    dataset = inputs.dataset
    identifiers = dataset[settings.id_columns]
    tabular = build_tabular_features(dataset, settings.columns)
    texts = build_text_features(dataset, inputs.catalog, settings, directory)
    table = pd.concat([identifiers, tabular, texts], axis=1)
    table[LABEL_MATCH] = build_label_match(table, dataset, settings)
    pool = settings.evaluation_pool_size
    relative = build_relative_features(table, settings.relative_columns, pool)
    return shrink_floats(pd.concat([table, relative], axis=1))


def save_feature_table(table: pd.DataFrame, directory: Path) -> Path:
    """Writes the feature table next to the cached vectors."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / FEATURE_FILE
    table.to_parquet(path, index=False)
    return path
