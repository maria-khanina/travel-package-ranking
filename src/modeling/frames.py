"""Selects the model inputs of one experiment from the feature table."""

import pandas as pd

from src.features.relative import RANK_SUFFIX
from src.settings.feature_models import FeatureColumns
from src.settings.modeling_models import ExperimentRecord

HOTEL_FEATURE = "hotel_id"


def keep_columns(columns: list[str], dropped: list[str]) -> list[str]:
    """Removes the dropped columns from a list, keeping the order."""
    return [column for column in columns if column not in dropped]


def prefixed_names(table_columns: list[str], prefixes: list[str]) -> list[str]:
    """Lists the table columns that start with any of the prefixes."""
    return [name for name in table_columns if name.startswith(tuple(prefixes))]


def relative_names(columns: list[str]) -> list[str]:
    """Lists the rank column names of the given base columns."""
    return [f"{column}{RANK_SUFFIX}" for column in columns]


def select_columns(
    table_columns: list[str],
    columns: FeatureColumns,
    experiment: ExperimentRecord,
) -> tuple[list[str], list[str]]:
    """Lists the model columns and the categorical ones of an experiment."""
    dropped = experiment.dropped_columns
    categorical = keep_columns(columns.categorical, dropped)
    numeric = keep_columns(columns.numeric + columns.derived, dropped)
    extra = prefixed_names(table_columns, experiment.extra_prefixes)
    relative = relative_names(experiment.relative_columns)
    names = categorical + numeric + experiment.text_columns + extra + relative
    return names, categorical


def build_frame(
    table: pd.DataFrame,
    hotels: pd.Series,
    columns: FeatureColumns,
    experiment: ExperimentRecord,
) -> tuple[pd.DataFrame, list[str]]:
    """Builds the input frame of an experiment and its categorical names."""
    names, categorical = select_columns(list(table.columns), columns, experiment)
    frame = table[names].copy()
    if experiment.include_hotel_id:
        frame[HOTEL_FEATURE] = hotels.to_numpy()
        categorical = categorical + [HOTEL_FEATURE]
    return frame, categorical
