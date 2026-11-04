"""Builds the destination reference table from the configuration."""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.settings.records import DestinationRecord
from src.world.columns import DESTINATION_PREFIX, destination_columns
from src.world.models import FloatArray


def build_destination_row(
    index: int,
    record: DestinationRecord,
    aspect_names: list[str],
) -> dict[str, float | str | bool]:
    """Converts one destination record into a flat table row."""
    row: dict[str, float | str | bool] = {
        "destination_index": index,
        "city": record.city,
        "country": record.country,
        "latitude": record.latitude,
        "longitude": record.longitude,
        "price_level": record.price_level,
        "competition": record.competition,
        "climate_offset": record.climate_offset,
        "ski": record.ski,
    }
    for name in aspect_names:
        row[f"{DESTINATION_PREFIX}{name}"] = record.aspect_levels[name]
    return row


def check_destination_levels(
    record: DestinationRecord,
    aspect_names: list[str],
) -> None:
    """Requires a destination to define exactly the configured aspects."""
    if set(record.aspect_levels) != set(aspect_names):
        raise ValueError(f"aspect levels of {record.city} do not match the aspects")


def build_destination_table(settings: Settings) -> pd.DataFrame:
    """Builds one row per destination with its aspect levels."""
    rows: list[dict[str, float | str | bool]] = []
    names = settings.catalogue.aspects.names
    for record in settings.destinations:
        check_destination_levels(record, names)
    for index, record in enumerate(settings.destinations):
        rows.append(build_destination_row(index, record, names))
    return pd.DataFrame(rows)


def read_destination_levels(
    destination: pd.Series,
    aspect_names: list[str],
) -> FloatArray:
    """Returns the aspect levels of one destination row in aspect order."""
    columns = destination_columns(aspect_names)
    return destination[columns].to_numpy(dtype=np.float64)
