"""Collects the distinct texts of the dataset and maps rows to them."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.world.helpers import read_integers
from src.world.models import IntArray


@dataclass
class TextCatalog:
    intent_ids: IntArray
    intent_texts: list[str]
    hotel_ids: IntArray
    hotel_texts: list[str]
    room_ids: IntArray
    room_texts: list[str]


def collect_distinct(
    table: pd.DataFrame,
    id_column: str,
    text_column: str,
) -> tuple[IntArray, list[str]]:
    """Lists each identifier once, with its text, sorted by identifier."""
    distinct = table.drop_duplicates(id_column).sort_values(id_column)
    texts = [str(value) for value in distinct[text_column].tolist()]
    return read_integers(distinct, id_column), texts


def build_text_catalog(
    intents: pd.DataFrame,
    hotels: pd.DataFrame,
    rooms: pd.DataFrame,
) -> TextCatalog:
    """Collects the distinct intents, hotel texts and room texts."""
    intent_ids, intent_texts = collect_distinct(intents, "session_id", "user_intent")
    hotel_ids, hotel_texts = collect_distinct(hotels, "hotel_id", "hotel_description")
    room_ids, room_texts = collect_distinct(rooms, "room_id", "room_description")
    return TextCatalog(
        intent_ids=intent_ids,
        intent_texts=intent_texts,
        hotel_ids=hotel_ids,
        hotel_texts=hotel_texts,
        room_ids=room_ids,
        room_texts=room_texts,
    )


def locate(known_ids: IntArray, row_ids: IntArray) -> IntArray:
    """Finds the position of every row identifier in the sorted known ids."""
    positions = np.searchsorted(known_ids, row_ids)
    return np.asarray(positions, dtype=np.int64)


def fingerprint(texts: list[str], length: int) -> str:
    """Hashes a list of texts into a short stable string."""
    digest = hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest()
    return digest[:length]


def cache_file(
    directory: Path,
    repository: str,
    texts: list[str],
    length: int,
) -> Path:
    """Names the cache file of the vectors of a model and a list of texts."""
    model = repository.replace("/", "__")
    return directory / f"vectors_{model}_{fingerprint(texts, length)}.npy"
