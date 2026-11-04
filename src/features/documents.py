"""Builds the texts that a query is compared with, and what each one is about."""

from dataclasses import dataclass

import pandas as pd

from src.features.catalog import TextCatalog, locate
from src.settings.feature_models import EmbeddingVariant
from src.world.helpers import read_integers, read_strings
from src.world.models import IntArray

HOTEL = "hotel"
PLACE = "place"
ROOM = "room"
HOTEL_PLACE = "hotel_place"
HOTEL_PLACE_ROOM = "hotel_place_room"
TEXT_VARIANTS: list[str] = [HOTEL, PLACE, ROOM, HOTEL_PLACE_ROOM, HOTEL_PLACE]


@dataclass
class Documents:
    texts: list[str]
    positions: IntArray


def lookup_texts(ids: IntArray, catalog_ids: IntArray, texts: list[str]) -> list[str]:
    """Finds the text of every identifier in the catalog."""
    positions: list[int] = locate(catalog_ids, ids).tolist()
    return [texts[at] for at in positions]


def make_places(table: pd.DataFrame) -> list[str]:
    """Writes the city and country of every hotel as one string."""
    cities, countries = read_strings(table, "city"), read_strings(table, "country")
    return [f"{city}, {country}" for city, country in zip(cities, countries)]


def make_stays(texts: list[str], places: list[str], head: str) -> list[str]:
    """Appends the location of the hotel to its description."""
    return [f"{text}{head} {place}" for text, place in zip(texts, places)]


def build_hotel_documents(
    rows: pd.DataFrame,
    catalog: TextCatalog,
    head: str,
) -> tuple[IntArray, list[str], dict[str, Documents]]:
    """Builds the hotel, place and hotel-with-place documents."""
    table = rows.drop_duplicates("hotel_id").sort_values("hotel_id")
    ids = read_integers(table, "hotel_id")
    texts = lookup_texts(ids, catalog.hotel_ids, catalog.hotel_texts)
    places = make_places(table)
    stays = make_stays(texts, places, head)
    where = locate(ids, read_integers(rows, "hotel_id"))
    documents = {
        HOTEL: Documents(texts, where),
        PLACE: Documents(places, where),
        HOTEL_PLACE: Documents(stays, where),
    }
    return ids, stays, documents


def build_room_documents(
    rows: pd.DataFrame,
    catalog: TextCatalog,
    hotel_part: tuple[IntArray, list[str]],
    separator: str,
) -> dict[str, Documents]:
    """Builds the room and the hotel-with-place-and-room documents."""
    hotel_ids, stays = hotel_part
    table = rows.drop_duplicates("room_id").sort_values("room_id")
    ids = read_integers(table, "room_id")
    texts = lookup_texts(ids, catalog.room_ids, catalog.room_texts)
    owners: list[int] = locate(hotel_ids, read_integers(table, "hotel_id")).tolist()
    full = [f"{stays[owner]}{separator}{text}" for owner, text in zip(owners, texts)]
    where = locate(ids, read_integers(rows, "room_id"))
    return {ROOM: Documents(texts, where), HOTEL_PLACE_ROOM: Documents(full, where)}


def build_documents(
    rows: pd.DataFrame,
    catalog: TextCatalog,
    separator: str,
    prefix: str,
) -> dict[str, Documents]:
    """Builds the five document variants of the hotels and rooms in the rows."""
    head = separator + prefix
    hotel_ids, stays, documents = build_hotel_documents(rows, catalog, head)
    documents.update(
        build_room_documents(rows, catalog, (hotel_ids, stays), separator)
    )
    return documents


def task_for(variant: EmbeddingVariant, text_variant: str) -> str:
    """Chooses the instruction that describes what is retrieved."""
    tasks = {
        HOTEL: variant.hotel_task,
        PLACE: variant.place_task,
        ROOM: variant.room_task,
        HOTEL_PLACE: variant.hotel_task,
        HOTEL_PLACE_ROOM: variant.combined_task,
    }
    return tasks[text_variant]
