"""Generates the room types offered by every hotel."""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.settings.records import RoomCategoryRecord, RoomSettings, RoomViewRecord
from src.world.columns import LEVEL_PREFIX, SHIFT_PREFIX
from src.world.helpers import build_aspect_vector, normalize, sample_index
from src.world.models import FloatArray

RoomRow = dict[str, float | str]


def read_hotel_level(hotel: pd.Series, aspect: str) -> float:
    """Reads the level of one aspect from a hotel row."""
    return float(hotel[f"{LEVEL_PREFIX}{aspect}"])


def apply_gate(
    weight: float,
    gate_aspect: str | None,
    gate_level: float,
    weight_when_unmet: float,
    hotel: pd.Series,
) -> float:
    """Reduces a weight when the hotel lacks the aspect a room option needs."""
    if gate_aspect is None:
        return weight
    if read_hotel_level(hotel, gate_aspect) >= gate_level:
        return weight
    return weight_when_unmet


def compute_category_weight(
    category: RoomCategoryRecord,
    hotel: pd.Series,
) -> float:
    """Computes how likely a hotel is to offer a room category."""
    types = category.hotel_types
    type_allowed = types is None or hotel["hotel_type"] in types
    if hotel["stars"] < category.minimum_stars or not type_allowed:
        return 0.0
    return apply_gate(
        category.weight,
        category.gate_aspect,
        category.gate_level,
        category.weight_when_gate_unmet,
        hotel,
    )


def compute_view_weight(view: RoomViewRecord, hotel: pd.Series) -> float:
    """Computes how likely a hotel is to offer a room view."""
    return apply_gate(
        view.weight,
        view.gate_aspect,
        view.gate_level,
        view.weight_when_gate_unmet,
        hotel,
    )


def compute_category_probabilities(
    rooms: RoomSettings,
    hotel: pd.Series,
) -> FloatArray:
    """Gives the probability of every room category for a hotel."""
    weights: list[float] = []
    for category in rooms.categories.values():
        weights.append(compute_category_weight(category, hotel))
    return normalize(np.array(weights))


def compute_view_probabilities(
    rooms: RoomSettings,
    hotel: pd.Series,
) -> FloatArray:
    """Gives the probability of every room view for a hotel."""
    weights: list[float] = []
    for view in rooms.views.values():
        weights.append(compute_view_weight(view, hotel))
    return normalize(np.array(weights))


def draw_room_option(
    random_generator: np.random.Generator,
    rooms: RoomSettings,
    category_probabilities: FloatArray,
    view_probabilities: FloatArray,
) -> tuple[str, str]:
    """Draws one room category together with one room view."""
    categories: list[str] = list(rooms.categories)
    views: list[str] = list(rooms.views)
    category = categories[sample_index(random_generator, category_probabilities)]
    view = views[sample_index(random_generator, view_probabilities)]
    return category, view


def draw_room_options(
    random_generator: np.random.Generator,
    rooms: RoomSettings,
    hotel: pd.Series,
) -> list[tuple[str, str]]:
    """Draws the distinct category-and-view combinations of one hotel."""
    count = int(
        random_generator.choice(rooms.count_values, p=rooms.count_probabilities)
    )
    category_probabilities = compute_category_probabilities(rooms, hotel)
    view_probabilities = compute_view_probabilities(rooms, hotel)
    chosen: set[tuple[str, str]] = set()
    for _ in range(rooms.maximum_tries):
        if len(chosen) >= count:
            break
        chosen.add(
            draw_room_option(
                random_generator,
                rooms,
                category_probabilities,
                view_probabilities,
            )
        )
    return sorted(chosen)


def build_room_row(
    hotel_id: int,
    category: str,
    view: str,
    settings: Settings,
) -> RoomRow:
    """Builds one room row with its price multiplier and aspect shifts."""
    names = settings.catalogue.aspects.names
    category_record = settings.rooms.categories[category]
    view_record = settings.rooms.views[view]
    shift = build_aspect_vector(category_record.shift, names)
    shift = shift + build_aspect_vector(view_record.shift, names)
    multiplier = category_record.price_multiplier * view_record.price_multiplier
    row: RoomRow = {
        "hotel_id": hotel_id,
        "room_category": category,
        "room_view": view,
        "price_multiplier": multiplier,
    }
    for name, value in zip(names, shift):
        row[f"{SHIFT_PREFIX}{name}"] = float(value)
    return row


def build_room_table(
    random_generator: np.random.Generator,
    hotels: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Generates the rooms of every hotel as one table."""
    rows: list[RoomRow] = []
    for _, hotel in hotels.iterrows():
        hotel_id = int(hotel["hotel_id"])
        options = draw_room_options(random_generator, settings.rooms, hotel)
        for category, view in options:
            rows.append(build_room_row(hotel_id, category, view, settings))
    rooms = pd.DataFrame(rows)
    rooms.insert(0, "room_id", np.arange(len(rooms)))
    return rooms
