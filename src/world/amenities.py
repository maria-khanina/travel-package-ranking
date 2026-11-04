"""Adds amenities and distances to hotels and features to rooms."""

import numpy as np
import pandas as pd

from src.settings.amenity_models import (
    AmenitySettings,
    DistanceRecord,
    FlagRecord,
)
from src.world.columns import LEVEL_PREFIX
from src.world.helpers import read_floats, read_integers, read_texts
from src.world.models import BoolArray, FloatArray


def clip_probability(values: FloatArray, amenities: AmenitySettings) -> FloatArray:
    """Keeps probabilities away from certainty so that no flag is constant."""
    low = amenities.minimum_probability
    return np.clip(values, low, amenities.maximum_probability)


def compute_flag_probability(
    record: FlagRecord,
    hotels: pd.DataFrame,
    amenities: AmenitySettings,
) -> FloatArray:
    """Computes how likely every hotel is to have one amenity."""
    stars = read_integers(hotels, "stars") - amenities.star_reference
    probability = record.base + record.star_weight * stars
    for aspect, weight in record.aspect_weights.items():
        probability = probability + weight * read_floats(hotels, LEVEL_PREFIX + aspect)
    types = read_texts(hotels, "hotel_type")
    for hotel_type, boost in record.type_boosts.items():
        probability = probability + boost * (types == hotel_type)
    return clip_probability(np.asarray(probability, dtype=np.float64), amenities)


def sample_flag(
    random_generator: np.random.Generator,
    record: FlagRecord,
    hotels: pd.DataFrame,
    amenities: AmenitySettings,
) -> BoolArray:
    """Draws whether every hotel has one amenity."""
    probability = compute_flag_probability(record, hotels, amenities)
    draws = np.asarray(random_generator.random(len(hotels)), dtype=np.float64)
    return draws < probability


def read_type_factors(hotels: pd.DataFrame, record: DistanceRecord) -> FloatArray:
    """Gives each hotel the distance multiplier of its hotel type."""
    types = read_texts(hotels, "hotel_type")
    return np.array([record.type_factors.get(str(name), 1.0) for name in types])


def sample_distance(
    random_generator: np.random.Generator,
    record: DistanceRecord,
    hotels: pd.DataFrame,
    amenities: AmenitySettings,
) -> FloatArray:
    """Draws the distance in meters from every hotel to a point of interest."""
    level = read_floats(hotels, LEVEL_PREFIX + record.aspect)
    log_far = np.log(record.far_meters)
    log_near = np.log(record.near_meters)
    noise = random_generator.normal(0.0, record.noise_standard_deviation, len(hotels))
    log_meters = log_far + (log_near - log_far) * level + noise
    meters: FloatArray = np.exp(log_meters) * read_type_factors(hotels, record)
    meters = np.clip(meters, record.near_meters, record.far_meters)
    step = amenities.rounding_meters
    rounded: FloatArray = np.round(meters / step) * step
    return rounded


def add_hotel_amenities(
    random_generator: np.random.Generator,
    hotels: pd.DataFrame,
    amenities: AmenitySettings,
) -> pd.DataFrame:
    """Appends the amenity flags and distance columns to the hotel table."""
    columns: dict[str, FloatArray | BoolArray] = {}
    for flag in amenities.flags:
        columns[flag.column] = sample_flag(random_generator, flag, hotels, amenities)
    for distance in amenities.distances:
        columns[distance.column] = sample_distance(
            random_generator,
            distance,
            hotels,
            amenities,
        )
    return pd.concat([hotels, pd.DataFrame(columns, index=hotels.index)], axis=1)


def compute_balcony_probability(
    rooms: pd.DataFrame,
    amenities: AmenitySettings,
) -> FloatArray:
    """Computes how likely every room is to have a balcony."""
    features = amenities.rooms
    categories = read_texts(rooms, "room_category")
    views = read_texts(rooms, "room_view")
    base = np.array([features.balcony_by_category[str(name)] for name in categories])
    bonus = np.array([features.balcony_view_bonus[str(name)] for name in views])
    return clip_probability(base + bonus, amenities)


def add_room_features(
    random_generator: np.random.Generator,
    rooms: pd.DataFrame,
    amenities: AmenitySettings,
) -> pd.DataFrame:
    """Appends the size and balcony columns to the room table."""
    features = amenities.rooms
    categories = read_texts(rooms, "room_category")
    base = np.array([features.size_by_category[str(name)] for name in categories])
    spread = features.size_noise_standard_deviation
    sizes = base * np.exp(random_generator.normal(0.0, spread, len(rooms)))
    draws = np.asarray(random_generator.random(len(rooms)), dtype=np.float64)
    columns = {
        features.size_column: np.round(sizes),
        features.balcony_column: draws < compute_balcony_probability(rooms, amenities),
    }
    return pd.concat([rooms, pd.DataFrame(columns, index=rooms.index)], axis=1)


def hotel_attribute_columns(amenities: AmenitySettings) -> list[str]:
    """Lists the amenity columns that belong to hotels."""
    flags = [record.column for record in amenities.flags]
    return flags + [record.column for record in amenities.distances]


def room_attribute_columns(amenities: AmenitySettings) -> list[str]:
    """Lists the feature columns that belong to rooms."""
    return [amenities.rooms.size_column, amenities.rooms.balcony_column]
