"""Flattens the catalogue into arrays describing every bookable package."""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.world.amenities import hotel_attribute_columns, room_attribute_columns
from src.world.columns import board_columns, level_columns, shift_columns
from src.world.flights import compute_distance_kilometers
from src.world.helpers import read_floats, read_integers, read_texts
from src.world.models import FloatArray, IntArray, PackageTable, TextArray


def encode_countries(countries: TextArray, codes: dict[str, int]) -> IntArray:
    """Replaces every country name by its integer code."""
    values: list[int] = [codes[str(country)] for country in countries]
    return np.array(values, dtype=np.int64)


def build_country_codes(countries: TextArray) -> dict[str, int]:
    """Assigns an integer code to every distinct country name."""
    codes: dict[str, int] = {}
    for code, country in enumerate(sorted(set(countries.tolist()))):
        codes[country] = code
    return codes


def build_attribute_table(
    hotels: pd.DataFrame,
    rooms: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Collects the amenity and room feature columns of every package."""
    amenities = settings.amenities
    hotel_part = hotels[hotel_attribute_columns(amenities)]
    hotel_part = hotel_part.iloc[read_integers(rooms, "hotel_id")]
    room_part = rooms[room_attribute_columns(amenities)]
    return pd.concat(
        [hotel_part.reset_index(drop=True), room_part.reset_index(drop=True)],
        axis=1,
    )


def build_package_table(
    hotels: pd.DataFrame,
    rooms: pd.DataFrame,
    settings: Settings,
) -> PackageTable:
    """Joins hotels and rooms into one array-based table of packages."""
    names = settings.catalogue.aspects.names
    hotel_of = read_integers(rooms, "hotel_id")
    hotel_levels = hotels[level_columns(names)].to_numpy(dtype=np.float64)[hotel_of]
    room_shift = rooms[shift_columns(names)].to_numpy(dtype=np.float64)
    boards = hotels[board_columns(settings.pricing.board_names)]
    countries = read_texts(hotels, "country")[hotel_of]
    codes = build_country_codes(countries)
    return PackageTable(
        attributes=build_attribute_table(hotels, rooms, settings),
        room_id=read_integers(rooms, "room_id"),
        hotel_id=hotel_of,
        destination=read_integers(hotels, "destination_index")[hotel_of],
        country_code=encode_countries(countries, codes),
        country_codes=codes,
        stars=read_integers(hotels, "stars")[hotel_of],
        rating=read_floats(hotels, "rating")[hotel_of],
        hotel_effect=read_floats(hotels, "hotel_effect")[hotel_of],
        price_noise=read_floats(hotels, "price_noise")[hotel_of],
        boards=boards.to_numpy(dtype=np.bool_)[hotel_of],
        levels=np.clip(hotel_levels + room_shift, 0.0, 1.0),
        room_multiplier=read_floats(rooms, "price_multiplier"),
        category=read_texts(rooms, "room_category"),
        view=read_texts(rooms, "room_view"),
        city=read_texts(hotels, "city")[hotel_of],
        country=countries,
        hotel_name=read_texts(hotels, "hotel_name")[hotel_of],
    )


def build_route_distances(
    packages: PackageTable,
    destinations: pd.DataFrame,
    hubs: pd.DataFrame,
) -> FloatArray:
    """Computes the flight distance from every hub to every package."""
    latitude = read_floats(destinations, "latitude")[packages.destination]
    longitude = read_floats(destinations, "longitude")[packages.destination]
    rows: list[FloatArray] = []
    for hub_latitude, hub_longitude in zip(hubs["latitude"], hubs["longitude"]):
        rows.append(
            compute_distance_kilometers(
                float(hub_latitude),
                float(hub_longitude),
                latitude,
                longitude,
            )
        )
    return np.stack(rows)
