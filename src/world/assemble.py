"""Assembles the row-level world table of packages shown in each session."""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.world.columns import MEAL_COST_COLUMN, WISH_PREFIX
from src.world.helpers import read_floats, read_integers
from src.world.margins import assign_margins
from src.world.meals import compute_meal_cost
from src.world.models import PackageTable, PoolSelection, SeasonTables
from src.world.pricing import (
    assign_boards,
    assign_flights,
    assign_free_cancellation,
    assign_prices,
)


def expand_sessions(sessions: pd.DataFrame, pool_size: int) -> pd.DataFrame:
    """Repeats every session row once per package shown in it."""
    repeated = np.repeat(np.arange(len(sessions)), pool_size)
    return sessions.iloc[repeated].reset_index(drop=True)


def attach_package_columns(
    rows: pd.DataFrame,
    packages: PackageTable,
    selection: PoolSelection,
) -> None:
    """Copies the package attributes into the row-level table."""
    flat = selection.package_index.reshape(-1)
    columns: dict[str, np.ndarray] = {
        "room_id": packages.room_id,
        "hotel_id": packages.hotel_id,
        "hotel_name": packages.hotel_name,
        "country": packages.country,
        "city": packages.city,
        "stars": packages.stars,
        "rating": packages.rating,
        "room_category": packages.category,
        "room_view": packages.view,
        "hid_destination": packages.destination,
        "hid_hotel_effect": packages.hotel_effect,
        "hid_price_noise": packages.price_noise,
        "hid_room_multiplier": packages.room_multiplier,
    }
    for name, values in columns.items():
        rows[name] = values[flat]
    for name in packages.attributes.columns:
        rows[name] = packages.attributes[name].to_numpy()[flat]
    rows["hid_fit"] = selection.fit.reshape(-1)
    hubs = read_integers(rows, "hub_id")
    rows["hid_route_km"] = selection.route_distance[hubs, flat]


def attach_stay_columns(
    random_generator: np.random.Generator,
    rows: pd.DataFrame,
    packages: PackageTable,
    selection: PoolSelection,
    settings: Settings,
) -> None:
    """Adds the board, cancellation and flight columns."""
    flat = selection.package_index.reshape(-1)
    assign_boards(random_generator, rows, packages.boards[flat], settings.pricing)
    assign_free_cancellation(random_generator, rows, settings.pricing)
    assign_flights(random_generator, rows, settings.pricing)


def attach_margins(
    random_generator: np.random.Generator,
    rows: pd.DataFrame,
    destinations: pd.DataFrame,
    settings: Settings,
) -> None:
    """Adds the logged margin and its propensity to every row."""
    destination = read_integers(rows, "hid_destination")
    price_level = read_floats(destinations, "price_level")[destination]
    assign_margins(random_generator, rows, price_level, settings)


def attach_noise_columns(
    random_generator: np.random.Generator,
    rows: pd.DataFrame,
    settings: Settings,
) -> None:
    """Adds the fixed random draws used by the booking simulation."""
    aspects = settings.catalogue.aspects
    rows["hid_luxury_wish"] = rows[f"{WISH_PREFIX}{aspects.luxury}"]
    rows["hid_eps_std"] = random_generator.normal(size=len(rows))
    rows["hid_gumbel"] = random_generator.gumbel(size=len(rows))


def build_rows(
    random_generator: np.random.Generator,
    sessions: pd.DataFrame,
    packages: PackageTable,
    selection: PoolSelection,
    destinations: pd.DataFrame,
    tables: SeasonTables,
    settings: Settings,
) -> pd.DataFrame:
    """Builds the full table with one row per package shown in a session."""
    rows = expand_sessions(sessions, settings.pool.size)
    attach_package_columns(rows, packages, selection)
    attach_stay_columns(random_generator, rows, packages, selection, settings)
    assign_prices(random_generator, rows, destinations, tables.price, settings)
    rows[MEAL_COST_COLUMN] = compute_meal_cost(rows, destinations, settings)
    attach_margins(random_generator, rows, destinations, settings)
    attach_noise_columns(random_generator, rows, settings)
    return rows
