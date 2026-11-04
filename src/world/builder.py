"""Builds the whole simulated world: catalogue, sessions, pools and prices."""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.world.amenities import add_hotel_amenities, add_room_features
from src.world.assemble import build_rows
from src.world.destinations import build_destination_table
from src.world.hotels import build_hotel_table
from src.world.models import CatalogueTables, IntArray, PoolSelection, WorldTables
from src.world.origins import build_hub_table
from src.world.packages import build_package_table, build_route_distances
from src.world.pool_selection import PoolContext, build_session_inputs, select_pools
from src.world.rooms import build_room_table
from src.world.seasons import build_season_tables
from src.world.sessions import build_session_table
from src.world.tastes import sample_tastes


def build_catalogue(
    random_generator: np.random.Generator,
    feature_generator: np.random.Generator,
    settings: Settings,
) -> CatalogueTables:
    """Generates the destinations, hotels and rooms available for booking."""
    amenities = settings.amenities
    destinations = build_destination_table(settings)
    hotels = build_hotel_table(random_generator, destinations, settings)
    rooms = build_room_table(random_generator, hotels, settings)
    hotels = add_hotel_amenities(feature_generator, hotels, amenities)
    rooms = add_room_features(feature_generator, rooms, amenities)
    return CatalogueTables(destinations=destinations, hotels=hotels, rooms=rooms)


def clear_dropped_filters(
    sessions: pd.DataFrame,
    applied_filters: IntArray,
    filter_order: list[str],
) -> None:
    """Blanks the filters that had to be dropped to keep enough packages."""
    for position, column in enumerate(filter_order):
        sessions.loc[applied_filters <= position, column] = None


def build_pool_context(
    catalogue: CatalogueTables,
    sessions: pd.DataFrame,
    hubs: pd.DataFrame,
    settings: Settings,
) -> PoolContext:
    """Prepares everything needed to select the pool of every session."""
    packages = build_package_table(catalogue.hotels, catalogue.rooms, settings)
    return PoolContext(
        inputs=build_session_inputs(sessions, settings),
        packages=packages,
        route_distances=build_route_distances(packages, catalogue.destinations, hubs),
        tables=build_season_tables(
            catalogue.destinations,
            settings.seasons,
            settings.catalogue.aspects,
        ),
        settings=settings,
    )


def select_session_pools(
    random_generator: np.random.Generator,
    context: PoolContext,
    sessions: pd.DataFrame,
) -> PoolSelection:
    """Chooses the pool of every session and blanks the dropped filters."""
    selection = select_pools(random_generator, context)
    filter_order = context.settings.pool.filter_order
    clear_dropped_filters(sessions, selection.applied_filters, filter_order)
    return selection


def build_feature_generator(seed: int, settings: Settings) -> np.random.Generator:
    """Creates the random stream used by amenities and traveler tastes."""
    return np.random.default_rng([seed, settings.amenities.random_stream])


def attach_tastes(
    feature_generator: np.random.Generator,
    sessions: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Adds the hidden taste weights to the session table."""
    tastes = sample_tastes(feature_generator, sessions, settings)
    return pd.concat([sessions, tastes], axis=1)


def build_world(settings: Settings, seed: int) -> WorldTables:
    """Builds the complete synthetic world for one random seed."""
    random_generator = np.random.default_rng(seed)
    feature_generator = build_feature_generator(seed, settings)
    catalogue = build_catalogue(random_generator, feature_generator, settings)
    hubs = build_hub_table(settings)
    sessions = build_session_table(
        random_generator, catalogue.destinations, hubs, settings
    )
    context = build_pool_context(catalogue, sessions, hubs, settings)
    selection = select_session_pools(random_generator, context, sessions)
    sessions = attach_tastes(feature_generator, sessions, settings)
    rows = build_rows(
        random_generator,
        sessions,
        context.packages,
        selection,
        catalogue.destinations,
        context.tables,
        settings,
    )
    return WorldTables(
        destinations=catalogue.destinations,
        hotels=catalogue.hotels,
        rooms=catalogue.rooms,
        sessions=sessions,
        rows=rows,
    )
