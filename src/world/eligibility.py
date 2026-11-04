"""Decides which packages satisfy the hard filters of a search session."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.world.helpers import read_floats, read_texts
from src.world.models import BoolArray, FloatArray, IntArray, PackageTable


@dataclass
class SessionFilter:
    country: str | None
    minimum_stars: float
    maximum_stops: float
    board: str | None


def clean_text(value: str | float | None) -> str | None:
    """Converts a possibly missing table value into text or None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value)


def build_session_filters(sessions: pd.DataFrame) -> list[SessionFilter]:
    """Reads the hard filters of every session from the session table."""
    countries = read_texts(sessions, "req_country")
    boards = read_texts(sessions, "req_board")
    minimum_stars = read_floats(sessions, "req_min_stars")
    maximum_stops = read_floats(sessions, "req_max_stops")
    filters: list[SessionFilter] = []
    for position in range(len(sessions)):
        filters.append(
            SessionFilter(
                country=clean_text(countries[position]),
                minimum_stars=float(minimum_stars[position]),
                maximum_stops=float(maximum_stops[position]),
                board=clean_text(boards[position]),
            )
        )
    return filters


def build_country_mask(
    packages: PackageTable,
    route_distance: FloatArray,
    session_filter: SessionFilter,
    settings: Settings,
) -> BoolArray:
    """Keeps packages located in the requested country."""
    if session_filter.country is None:
        return np.ones(len(packages.stars), dtype=np.bool_)
    return packages.country_code == packages.country_codes[session_filter.country]


def build_stars_mask(
    packages: PackageTable,
    route_distance: FloatArray,
    session_filter: SessionFilter,
    settings: Settings,
) -> BoolArray:
    """Keeps packages with at least the requested star rating."""
    if np.isnan(session_filter.minimum_stars):
        return np.ones(len(packages.stars), dtype=np.bool_)
    return packages.stars >= session_filter.minimum_stars


def build_stops_mask(
    packages: PackageTable,
    route_distance: FloatArray,
    session_filter: SessionFilter,
    settings: Settings,
) -> BoolArray:
    """Keeps routes short enough to be flown directly when that is required."""
    stops = session_filter.maximum_stops
    if np.isnan(stops) or stops > 0:
        return np.ones(len(packages.stars), dtype=np.bool_)
    return route_distance <= settings.pool.direct_flight_maximum_kilometers


def build_board_mask(
    packages: PackageTable,
    route_distance: FloatArray,
    session_filter: SessionFilter,
    settings: Settings,
) -> BoolArray:
    """Keeps hotels that offer the requested board type."""
    if session_filter.board is None:
        return np.ones(len(packages.stars), dtype=np.bool_)
    board_index = settings.pricing.board_names.index(session_filter.board)
    return packages.boards[:, board_index]


MaskBuilder = Callable[
    [PackageTable, FloatArray, SessionFilter, Settings],
    BoolArray,
]
MASK_BUILDERS: dict[str, MaskBuilder] = {
    "req_country": build_country_mask,
    "req_min_stars": build_stars_mask,
    "req_max_stops": build_stops_mask,
    "req_board": build_board_mask,
}


def build_eligibility_mask(
    packages: PackageTable,
    route_distance: FloatArray,
    session_filter: SessionFilter,
    applied_count: int,
    settings: Settings,
) -> BoolArray:
    """Combines the first applied_count filters into one package mask."""
    mask = np.ones(len(packages.stars), dtype=np.bool_)
    for name in settings.pool.filter_order[:applied_count]:
        builder = MASK_BUILDERS[name]
        mask &= builder(packages, route_distance, session_filter, settings)
    return mask


def is_pool_sufficient(
    eligible: IntArray,
    packages: PackageTable,
    settings: Settings,
) -> bool:
    """Checks that enough packages and hotels remain after filtering."""
    pool = settings.pool
    hotels = np.unique(packages.hotel_id[eligible]).size
    enough_packages = eligible.size >= pool.minimum_eligible_packages
    return enough_packages and hotels >= pool.minimum_eligible_hotels


def find_eligible_packages(
    packages: PackageTable,
    route_distance: FloatArray,
    session_filter: SessionFilter,
    settings: Settings,
) -> tuple[int, IntArray]:
    """Finds eligible packages, dropping the last filters if too few remain."""
    applied_count = len(settings.pool.filter_order)
    while applied_count > 0:
        mask = build_eligibility_mask(
            packages,
            route_distance,
            session_filter,
            applied_count,
            settings,
        )
        eligible = np.flatnonzero(mask)
        if is_pool_sufficient(eligible, packages, settings):
            return applied_count, eligible
        applied_count -= 1
    return 0, np.arange(len(packages.stars))
