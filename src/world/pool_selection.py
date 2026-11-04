"""Selects the pool of candidate packages shown in every session."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.settings.models import PoolSettings
from src.world.columns import MONTHS_PER_YEAR, wish_columns
from src.world.eligibility import (
    SessionFilter,
    build_session_filters,
    find_eligible_packages,
)
from src.world.helpers import read_integers
from src.world.models import (
    FloatArray,
    IntArray,
    PackageTable,
    PoolSelection,
    SeasonTables,
)


@dataclass
class SessionInputs:
    wishes: FloatArray
    wish_totals: FloatArray
    months: IntArray
    hub_identifiers: IntArray
    filters: list[SessionFilter]


@dataclass
class PoolContext:
    inputs: SessionInputs
    packages: PackageTable
    route_distances: FloatArray
    tables: SeasonTables
    settings: Settings


def build_session_inputs(sessions: pd.DataFrame, settings: Settings) -> SessionInputs:
    """Extracts the session columns needed to select pools."""
    names = settings.catalogue.aspects.names
    wishes = sessions[wish_columns(names)].to_numpy(dtype=np.float64)
    return SessionInputs(
        wishes=wishes,
        wish_totals=np.abs(wishes).sum(axis=1),
        months=read_integers(sessions, "departure_month"),
        hub_identifiers=read_integers(sessions, "hub_id"),
        filters=build_session_filters(sessions),
    )


def take_from_tier(
    random_generator: np.random.Generator,
    candidates: IntArray,
    quota: int,
    hotel_of: IntArray,
    hotel_counts: dict[int, int],
    chosen: list[int],
    cap: int,
) -> None:
    """Randomly takes up to quota packages while capping packages per hotel."""
    taken = 0
    for candidate in random_generator.permutation(candidates):
        if taken >= quota:
            break
        hotel = int(hotel_of[candidate])
        if hotel_counts.get(hotel, 0) >= cap:
            continue
        hotel_counts[hotel] = hotel_counts.get(hotel, 0) + 1
        chosen.append(int(candidate))
        taken += 1


def fill_remaining(
    random_generator: np.random.Generator,
    ranked: IntArray,
    hotel_of: IntArray,
    hotel_counts: dict[int, int],
    chosen: list[int],
    pool: PoolSettings,
) -> None:
    """Completes a short pool, first with the cap and then without it."""
    caps: list[int] = [pool.maximum_packages_per_hotel, pool.size]
    for cap in caps:
        if len(chosen) >= pool.size:
            break
        rest = np.setdiff1d(ranked, np.array(chosen, dtype=np.int64))
        missing = pool.size - len(chosen)
        take_from_tier(
            random_generator,
            rest,
            missing,
            hotel_of,
            hotel_counts,
            chosen,
            cap,
        )


def pick_pool(
    random_generator: np.random.Generator,
    ranked: IntArray,
    hotel_of: IntArray,
    pool: PoolSettings,
) -> list[int]:
    """Picks the pool by sampling a fixed quota from each relevance tier."""
    chosen: list[int] = []
    hotel_counts: dict[int, int] = {}
    cap = pool.maximum_packages_per_hotel
    tiers = np.array_split(ranked, len(pool.tier_quotas))
    for tier, quota in zip(tiers, pool.tier_quotas):
        take_from_tier(
            random_generator,
            tier,
            quota,
            hotel_of,
            hotel_counts,
            chosen,
            cap,
        )
    fill_remaining(random_generator, ranked, hotel_of, hotel_counts, chosen, pool)
    return chosen


def select_for_session(
    random_generator: np.random.Generator,
    fit_row: FloatArray,
    session_row: int,
    context: PoolContext,
) -> tuple[int, list[int]]:
    """Selects the pool of one session and reports how many filters applied."""
    pool = context.settings.pool
    hub = context.inputs.hub_identifiers[session_row]
    route_distance = context.route_distances[hub]
    applied_count, eligible = find_eligible_packages(
        context.packages,
        route_distance,
        context.inputs.filters[session_row],
        context.settings,
    )
    spread = pool.fit_noise_standard_deviation
    noise = random_generator.normal(0.0, spread, eligible.size)
    penalty = pool.distance_penalty_per_kilometers * route_distance[eligible]
    ranked = eligible[np.argsort(-(fit_row[eligible] + noise - penalty))]
    hotel_of = context.packages.hotel_id
    return applied_count, pick_pool(random_generator, ranked, hotel_of, pool)


def compute_month_levels(month: int, context: PoolContext) -> FloatArray:
    """Scales package aspect levels by the weather and snow of a month."""
    aspects = context.settings.catalogue.aspects
    packages = context.packages
    levels = packages.levels.copy()
    beach = aspects.names.index(aspects.beach)
    mountains = aspects.names.index(aspects.mountains)
    levels[:, beach] *= context.tables.beach[packages.destination, month]
    levels[:, mountains] *= context.tables.mountain[packages.destination, month]
    return levels


def select_month(
    random_generator: np.random.Generator,
    month: int,
    context: PoolContext,
    selection: PoolSelection,
) -> None:
    """Selects the pools of all sessions departing in one month."""
    inputs = context.inputs
    rows = np.flatnonzero(inputs.months == month)
    levels = compute_month_levels(month, context)
    totals = inputs.wish_totals[rows].reshape(-1, 1)
    fit_block = (inputs.wishes[rows] @ levels.T) / totals
    for position, session_row in enumerate(rows):
        applied_count, chosen = select_for_session(
            random_generator,
            fit_block[position],
            int(session_row),
            context,
        )
        selection.applied_filters[session_row] = applied_count
        selection.package_index[session_row] = chosen
        selection.fit[session_row] = fit_block[position, chosen]


def shuffle_pools(
    random_generator: np.random.Generator,
    selection: PoolSelection,
) -> None:
    """Shuffles the order of packages inside every pool."""
    keys = random_generator.random(selection.package_index.shape)
    order = np.argsort(keys, axis=1)
    selection.package_index = np.take_along_axis(selection.package_index, order, axis=1)
    selection.fit = np.take_along_axis(selection.fit, order, axis=1)


def select_pools(
    random_generator: np.random.Generator,
    context: PoolContext,
) -> PoolSelection:
    """Selects the candidate pool of every session."""
    shape = (len(context.inputs.months), context.settings.pool.size)
    selection = PoolSelection(
        package_index=np.zeros(shape, dtype=np.int64),
        fit=np.zeros(shape),
        route_distance=context.route_distances,
        applied_filters=np.zeros(shape[0], dtype=np.int64),
    )
    for month in range(1, MONTHS_PER_YEAR + 1):
        select_month(random_generator, month, context, selection)
    shuffle_pools(random_generator, selection)
    return selection
