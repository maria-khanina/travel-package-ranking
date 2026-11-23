"""Checks of the hidden demand shock, its margins and the propensity weights."""

import copy

import numpy as np
from tests.scenario_settings import (
    PROBABILITY_SUM_TOLERANCE,
    SHOCK_SHARE_TOLERANCE,
    WEIGHT_MEAN_TOLERANCE,
)

from src.behavior.utility import compute_utilities
from src.margin.repricing import all_candidates
from src.modeling.weights import compute_propensity_weights
from src.settings.loader import Settings
from src.world.confounder import (
    apply_confounder,
    compute_shocked_target,
    draw_demand_shock,
)
from src.world.helpers import read_floats, read_integers
from src.world.margins import compute_margin_probabilities, compute_target_margin
from src.world.models import WorldTables


def build_shocked_world(world: WorldTables, settings: Settings) -> WorldTables:
    """Copies the small world and applies the hidden demand shock to it."""
    shocked = copy.deepcopy(world)
    apply_confounder(shocked, settings)
    return shocked


def test_shock_covers_the_configured_share(
    world: WorldTables,
    settings: Settings,
) -> None:
    generator = np.random.default_rng(settings.confounder.seed)
    shock = draw_demand_shock(generator, world.rows, settings)
    gap = abs(shock.mean() - settings.confounder.shock_share)
    assert gap < SHOCK_SHARE_TOLERANCE


def test_shock_is_shared_by_a_session_and_destination(
    world: WorldTables,
    settings: Settings,
) -> None:
    shocked = build_shocked_world(world, settings)
    rows = shocked.rows
    pairs = rows.groupby(["session_id", "hid_destination"])
    spread = pairs[settings.confounder.shock_column].nunique()
    assert int(spread.max()) == 1


def test_shock_raises_the_target_only_where_present(
    world: WorldTables,
    settings: Settings,
) -> None:
    rows = world.rows
    shock = np.zeros(len(rows))
    shock[::2] = 1.0
    raised = compute_shocked_target(rows, world.destinations, shock, settings)
    destination = read_integers(rows, "hid_destination")
    price_level = read_floats(world.destinations, "price_level")[destination]
    base = compute_target_margin(rows, price_level, settings)
    expected = base + settings.confounder.margin_shift * shock
    assert np.allclose(raised, expected)


def test_margin_probabilities_sum_to_one(settings: Settings) -> None:
    grid = np.array(settings.run.margin_grid, dtype=np.float64)
    count = len(grid) * len(grid)
    targets = np.linspace(float(grid.min()), float(grid.max()), count)
    logging = settings.pricing.logging_policy
    probabilities = compute_margin_probabilities(targets, grid, logging)
    gap = np.abs(probabilities.sum(axis=1) - 1.0).max()
    assert gap < PROBABILITY_SUM_TOLERANCE


def test_shock_adds_utility_only_when_enabled(
    world: WorldTables,
    settings: Settings,
) -> None:
    shocked = build_shocked_world(world, settings)
    rows = shocked.rows
    margin = read_floats(rows, "margin_percentage")
    enabled = settings.model_copy(deep=True)
    enabled.confounder.enabled = True
    plain = compute_utilities(rows, settings, margin, False)
    lifted = compute_utilities(rows, enabled, margin, False)
    shock = read_floats(rows, settings.confounder.shock_column)
    assert np.allclose(lifted - plain, settings.confounder.utility_shift * shock)


def test_stabilized_weights_average_one(
    world: WorldTables,
    settings: Settings,
) -> None:
    shocked = build_shocked_world(world, settings)
    weights = compute_propensity_weights(
        read_floats(shocked.rows, "margin_percentage"),
        read_floats(shocked.rows, "margin_propensity"),
    )
    assert abs(weights.mean() - 1.0) < WEIGHT_MEAN_TOLERANCE


def test_repricing_keeps_every_candidate(settings: Settings) -> None:
    pool = settings.pool.size
    shown = all_candidates(2, pool)
    assert shown.shape == (2, pool)
    assert (shown == np.arange(pool)).all()
