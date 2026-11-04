"""Checks of amenities, room features, tastes and meal costs."""

import numpy as np

from src.dataset import DatasetBundle
from src.settings.loader import Settings
from src.texts.details import pick_hotel_facts, pick_query_details
from src.world.columns import MEAL_COST_COLUMN, TASTE_PREFIX
from src.world.helpers import read_floats, read_integers
from src.world.models import WorldTables

DRAW_COUNT = 50
MANY_DRAWS = 200
KID_TASTE_GAP = 0.5
DISTRACTOR_COLUMNS: list[str] = [
    "free_wifi",
    "has_gym",
    "has_parking",
    "pets_allowed",
]


def test_flags_vary_across_hotels(world: WorldTables, settings: Settings) -> None:
    for record in settings.amenities.flags:
        share = float(world.hotels[record.column].mean())
        assert 0.0 < share < 1.0


def test_distances_stay_within_bounds(world: WorldTables, settings: Settings) -> None:
    for record in settings.amenities.distances:
        meters = read_floats(world.hotels, record.column)
        assert meters.min() >= record.near_meters
        assert meters.max() <= record.far_meters


def test_larger_categories_have_larger_rooms(
    world: WorldTables,
    settings: Settings,
) -> None:
    column = settings.amenities.rooms.size_column
    means = world.rooms.groupby("room_category")[column].mean()
    sizes = settings.amenities.rooms.size_by_category
    ordered = sorted(sizes, key=lambda name: sizes[name])
    observed = [float(means[name]) for name in ordered if name in means.index]
    assert observed == sorted(observed)


def test_every_session_has_every_taste(world: WorldTables, settings: Settings) -> None:
    for record in settings.amenities.tastes:
        assert TASTE_PREFIX + record.name in world.sessions.columns


def test_children_raise_the_kids_club_taste(
    world: WorldTables,
    settings: Settings,
) -> None:
    weights = read_floats(world.sessions, TASTE_PREFIX + "kids_club")
    kids = read_integers(world.sessions, "n_children") > 0
    assert weights[kids].mean() > weights[~kids].mean() + KID_TASTE_GAP


def test_distractors_are_not_tastes(settings: Settings) -> None:
    columns = {record.column for record in settings.amenities.tastes}
    assert columns.isdisjoint(DISTRACTOR_COLUMNS)


def test_meal_cost_depends_on_board(world: WorldTables) -> None:
    rows = world.rows
    cost = read_floats(rows, MEAL_COST_COLUMN)
    boards = rows["board_type"].to_numpy()
    assert (cost[boards == "AI"] == 0.0).all()
    assert (cost[boards == "RO"] > 0.0).all()


def test_public_table_hides_tastes(bundle: DatasetBundle, settings: Settings) -> None:
    public = bundle.public
    assert not any(column.startswith(TASTE_PREFIX) for column in public.columns)
    for record in settings.amenities.flags:
        assert record.column in public.columns
    assert np.isfinite(read_floats(public, "beach_distance_m")).all()


def test_hotel_facts_state_only_real_amenities(
    world: WorldTables,
    settings: Settings,
) -> None:
    hotel = world.hotels.iloc[0].copy()
    for record in settings.amenities.mentions.hotel_facts:
        hotel[record.column] = False
    generator = np.random.default_rng(0)
    mentions = settings.amenities.mentions
    facts = pick_hotel_facts(generator, hotel, mentions)
    phrases = [text for record in mentions.distance_facts for text in record.phrases]
    assert set(facts) <= set(phrases)


def test_query_details_respect_the_limit(settings: Settings) -> None:
    tastes = np.full(len(settings.amenities.tastes), 2.0)
    generator = np.random.default_rng(1)
    mentions = settings.amenities.mentions
    for _ in range(DRAW_COUNT):
        details = pick_query_details(generator, tastes, settings, 1.0)
        assert len(details) <= mentions.maximum_details + 1


def test_weak_tastes_are_never_spelled_out(settings: Settings) -> None:
    tastes = np.full(len(settings.amenities.tastes), -1.0)
    generator = np.random.default_rng(2)
    noise = set(settings.amenities.mentions.noise_phrases)
    for _ in range(MANY_DRAWS):
        details = pick_query_details(generator, tastes, settings, 1.0)
        assert set(details) <= noise
