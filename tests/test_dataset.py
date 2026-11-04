"""Structural checks of the generated dataset."""

import numpy as np
import pytest

from src.dataset import DatasetBundle
from src.settings.loader import Settings
from src.world.builder import build_world
from src.world.destinations import build_destination_table
from src.world.models import WorldTables


def test_pool_size(bundle: DatasetBundle, settings: Settings) -> None:
    sizes = bundle.public.groupby("session_id").size()
    assert (sizes == settings.pool.size).all()


def test_at_most_one_booking(bundle: DatasetBundle) -> None:
    bookings = bundle.public.groupby("session_id")["is_booked"].sum()
    assert bookings.max() <= 1


def test_conversion_near_target(bundle: DatasetBundle, settings: Settings) -> None:
    conversion = bundle.public.groupby("session_id")["is_booked"].sum().mean()
    gap = abs(conversion - settings.run.target_conversion)
    assert gap < settings.testing.conversion_tolerance


def test_margins_on_grid(bundle: DatasetBundle, settings: Settings) -> None:
    margins: set[float] = set(bundle.public["margin_percentage"].round(2))
    assert margins <= set(settings.run.margin_grid)


def test_propensity_has_exploration_floor(
    bundle: DatasetBundle,
    settings: Settings,
) -> None:
    mix = settings.pricing.logging_policy.mix_uniform
    floor = mix / len(settings.run.margin_grid)
    propensity = bundle.public["margin_propensity"]
    assert propensity.min() >= floor - settings.testing.sum_tolerance
    assert propensity.max() <= 1.0


def test_country_and_star_filters_respected(bundle: DatasetBundle) -> None:
    countries = bundle.public.dropna(subset=["req_country"])
    stars = bundle.public.dropna(subset=["req_min_stars"])
    assert (countries["country"] == countries["req_country"]).all()
    assert (stars["stars"] >= stars["req_min_stars"]).all()


def test_board_and_stop_filters_respected(bundle: DatasetBundle) -> None:
    boards = bundle.public.dropna(subset=["req_board"])
    stops = bundle.public.dropna(subset=["req_max_stops"])
    assert (boards["board_type"] == boards["req_board"]).all()
    assert (stops["number_of_stops"] <= stops["req_max_stops"]).all()


def test_values_are_valid(bundle: DatasetBundle, settings: Settings) -> None:
    public = bundle.public
    assert set(public["board_type"]) <= set(settings.pricing.board_names)
    assert (public["net_cost"] > 0).all()
    assert (public["competitor_retail_price"] > 0).all()
    assert public["rating"].between(
        settings.catalogue.rating_minimum,
        settings.catalogue.rating_maximum,
    ).all()


def test_texts_are_attached(bundle: DatasetBundle) -> None:
    columns: list[str] = [
        "user_intent",
        "hotel_description",
        "room_description",
    ]
    assert bundle.public[columns].notna().all().all()


def test_low_star_hotels_are_rare(world: WorldTables, settings: Settings) -> None:
    stars = world.hotels["stars"]
    share = (stars <= settings.testing.low_star_limit).mean()
    assert share < settings.testing.maximum_low_star_share


def test_choice_probabilities_sum_to_one(
    bundle: DatasetBundle,
    settings: Settings,
) -> None:
    totals = bundle.probabilities.sum(axis=1)
    assert np.allclose(totals, 1.0, atol=settings.testing.sum_tolerance)


def test_seed_is_reproducible(settings: Settings) -> None:
    settings.run.session_count = settings.testing.small_session_count
    first = build_world(settings, settings.testing.seed).rows
    second = build_world(settings, settings.testing.seed).rows
    assert first["net_cost"].equals(second["net_cost"])
    assert first["margin_percentage"].equals(second["margin_percentage"])


def test_every_destination_defines_exactly_the_configured_aspects(
    settings: Settings,
) -> None:
    names = set(settings.catalogue.aspects.names)
    for record in settings.destinations:
        assert set(record.aspect_levels) == names


def test_destination_missing_an_aspect_is_rejected(settings: Settings) -> None:
    broken = settings.model_copy(deep=True)
    first = broken.destinations[0]
    del first.aspect_levels[broken.catalogue.aspects.names[0]]
    with pytest.raises(ValueError):
        build_destination_table(broken)
