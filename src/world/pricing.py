"""Computes boards, cancellation terms, flights and prices of the packages."""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.settings.models import CompetitorSettings, PricingSettings, SeasonSettings
from src.world.flights import (
    compute_base_fare,
    compute_lead_factor,
    sample_flight_duration,
    sample_stop_counts,
)
from src.world.helpers import (
    normalize_rows,
    read_floats,
    read_integers,
    sample_categorical,
)
from src.world.models import BoolArray, FloatArray


def assign_boards(
    random_generator: np.random.Generator,
    rows: pd.DataFrame,
    available_boards: BoolArray,
    pricing: PricingSettings,
) -> None:
    """Chooses the board type offered, honouring a requested board filter."""
    offer_weights = np.array(pricing.board_offer_weights)
    weights = available_boards.astype(np.float64) * offer_weights
    picked = sample_categorical(random_generator, normalize_rows(weights))
    offered = np.array(pricing.board_names, dtype=np.object_)[picked]
    requested = rows["req_board"]
    rows["board_type"] = np.where(requested.notna(), requested, offered)


def assign_free_cancellation(
    random_generator: np.random.Generator,
    rows: pd.DataFrame,
    pricing: PricingSettings,
) -> None:
    """Decides whether each package can be cancelled for free."""
    days = read_integers(rows, "days_to_departure")
    far = days > pricing.free_cancellation_far_days
    premium = read_integers(rows, "stars") >= pricing.free_cancellation_star_limit
    probability = pricing.free_cancellation_base
    probability = probability + pricing.free_cancellation_far_bonus * far
    probability = probability + pricing.free_cancellation_star_bonus * premium
    rows["free_cancellation"] = random_generator.random(len(rows)) < probability


def assign_flights(
    random_generator: np.random.Generator,
    rows: pd.DataFrame,
    pricing: PricingSettings,
) -> None:
    """Draws the stops and the duration of the flight of each package."""
    distance = read_floats(rows, "hid_route_km")
    limit = read_floats(rows, "req_max_stops")
    stops = sample_stop_counts(random_generator, distance, limit, pricing.flight)
    rows["number_of_stops"] = stops
    rows["flight_duration_hours"] = sample_flight_duration(
        random_generator,
        distance,
        stops,
        pricing.flight,
    )


def compute_travelers(rows: pd.DataFrame, child_share: float) -> FloatArray:
    """Counts the paying travelers, discounting children."""
    adults = read_floats(rows, "n_adults")
    return adults + child_share * read_floats(rows, "n_children")


def compute_hotel_cost(
    rows: pd.DataFrame,
    price_level: FloatArray,
    season: FloatArray,
    pricing: PricingSettings,
) -> FloatArray:
    """Computes the hotel cost of the whole stay for the whole party."""
    board_factor: dict[str, float] = dict(
        zip(pricing.board_names, pricing.board_factors)
    )
    star_factor = np.array(pricing.star_factors)[read_integers(rows, "stars") - 1]
    boards = rows["board_type"].map(board_factor).to_numpy(dtype=np.float64)
    nightly = price_level * pricing.price_scale * star_factor * season * boards
    nightly = nightly * read_floats(rows, "hid_room_multiplier")
    nightly = nightly * read_floats(rows, "hid_price_noise")
    travelers = compute_travelers(rows, pricing.child_hotel_share)
    return nightly * read_floats(rows, "nights") * travelers


def compute_flight_cost(
    random_generator: np.random.Generator,
    rows: pd.DataFrame,
    season: FloatArray,
    seasons: SeasonSettings,
    pricing: PricingSettings,
) -> FloatArray:
    """Computes the flight cost for the whole party."""
    flight = pricing.flight
    distance = read_floats(rows, "hid_route_km")
    stops = read_floats(rows, "number_of_stops")
    spread = flight.fare_noise_standard_deviation
    noise = np.exp(random_generator.normal(0.0, spread, len(rows)))
    fare = compute_base_fare(distance, flight) * (1 - flight.stop_discount * stops)
    fare = fare * compute_lead_factor(read_integers(rows, "days_to_departure"), flight)
    fare = fare * season**seasons.flight_price_exponent * noise
    return fare * compute_travelers(rows, pricing.child_flight_share)


def compute_competitor_price(
    random_generator: np.random.Generator,
    net_cost: FloatArray,
    competition: FloatArray,
    competitor: CompetitorSettings,
    decimals: int,
) -> FloatArray:
    """Simulates the retail price a competitor offers for the same package."""
    spread = competitor.markup_low_competition - competitor.markup_high_competition
    mean = competitor.markup_low_competition - spread * competition
    markup = random_generator.normal(mean, competitor.markup_standard_deviation)
    markup = np.maximum(markup, competitor.markup_minimum)
    noise_spread = competitor.price_noise_standard_deviation
    noise = np.exp(random_generator.normal(0.0, noise_spread, len(net_cost)))
    return np.round(net_cost * (1 + markup) * noise, decimals)


def assign_prices(
    random_generator: np.random.Generator,
    rows: pd.DataFrame,
    destinations: pd.DataFrame,
    price_season: FloatArray,
    settings: Settings,
) -> None:
    """Computes the net cost and the competitor price of every package."""
    pricing = settings.pricing
    destination = read_integers(rows, "hid_destination")
    season = price_season[destination, read_integers(rows, "departure_month")]
    price_level = read_floats(destinations, "price_level")[destination]
    cost = compute_hotel_cost(rows, price_level, season, pricing)
    cost = cost + compute_flight_cost(
        random_generator,
        rows,
        season,
        settings.seasons,
        pricing,
    )
    noise_spread = pricing.total_noise_standard_deviation
    noise = np.exp(random_generator.normal(0.0, noise_spread, len(rows)))
    rows["net_cost"] = np.round(cost * noise, pricing.money_decimals)
    rows["competitor_retail_price"] = compute_competitor_price(
        random_generator,
        read_floats(rows, "net_cost"),
        read_floats(destinations, "competition")[destination],
        pricing.competitor,
        pricing.money_decimals,
    )
