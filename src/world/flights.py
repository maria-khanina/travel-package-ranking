"""Flight geometry, stops, duration and fares between hubs and destinations."""

import numpy as np

from src.settings.models import FlightSettings
from src.world.columns import EARTH_RADIUS_KILOMETERS
from src.world.helpers import normalize_rows, sample_categorical
from src.world.models import FloatArray, IntArray


def compute_distance_kilometers(
    origin_latitude: float,
    origin_longitude: float,
    destination_latitude: FloatArray,
    destination_longitude: FloatArray,
) -> FloatArray:
    """Computes great-circle distances from one hub to many destinations."""
    origin_latitude_radians = np.radians(origin_latitude)
    destination_latitude_radians = np.radians(destination_latitude)
    latitude_gap = destination_latitude_radians - origin_latitude_radians
    longitude_gap = np.radians(destination_longitude - origin_longitude)
    haversine = (
        np.sin(latitude_gap / 2.0) ** 2
        + np.cos(origin_latitude_radians)
        * np.cos(destination_latitude_radians)
        * np.sin(longitude_gap / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KILOMETERS * np.arcsin(np.sqrt(haversine))


def compute_stop_probabilities(
    distance_kilometers: FloatArray,
    flight: FlightSettings,
) -> FloatArray:
    """Gives the probabilities of zero, one and two stops for a route."""
    scaled = distance_kilometers / flight.distance_unit_kilometers
    direct = np.clip(
        flight.direct_base - flight.direct_slope * scaled,
        flight.direct_minimum,
        flight.direct_maximum,
    )
    two_stops = np.clip(
        flight.two_stop_slope * scaled,
        0.0,
        flight.two_stop_maximum,
    )
    one_stop = np.clip(1.0 - direct - two_stops, 0.0, 1.0)
    return normalize_rows(np.stack([direct, one_stop, two_stops], axis=1))


def sample_stop_counts(
    random_generator: np.random.Generator,
    distance_kilometers: FloatArray,
    stop_limit: FloatArray,
    flight: FlightSettings,
) -> IntArray:
    """Draws the number of stops, respecting an optional per-row limit."""
    probabilities = compute_stop_probabilities(distance_kilometers, flight)
    limit = np.where(np.isnan(stop_limit), flight.maximum_stops, stop_limit)
    options = np.arange(probabilities.shape[1], dtype=np.int64).reshape(1, -1)
    allowed = np.asarray(options <= limit.reshape(-1, 1), dtype=np.float64)
    weights = normalize_rows(probabilities * allowed)
    return sample_categorical(random_generator, weights)


def sample_flight_duration(
    random_generator: np.random.Generator,
    distance_kilometers: FloatArray,
    stop_counts: IntArray,
    flight: FlightSettings,
) -> FloatArray:
    """Draws the total travel time in hours for every flight."""
    count = len(distance_kilometers)
    layover_noise = np.abs(
        random_generator.normal(0.0, flight.layover_standard_deviation_hours, count)
    )
    layover = stop_counts * (flight.layover_base_hours + layover_noise)
    noise = random_generator.normal(
        0.0,
        flight.duration_noise_standard_deviation,
        count,
    )
    cruise = distance_kilometers / flight.speed_kilometers_per_hour
    hours = cruise + flight.overhead_hours + layover + noise
    return np.round(hours, flight.duration_decimals)


def compute_base_fare(
    distance_kilometers: FloatArray,
    flight: FlightSettings,
) -> FloatArray:
    """Computes the economy fare per traveler before adjustments."""
    return flight.fare_base + flight.fare_per_kilometers * distance_kilometers


def compute_lead_factor(days: IntArray, flight: FlightSettings) -> FloatArray:
    """Scales fares by how close the departure date is."""
    thresholds = np.array(flight.lead_days)
    position = np.searchsorted(thresholds, days, side="right")
    return np.array(flight.lead_factors)[position]
