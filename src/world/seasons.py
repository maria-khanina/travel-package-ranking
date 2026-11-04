"""Seasonal availability and price factors of the destinations."""

import numpy as np
import pandas as pd

from src.settings.models import AspectSettings, SeasonSettings
from src.world.columns import DESTINATION_PREFIX, MONTHS_PER_YEAR
from src.world.helpers import read_floats
from src.world.models import BoolArray, FloatArray, IntArray, SeasonTables


def compute_temperature(
    latitude: FloatArray,
    climate_offset: FloatArray,
    month: IntArray,
    seasons: SeasonSettings,
) -> FloatArray:
    """Estimates the typical temperature of a destination in a month."""
    base = seasons.base_temperature - seasons.latitude_curvature * latitude**2
    amplitude = seasons.amplitude_per_degree * np.abs(latitude)
    peak = np.where(
        latitude >= 0,
        seasons.northern_peak_month,
        seasons.southern_peak_month,
    )
    phase = 2 * np.pi * (month - peak) / MONTHS_PER_YEAR
    return base + climate_offset + amplitude * np.cos(phase)


def compute_beach_factor(
    temperature: FloatArray,
    seasons: SeasonSettings,
) -> FloatArray:
    """Maps a temperature to how suitable the weather is for the beach."""
    scaled = (temperature - seasons.beach_threshold) / seasons.beach_scale
    return 1.0 / (1.0 + np.exp(-scaled))


def compute_mountain_factor(
    is_ski: BoolArray,
    month: IntArray,
    seasons: SeasonSettings,
) -> FloatArray:
    """Scales mountain appeal by the snow season for ski destinations."""
    snow = np.isin(month, seasons.snow_months)
    shoulder = np.isin(month, seasons.shoulder_months)
    summer = np.isin(month, seasons.summer_months)
    ski_factor = np.select(
        [snow, shoulder, summer],
        [
            seasons.ski_factor_snow,
            seasons.ski_factor_shoulder,
            seasons.ski_factor_summer,
        ],
        default=seasons.ski_factor_other,
    )
    return np.where(is_ski, ski_factor, seasons.non_ski_mountain_factor)


def compute_ski_demand(
    is_ski: BoolArray,
    month: IntArray,
    seasons: SeasonSettings,
) -> FloatArray:
    """Estimates the seasonal demand for ski destinations."""
    snow = np.isin(month, seasons.snow_months)
    summer = np.isin(month, seasons.summer_months)
    demand = np.select(
        [snow, summer],
        [seasons.ski_demand_snow, seasons.ski_demand_summer],
        default=seasons.ski_demand_other,
    )
    return is_ski.astype(np.float64) * demand


def compute_city_demand(
    city_level: FloatArray,
    month: IntArray,
    seasons: SeasonSettings,
) -> FloatArray:
    """Estimates the seasonal demand for city destinations."""
    peak = np.isin(month, seasons.city_peak_months)
    season = np.where(peak, seasons.city_demand_peak, seasons.city_demand_off)
    return city_level * season


def compute_price_factor(
    destinations: pd.DataFrame,
    beach_factor: FloatArray,
    month: IntArray,
    seasons: SeasonSettings,
    aspects: AspectSettings,
) -> FloatArray:
    """Estimates the seasonal price multiplier of every destination."""
    beach_column = f"{DESTINATION_PREFIX}{aspects.beach}"
    city_column = f"{DESTINATION_PREFIX}{aspects.city}"
    beach_level = read_floats(destinations, beach_column).reshape(-1, 1)
    city_level = read_floats(destinations, city_column).reshape(-1, 1)
    is_ski = destinations["ski"].to_numpy(dtype=np.bool_).reshape(-1, 1)
    demand = np.maximum(
        beach_level * beach_factor,
        compute_ski_demand(is_ski, month, seasons),
    )
    demand = np.maximum(demand, compute_city_demand(city_level, month, seasons))
    return seasons.price_base + seasons.price_gain * demand


def pad_month_axis(table: FloatArray) -> FloatArray:
    """Prepends a unit column so that months can be indexed from one."""
    return np.hstack([np.ones((table.shape[0], 1)), table])


def build_season_tables(
    destinations: pd.DataFrame,
    seasons: SeasonSettings,
    aspects: AspectSettings,
) -> SeasonTables:
    """Builds destination-by-month beach, mountain and price factors."""
    latitude = read_floats(destinations, "latitude").reshape(-1, 1)
    offset = read_floats(destinations, "climate_offset").reshape(-1, 1)
    is_ski = destinations["ski"].to_numpy(dtype=np.bool_).reshape(-1, 1)
    months = np.arange(1, MONTHS_PER_YEAR + 1).reshape(1, -1)
    temperature = compute_temperature(latitude, offset, months, seasons)
    beach = compute_beach_factor(temperature, seasons)
    return SeasonTables(
        beach=pad_month_axis(beach),
        mountain=pad_month_axis(compute_mountain_factor(is_ski, months, seasons)),
        price=pad_month_axis(
            compute_price_factor(destinations, beach, months, seasons, aspects)
        ),
    )
