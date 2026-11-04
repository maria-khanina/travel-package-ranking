"""Array type aliases and containers passed between world builder stages."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
VectorArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
TextArray = npt.NDArray[np.object_]


@dataclass
class CatalogueTables:
    destinations: pd.DataFrame
    hotels: pd.DataFrame
    rooms: pd.DataFrame


@dataclass
class SeasonTables:
    beach: FloatArray
    mountain: FloatArray
    price: FloatArray


@dataclass
class PackageTable:
    attributes: pd.DataFrame
    room_id: IntArray
    hotel_id: IntArray
    destination: IntArray
    country_code: IntArray
    country_codes: dict[str, int]
    stars: IntArray
    rating: FloatArray
    hotel_effect: FloatArray
    price_noise: FloatArray
    boards: BoolArray
    levels: FloatArray
    room_multiplier: FloatArray
    category: TextArray
    view: TextArray
    city: TextArray
    country: TextArray
    hotel_name: TextArray


@dataclass
class PoolSelection:
    package_index: IntArray
    fit: FloatArray
    route_distance: FloatArray
    applied_filters: IntArray


@dataclass
class WorldTables:
    destinations: pd.DataFrame
    hotels: pd.DataFrame
    rooms: pd.DataFrame
    sessions: pd.DataFrame
    rows: pd.DataFrame
