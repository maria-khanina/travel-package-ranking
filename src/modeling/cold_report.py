"""Compares experiments separately on bookings of known and unknown hotels."""

import numpy as np
import pandas as pd

from src.modeling.bootstrap import bootstrap_table, paired_table
from src.modeling.pipeline import load_ranks, load_split_masks
from src.modeling.ranking import locate_booked
from src.settings.loader import Settings
from src.storage import DATASET_FILE, dataset_directory, results_directory
from src.world.models import BoolArray, FloatArray

COLD_WARM_FILE = "cold_warm_intervals.csv"
COLD_WARM_PAIRED_FILE = "cold_warm_paired.csv"
SUBSETS: list[str] = ["cold_hotel_bookings", "warm_hotel_bookings"]


def booked_cold_flags(settings: Settings) -> BoolArray:
    """Marks, for each booked development session, whether the hotel was cold."""
    modeling = settings.modeling
    masks = load_split_masks(settings)
    path = dataset_directory(settings) / DATASET_FILE
    labels = pd.read_parquet(path, columns=[modeling.label_column])
    booked = labels[modeling.label_column].to_numpy(dtype=np.bool_)
    pool = modeling.split.pool_size
    sessions, positions = locate_booked(booked[masks.development], pool)
    return masks.cold[masks.development].reshape(-1, pool)[sessions, positions]


def split_ranks(
    ranks: dict[str, FloatArray],
    flags: BoolArray,
) -> dict[str, dict[str, FloatArray]]:
    """Splits the booked-session ranks of every model into cold and warm parts."""
    cold = {name: values[flags] for name, values in ranks.items()}
    warm = {name: values[~flags] for name, values in ranks.items()}
    return {SUBSETS[0]: cold, SUBSETS[1]: warm}


def run_cold_report(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reports intervals and paired differences on cold and warm bookings."""
    modeling = settings.modeling
    parts = split_ranks(load_ranks(settings), booked_cold_flags(settings))
    pool = modeling.split.pool_size
    intervals: list[pd.DataFrame] = []
    paired: list[pd.DataFrame] = []
    for subset, ranks in parts.items():
        top_k = modeling.top_k_values
        estimates = bootstrap_table(ranks, pool, top_k, modeling.bootstrap)
        differences = paired_table(ranks, pool, top_k, modeling.bootstrap)
        intervals.append(estimates.assign(subset=subset))
        paired.append(differences.assign(subset=subset))
    output = results_directory(settings)
    result = (pd.concat(intervals), pd.concat(paired))
    result[0].to_csv(output / COLD_WARM_FILE, index=False)
    result[1].to_csv(output / COLD_WARM_PAIRED_FILE, index=False)
    return result
