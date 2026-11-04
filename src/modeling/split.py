"""Splits sessions into a development part with folds and a closed test part."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.settings.modeling_models import SplitSettings
from src.world.helpers import read_integers
from src.world.models import BoolArray, IntArray

NO_FOLD = -1


@dataclass
class SplitMasks:
    development: BoolArray
    test: BoolArray
    cold: BoolArray
    fold: IntArray


def build_session_strata(
    dataset: pd.DataFrame,
    registers: pd.Series,
    label_column: str,
) -> pd.DataFrame:
    """Labels each session by whether it booked and by its message register."""
    frame = pd.DataFrame(
        {
            "session_id": dataset["session_id"].to_numpy(),
            "booked": dataset[label_column].to_numpy(),
            "practical": (registers == "practical").to_numpy(),
        }
    )
    sessions = frame.groupby("session_id").agg({"booked": "max", "practical": "max"})
    booked = sessions["booked"].astype(int)
    sessions["stratum"] = booked * 2 + sessions["practical"].astype(int)
    return sessions


def cut_shuffled(identifiers: IntArray, shares: list[float]) -> list[IntArray]:
    """Cuts an already shuffled array into consecutive parts of given shares."""
    edges = np.round(np.cumsum(shares)[:-1] * len(identifiers)).astype(np.int64)
    return [np.asarray(part, dtype=np.int64) for part in np.split(identifiers, edges)]


def divide_sessions(
    generator: np.random.Generator,
    sessions: pd.DataFrame,
    shares: list[float],
) -> list[IntArray]:
    """Divides sessions into parts, keeping every stratum balanced in all parts."""
    parts: list[list[IntArray]] = [[] for _ in shares]
    for _, group in sessions.groupby("stratum"):
        shuffled = generator.permutation(np.asarray(group.index, dtype=np.int64))
        for part, chunk in zip(parts, cut_shuffled(shuffled, shares)):
            part.append(chunk)
    return [np.concatenate(part) for part in parts]


def choose_cold_hotels(hotel_ids: IntArray, split: SplitSettings) -> IntArray:
    """Picks the hotels that the training data must never contain."""
    generator = np.random.default_rng(split.seed)
    distinct = np.unique(hotel_ids)
    count = int(round(split.cold_hotel_share * len(distinct)))
    return np.asarray(generator.choice(distinct, size=count, replace=False))


def assign_folds(
    session_ids: IntArray,
    folds: list[IntArray],
) -> IntArray:
    """Gives every development row the number of its fold, others none."""
    fold = np.full(len(session_ids), NO_FOLD, dtype=np.int64)
    for number, members in enumerate(folds):
        fold[np.isin(session_ids, members)] = number
    return fold


def build_masks(
    dataset: pd.DataFrame,
    registers: pd.Series,
    split: SplitSettings,
    label_column: str,
    hotel_column: str,
) -> SplitMasks:
    """Builds the row masks of development, test, cold hotels and folds."""
    sessions = build_session_strata(dataset, registers, label_column)
    generator = np.random.default_rng(split.seed)
    development, test = divide_sessions(
        generator, sessions, [1.0 - split.test_share, split.test_share]
    )
    fold_shares = [1.0 / split.fold_count] * split.fold_count
    folds = divide_sessions(generator, sessions.loc[development], fold_shares)
    session_ids = read_integers(dataset, "session_id")
    hotels = read_integers(dataset, hotel_column)
    return SplitMasks(
        development=np.isin(session_ids, development),
        test=np.isin(session_ids, test),
        cold=np.isin(hotels, choose_cold_hotels(hotels, split)),
        fold=assign_folds(session_ids, folds),
    )
