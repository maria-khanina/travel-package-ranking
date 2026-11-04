"""Computes the true booking probability of every package from the simulator."""

import json

import numpy as np
import pandas as pd

from src.behavior.utility import compute_outside_utilities, compute_utilities
from src.settings.loader import Settings
from src.storage import (
    DATASET_FILE,
    SIMULATION_FILE,
    TRUE_FILE,
    TRUTH_FILE,
    dataset_directory,
)
from src.world.helpers import read_floats
from src.world.models import FloatArray

TEXT_COLUMNS: list[str] = ["user_intent", "hotel_description", "room_description"]


def read_constant(settings: Settings) -> float:
    """Reads the outside-option constant saved with the dataset."""
    path = dataset_directory(settings) / SIMULATION_FILE
    return float(json.loads(path.read_text(encoding="utf-8"))["outside_constant"])


def softmax_with_outside(outside: FloatArray, scores: FloatArray) -> FloatArray:
    """Gives the package booking probabilities, the outside option included."""
    column = np.broadcast_to(outside, scores.shape[:-1] + (1,))
    joined = np.concatenate([column, scores], axis=-1)
    joined = joined - joined.max(axis=-1, keepdims=True)
    weights = np.exp(joined)
    return weights[..., 1:] / weights.sum(axis=-1, keepdims=True)


def integrate_noise(
    outside: FloatArray,
    utilities: FloatArray,
    settings: Settings,
    generator: np.random.Generator,
) -> FloatArray:
    """Averages the booking probabilities over the unobserved package noise."""
    draws = settings.modeling.truth.draws
    noise = generator.normal(size=(draws,) + utilities.shape)
    scores = utilities[None] + settings.choice.noise_sigma * noise
    return softmax_with_outside(outside.reshape(1, -1, 1), scores).mean(axis=0)


def chunk_rows(settings: Settings, start: int, stop: int) -> pd.DataFrame:
    """Reads the rows of a range of sessions, open and hidden columns together."""
    directory = dataset_directory(settings)
    window = [("session_id", ">=", start), ("session_id", "<", stop)]
    public = pd.read_parquet(directory / DATASET_FILE, filters=window)
    hidden = pd.read_parquet(directory / TRUTH_FILE, filters=window)
    public = public.drop(columns=TEXT_COLUMNS)
    hidden = hidden.drop(columns=["session_id", "room_id"])
    parts = [public.reset_index(drop=True), hidden.reset_index(drop=True)]
    return pd.concat(parts, axis=1)


def true_chunk(
    rows: pd.DataFrame,
    constant: float,
    settings: Settings,
    generator: np.random.Generator,
) -> pd.DataFrame:
    """Computes the noise-free utility and the true probability of a chunk."""
    pool = settings.pool.size
    margin = read_floats(rows, "margin_percentage")
    utilities = compute_utilities(rows, settings, margin, False)
    outside = compute_outside_utilities(rows.iloc[::pool], settings, constant)
    shaped = utilities.reshape(-1, pool)
    probability = integrate_noise(outside, shaped, settings, generator)
    return pd.DataFrame(
        {
            "session_id": rows["session_id"],
            "room_id": rows["room_id"],
            "hid_true_utility": utilities.astype(np.float32),
            "hid_true_probability": probability.reshape(-1).astype(np.float32),
        }
    )


def compute_true_table(settings: Settings) -> pd.DataFrame:
    """Computes the true probabilities of all rows, chunk by chunk."""
    truth = settings.modeling.truth
    constant = read_constant(settings)
    generator = np.random.default_rng(truth.seed)
    tables: list[pd.DataFrame] = []
    for start in range(0, settings.run.session_count, truth.chunk_sessions):
        rows = chunk_rows(settings, start, start + truth.chunk_sessions)
        tables.append(true_chunk(rows, constant, settings, generator))
    result = pd.concat(tables, ignore_index=True)
    result.to_parquet(dataset_directory(settings) / TRUE_FILE, index=False)
    return result
