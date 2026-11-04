"""Computes true utilities and probabilities of the sample, for grading only."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.behavior.utility import compute_outside_utilities, compute_utilities
from src.margin.sample import MarginSample
from src.modeling.truth import read_constant
from src.settings.loader import Settings
from src.world.helpers import read_floats
from src.world.models import FloatArray, IntArray


@dataclass
class TrueTables:
    utility: FloatArray
    outside: FloatArray
    logged_index: IntArray
    single_probability: FloatArray


def logged_margin_index(rows: pd.DataFrame, grid: list[float], pool: int) -> IntArray:
    """Finds which grid margin was shown in every row."""
    margins = read_floats(rows, "margin_percentage")
    distance = np.abs(margins[:, None] - np.array(grid)[None, :])
    return distance.argmin(axis=1).reshape(-1, pool)


def utility_table(rows: pd.DataFrame, settings: Settings) -> FloatArray:
    """Computes the noise-free utility of every package at every margin."""
    pool = settings.pool.size
    layers: list[FloatArray] = []
    for margin in settings.run.margin_grid:
        values = np.full(len(rows), margin)
        utilities = compute_utilities(rows, settings, values, False)
        layers.append(utilities.reshape(-1, pool))
    return np.stack(layers, axis=2)


def single_probability_chunk(
    utility: FloatArray,
    logged: IntArray,
    outside: FloatArray,
    noise: FloatArray,
) -> FloatArray:
    """Averages over noise the booking probability of one repriced package."""
    logged_utility = np.take_along_axis(utility, logged[:, :, None], axis=2)[:, :, 0]
    logged_weight = np.exp(logged_utility[None] + noise)
    total = np.exp(outside)[None, :] + logged_weight.sum(axis=2)
    rest = total[:, :, None] - logged_weight
    layers: list[FloatArray] = []
    for position in range(utility.shape[2]):
        weight = np.exp(utility[None, :, :, position] + noise)
        layers.append((weight / (rest + weight)).mean(axis=0))
    return np.stack(layers, axis=2)


def single_probabilities(
    tables: tuple[FloatArray, FloatArray, IntArray],
    settings: Settings,
) -> FloatArray:
    """Computes the true booking probability of each package at each margin."""
    utility, outside, logged = tables
    options = settings.modeling.margin
    generator = np.random.default_rng(options.seed)
    parts: list[FloatArray] = []
    for start in range(0, len(utility), options.chunk_sessions):
        stop = start + options.chunk_sessions
        noise = settings.choice.noise_sigma * generator.normal(
            size=(options.draws,) + utility[start:stop].shape[:2]
        )
        parts.append(
            single_probability_chunk(
                utility[start:stop], logged[start:stop], outside[start:stop], noise
            )
        )
    return np.concatenate(parts)


def build_true_tables(sample: MarginSample, settings: Settings) -> TrueTables:
    """Builds the true utilities and probabilities needed to grade policies."""
    pool = settings.pool.size
    utility = utility_table(sample.rows, settings)
    outside = compute_outside_utilities(
        sample.rows.iloc[::pool], settings, read_constant(settings)
    )
    logged = logged_margin_index(sample.rows, settings.run.margin_grid, pool)
    probability = single_probabilities((utility, outside, logged), settings)
    return TrueTables(utility, outside, logged, probability)
