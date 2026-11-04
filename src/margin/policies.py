"""Builds the slates and margins that each pricing policy would show."""

from dataclasses import dataclass

import numpy as np

from src.world.models import FloatArray, IntArray

RANDOM = "random_logged_margin"
CONVERSION = "conversion_ranker_logged_margin"
PROFIT = "profit_ranker_logged_margin"
MARGIN_ONLY = "optimized_margin_conversion_slate"
OPTIMIZED = "optimized_margin_profit_slate"
LAB_ORACLE = "lab_oracle_true_probabilities"


@dataclass
class Policy:
    slate: IntArray
    margin_index: IntArray


def top_slate(score: FloatArray, size: int) -> IntArray:
    """Picks the packages with the highest score in every session."""
    return np.argsort(-score, axis=1, kind="stable")[:, :size]


def gather(values: FloatArray | IntArray, slate: IntArray) -> FloatArray:
    """Reads the values of the shown packages in every session."""
    chosen = np.take_along_axis(values, slate, axis=1)
    return np.asarray(chosen, dtype=np.float64)


def logged_probability(probability: FloatArray, logged: IntArray) -> FloatArray:
    """Reads the probability at the margin that was actually shown."""
    return np.take_along_axis(probability, logged[:, :, None], axis=2)[:, :, 0]


def profit_grid(net_cost: FloatArray, grid: list[float]) -> FloatArray:
    """Computes the money earned per booking at every margin of the grid."""
    return net_cost[:, :, None] * np.array(grid)[None, None, :]


def random_policy(
    logged: IntArray,
    size: int,
    generator: np.random.Generator,
) -> Policy:
    """Shows random packages at their logged margins."""
    scores = generator.random(logged.shape)
    slate = top_slate(scores, size)
    return Policy(slate, np.asarray(gather(logged, slate), dtype=np.int64))


def logged_margin_policy(score: FloatArray, logged: IntArray, size: int) -> Policy:
    """Ranks packages by a score and keeps their logged margins."""
    slate = top_slate(score, size)
    return Policy(slate, np.asarray(gather(logged, slate), dtype=np.int64))


def fixed_margin_policy(score: FloatArray, position: int, size: int) -> Policy:
    """Ranks packages by a score and charges one margin for all of them."""
    slate = top_slate(score, size)
    return Policy(slate, np.full(slate.shape, position, dtype=np.int64))


def optimized_policy(
    probability: FloatArray,
    net_cost: FloatArray,
    grid: list[float],
    size: int,
) -> Policy:
    """Picks the margin with the best expected profit and the best packages."""
    value = probability * profit_grid(net_cost, grid)
    best = value.argmax(axis=2)
    slate = top_slate(value.max(axis=2), size)
    return Policy(slate, np.asarray(gather(best, slate), dtype=np.int64))


def optimized_margins_policy(
    probability: FloatArray,
    net_cost: FloatArray,
    grid: list[float],
    slate: IntArray,
) -> Policy:
    """Keeps a given slate and picks the most profitable margin per package."""
    best = (probability * profit_grid(net_cost, grid)).argmax(axis=2)
    return Policy(slate, np.asarray(gather(best, slate), dtype=np.int64))
