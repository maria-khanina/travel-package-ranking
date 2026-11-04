"""Estimates the revenue of a policy from the model, from logs and from truth."""

from dataclasses import dataclass

import numpy as np

from src.margin.policies import Policy, gather, logged_probability
from src.margin.truth_tables import TrueTables
from src.settings.loader import Settings
from src.world.models import FloatArray, IntArray


@dataclass
class LoggedData:
    net_cost: FloatArray
    booked: FloatArray
    propensity: FloatArray
    logged_index: IntArray


def slate_profit(
    net_cost: FloatArray,
    policy: Policy,
    grid: list[float],
) -> FloatArray:
    """Computes the money earned per booking of every shown package."""
    cost = np.asarray(gather(net_cost, policy.slate), dtype=np.float64)
    return cost * np.array(grid)[policy.margin_index]


def slate_values(values: FloatArray, policy: Policy) -> FloatArray:
    """Reads the value at the chosen margin for every shown package."""
    sessions = np.arange(values.shape[0], dtype=np.int64).reshape(-1, 1)
    return values[sessions, policy.slate, policy.margin_index]


def model_estimate(
    probability: FloatArray,
    data: LoggedData,
    policy: Policy,
    settings: Settings,
) -> FloatArray:
    """Estimates revenue from the model with a logit choice among the slate."""
    floor = settings.modeling.margin.probability_floor
    shown = logged_probability(probability, data.logged_index)
    outside = np.clip(1.0 - shown.sum(axis=1), floor, 1.0)
    weights = slate_values(probability, policy) / outside[:, None]
    chance = weights / (1.0 + weights.sum(axis=1, keepdims=True))
    profit = slate_profit(data.net_cost, policy, settings.run.margin_grid)
    return (chance * profit).sum(axis=1)


def logged_estimate(
    data: LoggedData,
    policy: Policy,
    settings: Settings,
) -> FloatArray:
    """Estimates revenue by weighting logged bookings with margin propensities."""
    booked = np.asarray(gather(data.booked, policy.slate), dtype=np.float64)
    propensity = np.asarray(gather(data.propensity, policy.slate), dtype=np.float64)
    logged = np.asarray(gather(data.logged_index, policy.slate))
    matched = (logged == policy.margin_index).astype(np.float64)
    profit = slate_profit(data.net_cost, policy, settings.run.margin_grid)
    return (booked * matched * profit / propensity).sum(axis=1)


def noise_revenue(
    shown: FloatArray,
    outside: FloatArray,
    profit: FloatArray,
    noise: FloatArray,
) -> FloatArray:
    """Averages over noise the revenue of a slate with a logit choice."""
    weight = np.exp(shown[None] + noise)
    denominator = np.exp(outside[None, :]) + weight.sum(axis=2)
    chance = weight / denominator[:, :, None]
    return (chance * profit[None]).sum(axis=2).mean(axis=0)


def true_estimate(
    tables: TrueTables,
    data: LoggedData,
    policy: Policy,
    settings: Settings,
) -> FloatArray:
    """Computes the true expected revenue of a policy, for grading only."""
    options = settings.modeling.margin
    sessions = np.arange(tables.utility.shape[0], dtype=np.int64).reshape(-1, 1)
    shown = tables.utility[sessions, policy.slate, policy.margin_index]
    profit = slate_profit(data.net_cost, policy, settings.run.margin_grid)
    generator = np.random.default_rng(options.seed)
    parts: list[FloatArray] = []
    for start in range(0, len(shown), options.chunk_sessions):
        part = slice(start, start + options.chunk_sessions)
        shape = (options.draws,) + shown[part].shape
        noise = settings.choice.noise_sigma * generator.normal(size=shape)
        revenue = noise_revenue(shown[part], tables.outside[part], profit[part], noise)
        parts.append(revenue)
    return np.concatenate(parts)
