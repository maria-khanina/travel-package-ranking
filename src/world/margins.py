"""Simulates the historical pricing policy that set the logged margins."""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.settings.models import LoggingPolicySettings
from src.world.helpers import normalize_rows, read_integers, sample_categorical
from src.world.models import FloatArray


def compute_target_margin(
    rows: pd.DataFrame,
    price_level: FloatArray,
    settings: Settings,
) -> FloatArray:
    """Computes the margin the historical rule aims for each package."""
    pricing = settings.pricing
    logging = pricing.logging_policy
    rich_boards: list[str] = [
        pricing.board_half_board,
        pricing.board_all_inclusive,
    ]
    rich_board = rows["board_type"].isin(rich_boards).to_numpy(dtype=np.bool_)
    urgent = read_integers(rows, "days_to_departure") < logging.last_minute_days
    stars_gap = read_integers(rows, "stars") - logging.reference_stars
    target = logging.base + logging.per_star * stars_gap
    target = target + logging.board * rich_board + logging.last_minute * urgent
    target = target + logging.premium * (price_level >= logging.premium_price_level)
    return np.clip(target, logging.lower, logging.upper)


def compute_margin_probabilities(
    target: FloatArray,
    grid: FloatArray,
    logging: LoggingPolicySettings,
) -> FloatArray:
    """Gives the chance of each grid margin: near the target, with exploration."""
    gap = grid.reshape(1, -1) - target.reshape(-1, 1)
    logits = -(gap**2) / (2 * logging.bandwidth**2)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    focused = normalize_rows(weights)
    return (1 - logging.mix_uniform) * focused + logging.mix_uniform / len(grid)


def assign_margins(
    random_generator: np.random.Generator,
    rows: pd.DataFrame,
    price_level: FloatArray,
    settings: Settings,
) -> None:
    """Draws the logged margin and records the chance it was drawn with."""
    grid = np.array(settings.run.margin_grid)
    target = compute_target_margin(rows, price_level, settings)
    logging = settings.pricing.logging_policy
    probabilities = compute_margin_probabilities(target, grid, logging)
    picked = sample_categorical(random_generator, probabilities)
    rows["margin_percentage"] = grid[picked]
    rows["margin_propensity"] = probabilities[np.arange(len(picked)), picked]
