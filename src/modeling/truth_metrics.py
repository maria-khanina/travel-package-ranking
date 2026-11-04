"""Ranking metrics that use the true booking probabilities as the reference."""

import numpy as np
import pandas as pd

from src.modeling.bootstrap import CHUNK_REPLICATES, draw_indices, interval
from src.settings.modeling_models import BootstrapSettings
from src.world.models import BoolArray, FloatArray, IntArray

SessionParts = tuple[FloatArray, FloatArray]


def session_discount(pool_size: int) -> FloatArray:
    """Gives the discount of each rank position in the cumulative gain."""
    return 1.0 / np.log2(np.arange(pool_size) + 2.0)


def hit_parts(order: IntArray, probability: FloatArray, top_k: int) -> SessionParts:
    """Splits the expected hit rate into its per-session numerator and denominator."""
    chosen = np.take_along_axis(probability, order, axis=1)[:, :top_k]
    return chosen.sum(axis=1), probability.sum(axis=1)


def subset_hit_parts(
    order: IntArray,
    probability: FloatArray,
    subset: BoolArray,
    top_k: int,
) -> SessionParts:
    """Splits the expected hit rate for bookings landing on a subset of packages."""
    masked = probability * subset
    chosen = np.take_along_axis(masked, order, axis=1)[:, :top_k]
    return chosen.sum(axis=1), masked.sum(axis=1)


def ndcg_parts(order: IntArray, probability: FloatArray, top_k: int) -> SessionParts:
    """Splits the true NDCG into its per-session gain and ideal gain."""
    discount = session_discount(probability.shape[1])[:top_k]
    chosen = np.take_along_axis(probability, order, axis=1)[:, :top_k]
    ideal = -np.sort(-probability, axis=1)[:, :top_k]
    return (chosen * discount).sum(axis=1), (ideal * discount).sum(axis=1)


def session_spearman(scores: FloatArray, utility: FloatArray) -> FloatArray:
    """Correlates the ranks of scores and true utilities inside each session."""
    left = scores.argsort(axis=1).argsort(axis=1).astype(np.float64)
    right = utility.argsort(axis=1).argsort(axis=1).astype(np.float64)
    left -= left.mean(axis=1, keepdims=True)
    right -= right.mean(axis=1, keepdims=True)
    norm = np.sqrt((left**2).sum(axis=1) * (right**2).sum(axis=1))
    return (left * right).sum(axis=1) / np.where(norm > 0, norm, 1.0)


def metric_parts(
    scores: FloatArray,
    probability: FloatArray,
    utility: FloatArray,
    cold: BoolArray,
    top_k_values: list[int],
) -> dict[str, SessionParts]:
    """Builds the per-session parts of every true-reference metric."""
    order = np.argsort(-scores, axis=1, kind="stable")
    parts: dict[str, SessionParts] = {}
    for top_k in top_k_values:
        key = f"expected_hit_rate_at_{top_k}"
        parts[key] = hit_parts(order, probability, top_k)
        parts[f"{key}_cold"] = subset_hit_parts(order, probability, cold, top_k)
        parts[f"{key}_warm"] = subset_hit_parts(order, probability, ~cold, top_k)
        parts[f"true_ndcg_at_{top_k}"] = ndcg_parts(order, probability, top_k)
    ones = np.ones(len(scores))
    parts["rank_correlation_with_utility"] = (session_spearman(scores, utility), ones)
    return parts


def ratio_replicates(
    parts: SessionParts,
    indices: IntArray,
) -> FloatArray:
    """Computes the ratio of sums on each resampled set of sessions."""
    numerator, denominator = parts
    return numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1)


def ratio_interval(
    parts: SessionParts,
    settings: BootstrapSettings,
) -> tuple[float, float, float]:
    """Reports the ratio of sums with a bootstrap interval over sessions."""
    generator = np.random.default_rng(settings.seed)
    count = len(parts[0])
    samples: list[FloatArray] = []
    for start in range(0, settings.replicates, CHUNK_REPLICATES):
        size = min(CHUNK_REPLICATES, settings.replicates - start)
        samples.append(ratio_replicates(parts, draw_indices(generator, count, size)))
    low, high = interval(np.concatenate(samples), settings)
    return float(parts[0].sum() / parts[1].sum()), low, high


def metric_table(
    model: str,
    parts: dict[str, SessionParts],
    settings: BootstrapSettings,
) -> pd.DataFrame:
    """Builds the report rows of one model with bootstrap intervals."""
    rows: list[dict[str, float | str]] = []
    for metric, session_parts in parts.items():
        estimate, low, high = ratio_interval(session_parts, settings)
        rows.append(
            {
                "model": model,
                "metric": metric,
                "estimate": estimate,
                "lower": low,
                "upper": high,
            }
        )
    return pd.DataFrame(rows)


def difference_parts(first: SessionParts, second: SessionParts) -> SessionParts:
    """Gives the parts of the difference between two models on one metric."""
    return first[0] - second[0], first[1]
