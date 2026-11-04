"""Metrics of a scored set of complete session pools."""

import numpy as np
import pandas as pd

from src.modeling.ranking import (
    locate_booked,
    rank_of_booked,
    ranking_metrics,
)
from src.world.models import BoolArray, FloatArray

LOSS_EPSILON = 1e-15
COLD_REFERENCE_K = 3


def safe_auc(labels: BoolArray, scores: FloatArray) -> float:
    """Computes the AUC, or NaN when only one class is present."""
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(scores).rank().to_numpy(dtype=np.float64)
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def compute_log_loss(labels: BoolArray, scores: FloatArray) -> float:
    """Computes the mean negative log-likelihood of the predicted probabilities."""
    clipped = np.clip(scores, LOSS_EPSILON, 1.0 - LOSS_EPSILON)
    likelihood = np.where(labels, np.log(clipped), np.log(1.0 - clipped))
    return float(-np.mean(likelihood))


def booked_session_ranks(
    scores: FloatArray,
    labels: BoolArray,
    pool_size: int,
) -> FloatArray:
    """Gives the rank of the booked package in every session that booked."""
    sessions, positions = locate_booked(labels, pool_size)
    return rank_of_booked(scores, sessions, positions, pool_size)


def cold_warm_metrics(
    ranks: FloatArray,
    cold_booked: BoolArray,
) -> dict[str, float]:
    """Compares the hit rate of sessions that booked a cold or a warm hotel."""
    key = f"hit_rate_at_{COLD_REFERENCE_K}"
    return {
        f"{key}_cold_booked": float(np.mean(ranks[cold_booked] <= COLD_REFERENCE_K)),
        f"{key}_warm_booked": float(np.mean(ranks[~cold_booked] <= COLD_REFERENCE_K)),
        "cold_booked_sessions": float(cold_booked.sum()),
    }


def auc_metrics(
    scores: FloatArray,
    labels: BoolArray,
    cold: BoolArray,
) -> dict[str, float]:
    """Computes the AUC on all rows and on warm and cold hotels separately."""
    return {
        "auc": safe_auc(labels, scores),
        "auc_warm_hotels": safe_auc(labels[~cold], scores[~cold]),
        "auc_cold_hotels": safe_auc(labels[cold], scores[cold]),
    }


def evaluate_scores(
    scores: FloatArray,
    labels: BoolArray,
    cold: BoolArray,
    pool_size: int,
    top_k_values: list[int],
) -> dict[str, float]:
    """Computes classification and ranking metrics on complete pools."""
    sessions, positions = locate_booked(labels, pool_size)
    ranks = rank_of_booked(scores, sessions, positions, pool_size)
    cold_booked = cold.reshape(-1, pool_size)[sessions, positions]
    metrics = ranking_metrics(ranks, top_k_values)
    metrics["booked_sessions"] = float(len(sessions))
    metrics.update(cold_warm_metrics(ranks, cold_booked))
    metrics.update(auc_metrics(scores, labels, cold))
    return metrics
