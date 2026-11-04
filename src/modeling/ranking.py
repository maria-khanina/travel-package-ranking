"""Ranking metrics computed on pools of packages shown in one session."""

import numpy as np

from src.world.models import BoolArray, FloatArray, IntArray


def reshape_pools(values: FloatArray, pool_size: int) -> FloatArray:
    """Arranges row values into one row per session."""
    return values.reshape(-1, pool_size)


def locate_booked(booked: BoolArray, pool_size: int) -> tuple[IntArray, IntArray]:
    """Finds the session and the position of every booked package."""
    flags = booked.reshape(-1, pool_size)
    sessions = np.flatnonzero(flags.any(axis=1))
    positions = flags[sessions].argmax(axis=1)
    return sessions, np.asarray(positions, dtype=np.int64)


def rank_of_booked(
    scores: FloatArray,
    sessions: IntArray,
    positions: IntArray,
    pool_size: int,
) -> FloatArray:
    """Gives the rank of the booked package in every session that booked."""
    pools = reshape_pools(scores, pool_size)[sessions]
    chosen = pools[np.arange(len(sessions)), positions]
    better = (pools > chosen[:, None]).sum(axis=1)
    ties = (pools == chosen[:, None]).sum(axis=1) - 1
    return np.asarray(1 + better + ties / 2.0, dtype=np.float64)


def hit_rate(ranks: FloatArray, top_k: int) -> float:
    """Computes the share of booked sessions with the booking in the top K."""
    return float(np.mean(ranks <= top_k))


def ndcg(ranks: FloatArray, top_k: int) -> float:
    """Computes NDCG at K when each session has one relevant package."""
    gain = np.where(ranks <= top_k, 1.0 / np.log2(ranks + 1.0), 0.0)
    return float(np.mean(gain))


def ranking_metrics(
    ranks: FloatArray,
    top_k_values: list[int],
) -> dict[str, float]:
    """Computes hit rate and NDCG for every K."""
    metrics: dict[str, float] = {}
    for top_k in top_k_values:
        metrics[f"hit_rate_at_{top_k}"] = hit_rate(ranks, top_k)
        metrics[f"ndcg_at_{top_k}"] = ndcg(ranks, top_k)
    return metrics


def random_ranking_metrics(
    pool_size: int,
    top_k_values: list[int],
) -> dict[str, float]:
    """Gives the metrics of ranking the pool at random."""
    ranks = np.arange(1, pool_size + 1, dtype=np.float64)
    return ranking_metrics(ranks, top_k_values)
