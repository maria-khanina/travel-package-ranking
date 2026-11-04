"""Bootstrap confidence intervals and paired comparisons over sessions."""

import numpy as np
import pandas as pd

from src.settings.modeling_models import BootstrapSettings
from src.world.models import FloatArray, IntArray

CHUNK_REPLICATES = 250
SESSION_AUC = "session_auc"


def session_values(
    ranks: FloatArray,
    pool_size: int,
    top_k_values: list[int],
) -> dict[str, FloatArray]:
    """Turns the rank of the booked package into one number per session."""
    values: dict[str, FloatArray] = {}
    for top_k in top_k_values:
        values[f"hit_rate_at_{top_k}"] = (ranks <= top_k).astype(np.float64)
        gain = 1.0 / np.log2(ranks + 1.0)
        values[f"ndcg_at_{top_k}"] = np.where(ranks <= top_k, gain, 0.0)
    values[SESSION_AUC] = (pool_size - ranks) / (pool_size - 1.0)
    return values


def draw_indices(
    generator: np.random.Generator,
    count: int,
    size: int,
) -> IntArray:
    """Draws sessions with replacement for several bootstrap replicates."""
    return np.asarray(generator.integers(0, count, size=(size, count)))


def replicate_means(
    values: dict[str, FloatArray],
    settings: BootstrapSettings,
) -> dict[str, FloatArray]:
    """Computes every metric on each bootstrap replicate of the sessions."""
    generator = np.random.default_rng(settings.seed)
    count = len(next(iter(values.values())))
    means: dict[str, list[FloatArray]] = {name: [] for name in values}
    for start in range(0, settings.replicates, CHUNK_REPLICATES):
        size = min(CHUNK_REPLICATES, settings.replicates - start)
        indices = draw_indices(generator, count, size)
        for name, array in values.items():
            means[name].append(array[indices].mean(axis=1))
    return {name: np.concatenate(parts) for name, parts in means.items()}


def interval(samples: FloatArray, settings: BootstrapSettings) -> tuple[float, float]:
    """Takes the percentile interval of the bootstrap replicates."""
    low = float(np.percentile(samples, settings.lower_percentile))
    return low, float(np.percentile(samples, settings.upper_percentile))


def metric_rows(
    model: str,
    values: dict[str, FloatArray],
    means: dict[str, FloatArray],
    settings: BootstrapSettings,
    is_difference: bool,
) -> list[dict[str, float | str]]:
    """Builds the report rows of one model: estimate, interval and win share."""
    rows: list[dict[str, float | str]] = []
    for metric, array in values.items():
        low, high = interval(means[metric], settings)
        row: dict[str, float | str] = {"model": model, "metric": metric}
        row["difference" if is_difference else "estimate"] = float(array.mean())
        row.update({"lower": low, "upper": high})
        if is_difference:
            row["share_better"] = float(np.mean(means[metric] > 0.0))
        rows.append(row)
    return rows


def bootstrap_table(
    ranks_by_model: dict[str, FloatArray],
    pool_size: int,
    top_k_values: list[int],
    settings: BootstrapSettings,
) -> pd.DataFrame:
    """Reports every metric of every model with its confidence interval."""
    rows: list[dict[str, float | str]] = []
    for model, ranks in ranks_by_model.items():
        values = session_values(ranks, pool_size, top_k_values)
        means = replicate_means(values, settings)
        rows += metric_rows(model, values, means, settings, False)
    return pd.DataFrame(rows)


def paired_table(
    ranks_by_model: dict[str, FloatArray],
    pool_size: int,
    top_k_values: list[int],
    settings: BootstrapSettings,
) -> pd.DataFrame:
    """Compares every model with the baseline on the same resampled sessions."""
    base = session_values(ranks_by_model[settings.baseline], pool_size, top_k_values)
    rows: list[dict[str, float | str]] = []
    for model, ranks in ranks_by_model.items():
        if model == settings.baseline:
            continue
        values = session_values(ranks, pool_size, top_k_values)
        differences = {name: values[name] - base[name] for name in values}
        means = replicate_means(differences, settings)
        rows += metric_rows(model, differences, means, settings, True)
    return pd.DataFrame(rows)
