"""Evaluates the saved scores of every experiment against the true probabilities."""

import numpy as np
import pandas as pd

from src.modeling.persistence import SCORES_SUFFIX
from src.modeling.pipeline import experiment_folder, load_split_masks
from src.modeling.truth_metrics import (
    SessionParts,
    difference_parts,
    metric_parts,
    metric_table,
    ratio_interval,
)
from src.settings.loader import Settings
from src.storage import MATCH_FILE, TRUE_FILE, dataset_directory, results_directory
from src.world.helpers import read_floats
from src.world.models import BoolArray, FloatArray

PAIRED_FILE = "truth_metrics_paired.csv"
REFERENCES: list[str] = ["hid_aspect_fit", "hid_open_match", "hid_stated_match"]
ORACLE = "reference_true_utility"
CHANCE = "reference_random"

ModelParts = dict[str, dict[str, SessionParts]]


def load_scores(settings: Settings) -> dict[str, FloatArray]:
    """Reads the saved out-of-fold scores of the trained experiments."""
    folder = experiment_folder(settings)
    scores: dict[str, FloatArray] = {}
    for path in sorted(folder.glob(f"*{SCORES_SUFFIX}")):
        scores[path.name.removesuffix(SCORES_SUFFIX)] = np.load(path).astype(np.float64)
    return scores


def reference_scores(settings: Settings, utility: FloatArray) -> dict[str, FloatArray]:
    """Builds the scores of the reference rankings, including pure chance."""
    match = pd.read_parquet(dataset_directory(settings) / MATCH_FILE)
    scores = {name: read_floats(match, name) for name in REFERENCES}
    scores[ORACLE] = utility
    seed = settings.modeling.truth.seed
    scores[CHANCE] = np.random.default_rng(seed).random(len(utility))
    return scores


def development_sessions(scores: dict[str, FloatArray], pool: int) -> BoolArray:
    """Finds the sessions that have out-of-fold scores."""
    first = next(iter(scores.values()))
    return ~np.isnan(first.reshape(-1, pool)[:, 0])


def evaluate_models(
    scores: dict[str, FloatArray],
    reference: tuple[FloatArray, FloatArray, BoolArray],
    mask: BoolArray,
    settings: Settings,
) -> ModelParts:
    """Computes the per-session parts of every metric for every model."""
    pool = settings.pool.size
    probability, utility, cold = reference
    shaped_probability = probability.reshape(-1, pool)[mask]
    shaped_utility = utility.reshape(-1, pool)[mask]
    shaped_cold = cold.reshape(-1, pool)[mask]
    result: ModelParts = {}
    for name, values in scores.items():
        result[name] = metric_parts(
            values.reshape(-1, pool)[mask],
            shaped_probability,
            shaped_utility,
            shaped_cold,
            settings.modeling.top_k_values,
        )
    return result


def paired_rows(parts: ModelParts, settings: Settings) -> pd.DataFrame:
    """Compares every model with the baseline on the same resampled sessions."""
    bootstrap = settings.modeling.bootstrap
    base = parts[bootstrap.baseline]
    rows: list[dict[str, float | str]] = []
    for name, model in parts.items():
        if name == bootstrap.baseline:
            continue
        for metric, session_parts in model.items():
            diff = difference_parts(session_parts, base[metric])
            estimate, low, high = ratio_interval(diff, bootstrap)
            rows.append(
                {
                    "model": name,
                    "metric": metric,
                    "difference": estimate,
                    "lower": low,
                    "upper": high,
                }
            )
    return pd.DataFrame(rows)


def run_truth_metrics(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates all saved scores and writes the interval and paired tables."""
    truth = pd.read_parquet(dataset_directory(settings) / TRUE_FILE)
    probability = read_floats(truth, "hid_true_probability")
    utility = read_floats(truth, "hid_true_utility")
    scores = load_scores(settings)
    mask = development_sessions(scores, settings.pool.size)
    scores.update(reference_scores(settings, utility))
    cold = load_split_masks(settings).cold
    parts = evaluate_models(scores, (probability, utility, cold), mask, settings)
    bootstrap = settings.modeling.bootstrap
    frames = [metric_table(name, value, bootstrap) for name, value in parts.items()]
    intervals = pd.concat(frames, ignore_index=True)
    paired = paired_rows(parts, settings)
    output = results_directory(settings)
    intervals.to_csv(output / settings.modeling.truth.output_file, index=False)
    paired.to_csv(output / PAIRED_FILE, index=False)
    return intervals, paired
