"""Runs the cross-validated experiments of the booking model."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.modeling.evaluation import (
    booked_session_ranks,
    compute_log_loss,
    evaluate_scores,
    safe_auc,
)
from src.modeling.frames import build_frame
from src.modeling.persistence import (
    experiment_models_folder,
    save_fold_model,
    save_fit_diagnostics,
    save_manifest,
    save_scores,
)
from src.modeling.ranking import random_ranking_metrics
from src.modeling.split import SplitMasks
from src.modeling.training import (
    POINTWISE,
    build_pool,
    predict_scores,
    train_model,
)
from src.settings.loader import Settings
from src.settings.modeling_models import ExperimentRecord
from src.world.models import BoolArray, FloatArray, IntArray

ORACLE_NAME = "reference_hidden_fit"
RANDOM_NAME = "reference_random"

ResultRow = dict[str, float | str]


@dataclass
class ModelInputs:
    table: pd.DataFrame
    labels: BoolArray
    sessions: IntArray
    hotels: pd.Series
    hidden_fit: FloatArray
    masks: SplitMasks


def keep_informative_groups(
    mask: BoolArray,
    inputs: ModelInputs,
    loss: str,
) -> BoolArray:
    """For listwise training keeps only the sessions that contain a booking."""
    if loss == POINTWISE:
        return mask
    booked = np.unique(inputs.sessions[inputs.labels])
    return mask & np.isin(inputs.sessions, booked)


def fold_masks(
    fold: int,
    inputs: ModelInputs,
    settings: Settings,
    loss: str,
) -> tuple[BoolArray, BoolArray, BoolArray]:
    """Builds the train, early-stopping and evaluation masks of one fold."""
    count = settings.modeling.split.fold_count
    stopping_fold = (fold + 1) % count
    masks = inputs.masks
    warm = masks.development & ~masks.cold
    train = warm & (masks.fold != fold) & (masks.fold != stopping_fold)
    stopping = warm & (masks.fold == stopping_fold)
    return (
        keep_informative_groups(train, inputs, loss),
        keep_informative_groups(stopping, inputs, loss),
        masks.fold == fold,
    )


def build_fold_pools(
    frame: pd.DataFrame,
    categorical: list[str],
    inputs: ModelInputs,
    masks: tuple[BoolArray, BoolArray],
    loss: str,
) -> list[object]:
    """Builds the training pool and the early-stopping pool of a fold."""
    pools: list[object] = []
    for mask in masks:
        labels = inputs.labels[mask]
        groups = inputs.sessions[mask]
        pools.append(build_pool(frame[mask], labels, categorical, groups, loss))
    return pools


def fit_fold(
    frame: pd.DataFrame,
    categorical: list[str],
    inputs: ModelInputs,
    experiment: ExperimentRecord,
    fold: int,
    settings: Settings,
) -> tuple[FloatArray, BoolArray, Any]:
    """Trains on some folds and scores the held-out fold."""
    loss = experiment.loss
    train, stopping, evaluation = fold_masks(fold, inputs, settings, loss)
    pools = build_fold_pools(frame, categorical, inputs, (train, stopping), loss)
    model = train_model(pools[0], pools[1], settings.modeling.catboost, loss)
    scores = predict_scores(model, frame[evaluation], categorical, loss)
    return scores, evaluation, model


def diagnose_fold(
    model: Any,
    frame: pd.DataFrame,
    categorical: list[str],
    inputs: ModelInputs,
    context: tuple[ExperimentRecord, int, FloatArray],
    settings: Settings,
) -> dict[str, float]:
    """Compares the quality on training, early-stopping and held-out rows."""
    experiment, fold, scores = context
    train, stopping, evaluation = fold_masks(fold, inputs, settings, experiment.loss)
    sample = np.flatnonzero(train)[: settings.modeling.diagnostic_rows]
    quality: dict[str, float] = {"fold": float(fold)}
    for name, rows in (("train", sample), ("stopping", np.flatnonzero(stopping))):
        predicted = predict_scores(
            model, frame.iloc[rows], categorical, experiment.loss
        )
        quality[f"{name}_auc"] = safe_auc(inputs.labels[rows], predicted)
    quality["evaluation_auc"] = safe_auc(inputs.labels[evaluation], scores)
    quality["best_iteration"] = float(model.get_best_iteration() or 0)
    return quality


def cross_validate(
    experiment: ExperimentRecord,
    inputs: ModelInputs,
    settings: Settings,
) -> tuple[FloatArray, list[dict[str, float]]]:
    """Produces out-of-fold scores and the fit diagnostics of every fold."""
    frame, categorical = build_frame(
        inputs.table, inputs.hotels, settings.features.columns, experiment
    )
    scores = np.full(len(inputs.labels), np.nan)
    diagnostics: list[dict[str, float]] = []
    folder = experiment_models_folder(settings, experiment.name)
    save_manifest(folder, list(frame.columns), categorical, experiment.loss)
    for fold in range(settings.modeling.split.fold_count):
        fold_scores, evaluation, model = fit_fold(
            frame, categorical, inputs, experiment, fold, settings
        )
        scores[evaluation] = fold_scores
        context = (experiment, fold, fold_scores)
        diagnostics.append(
            diagnose_fold(model, frame, categorical, inputs, context, settings)
        )
        save_fold_model(model, folder, fold)
    return scores, diagnostics


def summarize_scores(
    name: str,
    scores: FloatArray,
    inputs: ModelInputs,
    settings: Settings,
    iterations: list[float],
) -> ResultRow:
    """Computes the development metrics of out-of-fold scores."""
    development = inputs.masks.development
    metrics = evaluate_scores(
        scores[development],
        inputs.labels[development],
        inputs.masks.cold[development],
        settings.modeling.split.pool_size,
        settings.modeling.top_k_values,
    )
    result: ResultRow = {"experiment": name, **metrics}
    result["mean_best_iteration"] = float(np.mean(iterations))
    return result


def attach_log_loss(
    row: ResultRow,
    scores: FloatArray,
    inputs: ModelInputs,
    loss: str,
) -> None:
    """Adds the log loss for pointwise models, where it is meaningful."""
    development = inputs.masks.development
    if loss == POINTWISE:
        row["log_loss"] = compute_log_loss(
            inputs.labels[development], scores[development]
        )


def reference_rows(
    inputs: ModelInputs,
    settings: Settings,
) -> tuple[list[ResultRow], dict[str, FloatArray]]:
    """Reports random ranking and ranking by hidden fit on development data."""
    development = inputs.masks.development
    pool = settings.modeling.split.pool_size
    top_k = settings.modeling.top_k_values
    fit = inputs.hidden_fit[development]
    labels = inputs.labels[development]
    cold = inputs.masks.cold[development]
    oracle: ResultRow = {
        "experiment": ORACLE_NAME,
        **evaluate_scores(fit, labels, cold, pool, top_k),
    }
    chance: ResultRow = {
        "experiment": RANDOM_NAME,
        **random_ranking_metrics(pool, top_k),
    }
    return [oracle, chance], {ORACLE_NAME: booked_session_ranks(fit, labels, pool)}


def run_experiment(
    experiment: ExperimentRecord,
    inputs: ModelInputs,
    settings: Settings,
) -> tuple[ResultRow, FloatArray]:
    """Cross-validates one experiment and returns its metrics and booked ranks."""
    scores, diagnostics = cross_validate(experiment, inputs, settings)
    save_scores(settings, experiment.name, scores)
    save_fit_diagnostics(settings, experiment.name, diagnostics)
    iterations = [fold["best_iteration"] for fold in diagnostics]
    row = summarize_scores(experiment.name, scores, inputs, settings, iterations)
    attach_log_loss(row, scores, inputs, experiment.loss)
    development = inputs.masks.development
    pool = settings.modeling.split.pool_size
    labels = inputs.labels[development]
    return row, booked_session_ranks(scores[development], labels, pool)
