"""Trains and applies the CatBoost model that predicts a booking."""

from catboost import CatBoostClassifier, CatBoostRanker, Pool
import numpy as np
import pandas as pd

from src.settings.modeling_models import CatBoostSettings
from src.world.models import BoolArray, FloatArray, IntArray

POINTWISE = "Logloss"
POINTWISE_METRIC = "AUC"

BookingModel = CatBoostClassifier | CatBoostRanker


def build_pool(
    frame: pd.DataFrame,
    labels: BoolArray,
    categorical: list[str],
    groups: IntArray,
    loss: str,
    weights: FloatArray | None,
) -> Pool:
    """Wraps rows, labels and optional row weights into a CatBoost pool."""
    group_ids = None if loss == POINTWISE else groups
    return Pool(
        frame,
        label=labels.astype(np.int64),
        cat_features=categorical,
        group_id=group_ids,
        weight=weights,
    )


def build_model(settings: CatBoostSettings, loss: str) -> BookingModel:
    """Creates an untrained classifier, or a ranker for listwise losses."""
    pointwise = loss == POINTWISE
    model_class = CatBoostClassifier if pointwise else CatBoostRanker
    return model_class(
        iterations=settings.iterations,
        learning_rate=settings.learning_rate,
        depth=settings.depth,
        l2_leaf_reg=settings.l2_leaf_reg,
        early_stopping_rounds=settings.early_stopping_rounds,
        thread_count=settings.thread_count,
        task_type=settings.task_type,
        random_seed=settings.random_seed,
        verbose=settings.verbose,
        loss_function=loss,
        eval_metric=POINTWISE_METRIC if pointwise else None,
    )


def train_model(
    train: Pool,
    stopping: Pool,
    settings: CatBoostSettings,
    loss: str,
) -> BookingModel:
    """Trains a model and keeps its best iteration on the stopping set."""
    model = build_model(settings, loss)
    model.fit(train, eval_set=stopping, use_best_model=True)
    return model


def predict_scores(
    model: BookingModel,
    frame: pd.DataFrame,
    categorical: list[str],
) -> FloatArray:
    """Predicts a booking probability, or a ranking score for listwise models."""
    pool = Pool(frame, cat_features=categorical)
    if isinstance(model, CatBoostClassifier):
        return model.predict_proba(pool)[:, 1]
    return model.predict(pool)
