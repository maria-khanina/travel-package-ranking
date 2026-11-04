"""Trains and applies the CatBoost model that predicts a booking."""

import importlib
from typing import Any

import pandas as pd

from src.settings.modeling_models import CatBoostSettings
from src.world.models import BoolArray, FloatArray, IntArray

catboost_library: Any = importlib.import_module("catboost")
POINTWISE = "Logloss"


def build_pool(
    frame: pd.DataFrame,
    labels: BoolArray,
    categorical: list[str],
    groups: IntArray,
    loss: str,
) -> Any:
    """Wraps rows and labels into a CatBoost pool, with groups for ranking."""
    group_ids = None if loss == POINTWISE else groups
    return catboost_library.Pool(
        frame,
        label=labels.astype(int),
        cat_features=categorical,
        group_id=group_ids,
    )


def build_model(settings: CatBoostSettings, loss: str) -> Any:
    """Creates an untrained classifier, or a ranker for listwise losses."""
    pointwise = loss == POINTWISE
    model_class = (
        catboost_library.CatBoostClassifier
        if pointwise
        else catboost_library.CatBoostRanker
    )
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
        eval_metric="AUC" if pointwise else None,
    )


def train_model(
    train: Any,
    stopping: Any,
    settings: CatBoostSettings,
    loss: str,
) -> Any:
    """Trains a model and keeps its best iteration on the stopping set."""
    model = build_model(settings, loss)
    model.fit(train, eval_set=stopping, use_best_model=True)
    return model


def predict_scores(
    model: Any,
    frame: pd.DataFrame,
    categorical: list[str],
    loss: str,
) -> FloatArray:
    """Predicts a booking probability, or a ranking score for listwise models."""
    pool = catboost_library.Pool(frame, cat_features=categorical)
    if loss == POINTWISE:
        return model.predict_proba(pool)[:, 1]
    return model.predict(pool)
