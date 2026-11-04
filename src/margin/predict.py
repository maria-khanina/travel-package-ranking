"""Predicts booking probabilities of every package at every margin of the grid."""

import importlib
from typing import Any

import numpy as np
import pandas as pd

from src.features.tabular import compute_price_features
from src.margin.sample import MarginSample
from src.modeling.training import POINTWISE, predict_scores
from src.settings.loader import Settings
from src.storage import models_directory
from src.world.models import FloatArray

catboost_library: Any = importlib.import_module("catboost")
MARGIN_COLUMN = "margin_percentage"


def priced_frame(
    features: pd.DataFrame,
    rows: pd.DataFrame,
    margin: float,
) -> pd.DataFrame:
    """Rewrites the model inputs that depend on the margin."""
    frame = features.copy()
    values = np.full(len(rows), margin)
    for name, column in compute_price_features(rows, values).items():
        if name in frame.columns:
            frame[name] = column
    if MARGIN_COLUMN in frame.columns:
        frame[MARGIN_COLUMN] = values
    return frame


def load_fold_model(settings: Settings, fold: int) -> Any:
    """Loads the saved model of one fold."""
    name = settings.modeling.margin.experiment
    path = models_directory(settings) / name / f"fold_{fold}.cbm"
    model = catboost_library.CatBoostClassifier()
    model.load_model(str(path))
    return model


def predict_grid(sample: MarginSample, settings: Settings) -> FloatArray:
    """Predicts out-of-fold booking probabilities for every package and margin."""
    pool = settings.pool.size
    grid = settings.run.margin_grid
    result = np.zeros((sample.sessions, pool, len(grid)))
    for fold in np.unique(sample.folds):
        chosen = np.flatnonzero(sample.folds == fold)
        index = (chosen[:, None] * pool + np.arange(pool)[None, :]).reshape(-1)
        model = load_fold_model(settings, int(fold))
        for position, margin in enumerate(grid):
            frame = priced_frame(
                sample.features.iloc[index], sample.rows.iloc[index], margin
            )
            scores = predict_scores(model, frame, sample.categorical, POINTWISE)
            result[chosen, :, position] = scores.reshape(-1, pool)
    return result
