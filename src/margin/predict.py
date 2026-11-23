"""Predicts booking probabilities of every package at every margin of the grid."""

from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from src.features.tabular import compute_price_features
from src.margin.sample import MarginSample
from src.modeling.training import predict_scores
from src.settings.loader import Settings
from src.storage import models_directory
from src.world.models import FloatArray, IntArray

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


def load_fold_model(settings: Settings, fold: int) -> CatBoostClassifier:
    """Loads the saved model of one fold."""
    name = settings.modeling.margin.experiment
    path = models_directory(settings) / name / f"fold_{fold}.cbm"
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def fold_row_index(chosen: IntArray, pool: int) -> IntArray:
    """Lists the rows of the sessions scored by one fold model."""
    return (chosen[:, None] * pool + np.arange(pool)[None, :]).reshape(-1)


def predict_fold(
    sample: MarginSample,
    fold: int,
    result: FloatArray,
    settings: Settings,
) -> None:
    """Fills the probabilities of the sessions held out by one fold model."""
    pool = settings.pool.size
    chosen = np.flatnonzero(sample.folds == fold)
    index = fold_row_index(chosen, pool)
    model = load_fold_model(settings, fold)
    features = sample.features.iloc[index]
    rows = sample.rows.iloc[index]
    for position, margin in enumerate(settings.run.margin_grid):
        frame = priced_frame(features, rows, margin)
        scores = predict_scores(model, frame, sample.categorical)
        result[chosen, :, position] = scores.reshape(-1, pool)


def predict_grid(sample: MarginSample, settings: Settings) -> FloatArray:
    """Predicts out-of-fold booking probabilities for every package and margin."""
    shape = (sample.sessions, settings.pool.size, len(settings.run.margin_grid))
    result = np.zeros(shape)
    for fold in np.unique(sample.folds):
        predict_fold(sample, int(fold), result, settings)
    return result
