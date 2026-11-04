"""Loads the development sessions on which margin policies are compared."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.explain.shap_values import read_manifest
from src.features.build import FEATURE_FILE
from src.modeling.pipeline import load_split_masks
from src.modeling.truth import chunk_rows
from src.settings.loader import Settings
from src.storage import features_directory
from src.world.models import IntArray


@dataclass
class MarginSample:
    rows: pd.DataFrame
    features: pd.DataFrame
    categorical: list[str]
    folds: IntArray
    sessions: int


def sample_row_index(sessions: IntArray, pool_size: int) -> IntArray:
    """Lists the row numbers of all packages of the given sessions."""
    return (sessions[:, None] * pool_size + np.arange(pool_size)[None, :]).reshape(-1)


def load_margin_sample(settings: Settings) -> MarginSample:
    """Reads the first development sessions with their rows and model inputs."""
    options = settings.modeling.margin
    pool = settings.pool.size
    masks = load_split_masks(settings)
    rows = chunk_rows(settings, 0, options.window_sessions)
    count = len(rows)
    development = masks.development[:count].reshape(-1, pool)[:, 0]
    sessions = np.flatnonzero(development)[: options.sample_sessions]
    columns, categorical = read_manifest(settings)
    path = features_directory(settings) / FEATURE_FILE
    window = [("session_id", "<", options.window_sessions)]
    table = pd.read_parquet(path, columns=columns, filters=window)
    index = sample_row_index(sessions, pool)
    return MarginSample(
        rows=rows.iloc[index].reset_index(drop=True),
        features=table.iloc[index].reset_index(drop=True),
        categorical=categorical,
        folds=masks.fold[:count].reshape(-1, pool)[:, 0][sessions],
        sessions=len(sessions),
    )
