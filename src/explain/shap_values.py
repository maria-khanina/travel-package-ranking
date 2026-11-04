"""Computes Shapley values of the saved fold models on held-out rows."""

import importlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.features.build import FEATURE_FILE
from src.modeling.pipeline import load_split_masks
from src.modeling.split import SplitMasks
from src.settings.loader import Settings
from src.storage import features_directory, models_directory
from src.world.models import FloatArray, IntArray

catboost_library: Any = importlib.import_module("catboost")
MANIFEST_FILE = "manifest.json"


@dataclass
class ShapResult:
    columns: list[str]
    contributions: FloatArray
    frame: pd.DataFrame
    categorical: list[str]


def read_manifest(settings: Settings) -> tuple[list[str], list[str]]:
    """Reads which inputs the saved models of the experiment expect."""
    name = settings.modeling.explain.experiment
    path = models_directory(settings) / name / MANIFEST_FILE
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return list(manifest["features"]), list(manifest["categorical"])


def sample_by_fold(masks: SplitMasks, settings: Settings) -> dict[int, IntArray]:
    """Draws random whole sessions from every fold and lists their rows."""
    options = settings.modeling.explain
    pool = settings.pool.size
    generator = np.random.default_rng(options.seed)
    firsts = masks.fold.reshape(-1, pool)[:, 0]
    chosen: dict[int, IntArray] = {}
    for fold in range(settings.modeling.split.fold_count):
        sessions = np.flatnonzero(firsts == fold)
        picked = generator.choice(sessions, options.sessions_per_fold, replace=False)
        rows = picked[:, None] * pool + np.arange(pool)[None, :]
        chosen[fold] = np.sort(rows.reshape(-1))
    return chosen


def fold_contributions(
    settings: Settings,
    fold: int,
    frame: pd.DataFrame,
    categorical: list[str],
) -> FloatArray:
    """Computes the Shapley values of one fold's model on its held-out rows."""
    name = settings.modeling.explain.experiment
    path = models_directory(settings) / name / f"fold_{fold}.cbm"
    model = catboost_library.CatBoostClassifier()
    model.load_model(str(path))
    pool = catboost_library.Pool(frame, cat_features=categorical)
    values = model.get_feature_importance(pool, type="ShapValues")
    return np.asarray(values[:, :-1], dtype=np.float64)


def compute_shap(settings: Settings) -> ShapResult:
    """Computes Shapley values for sampled out-of-fold rows of every fold."""
    columns, categorical = read_manifest(settings)
    samples = sample_by_fold(load_split_masks(settings), settings)
    path = features_directory(settings) / FEATURE_FILE
    table = pd.read_parquet(path, columns=columns)
    frames: list[pd.DataFrame] = []
    parts: list[FloatArray] = []
    for fold, rows in samples.items():
        frame = table.iloc[rows].reset_index(drop=True)
        parts.append(fold_contributions(settings, fold, frame, categorical))
        frames.append(frame)
    return ShapResult(columns, np.concatenate(parts), pd.concat(frames), categorical)
