"""Saves the trained models, their inputs and their out-of-fold scores."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.storage import models_directory, results_directory
from src.world.models import FloatArray

MANIFEST_FILE = "manifest.json"
SCORES_SUFFIX = "_scores.npy"
FIT_SUFFIX = "_fit.csv"


def experiment_models_folder(settings: Settings, name: str) -> Path:
    """Returns the folder that holds the models of one experiment."""
    folder = models_directory(settings) / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_fold_model(model: Any, folder: Path, fold: int) -> None:
    """Writes the trained model of one fold in the CatBoost format."""
    model.save_model(str(folder / f"fold_{fold}.cbm"))


def save_manifest(
    folder: Path,
    columns: list[str],
    categorical: list[str],
    loss: str,
) -> None:
    """Writes which inputs a saved model expects and how it was trained."""
    manifest = {"features": columns, "categorical": categorical, "loss": loss}
    text = json.dumps(manifest, indent=2)
    (folder / MANIFEST_FILE).write_text(text, encoding="utf-8")


def save_fit_diagnostics(
    settings: Settings,
    name: str,
    diagnostics: list[dict[str, float]],
) -> None:
    """Writes the per-fold training, stopping and held-out quality."""
    folder = results_directory(settings) / "experiments"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(folder / f"{name}{FIT_SUFFIX}", index=False)


def save_scores(settings: Settings, name: str, scores: FloatArray) -> None:
    """Writes the out-of-fold scores of every row of an experiment."""
    folder = results_directory(settings) / "experiments"
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"{name}{SCORES_SUFFIX}", scores.astype(np.float32))
