"""Loads the out-of-fold predictions and labels used to study calibration."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.build import FEATURE_FILE
from src.modeling.persistence import SCORES_SUFFIX
from src.modeling.pipeline import experiment_folder, load_split_masks
from src.settings.loader import Settings
from src.storage import DATASET_FILE, dataset_directory, features_directory
from src.world.helpers import read_floats, read_integers
from src.world.models import BoolArray, FloatArray, IntArray

FEATURE_COLUMNS: list[str] = ["n_children", "price_ratio_vs_competitor"]


@dataclass
class CalibrationData:
    predictions: FloatArray
    labels: BoolArray
    fold: IntArray
    cold: BoolArray
    children: IntArray
    price_ratio: FloatArray
    rows: IntArray


def load_calibration_data(settings: Settings) -> CalibrationData:
    """Reads the development rows with their out-of-fold predictions."""
    modeling = settings.modeling
    masks = load_split_masks(settings)
    development = masks.development
    scores_path = experiment_folder(settings) / (
        f"{modeling.calibration.experiment}{SCORES_SUFFIX}"
    )
    scores = np.load(scores_path).astype(np.float64)
    path = dataset_directory(settings) / DATASET_FILE
    labels = pd.read_parquet(path, columns=[modeling.label_column])
    table = pd.read_parquet(
        features_directory(settings) / FEATURE_FILE, columns=FEATURE_COLUMNS
    )
    return CalibrationData(
        predictions=scores[development],
        labels=labels[modeling.label_column].to_numpy(dtype=np.bool_)[development],
        fold=masks.fold[development],
        cold=masks.cold[development],
        children=read_integers(table, "n_children")[development],
        price_ratio=read_floats(table, "price_ratio_vs_competitor")[development],
        rows=np.flatnonzero(development),
    )
