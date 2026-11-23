"""Loads the saved tables and runs the modeling stage."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build import FEATURE_FILE
from src.modeling.bootstrap import bootstrap_table, paired_table
from src.modeling.experiments import (
    ORACLE_NAME,
    RANDOM_NAME,
    ModelInputs,
    ResultRow,
    reference_rows,
    run_experiment,
)
from src.modeling.split import SplitMasks, build_masks
from src.modeling.weights import compute_propensity_weights
from src.settings.loader import Settings
from src.storage import (
    DATASET_FILE,
    TRUTH_FILE,
    dataset_directory,
    features_directory,
    results_directory,
)
from src.world.helpers import read_floats, read_integers
from src.world.models import BoolArray, FloatArray

EXPERIMENT_FOLDER = "experiments"
SPLIT_FILE = "split_summary.csv"
CROSS_VALIDATION_FILE = "cross_validation.csv"
INTERVAL_FILE = "bootstrap_intervals.csv"
PAIRED_FILE = "bootstrap_paired.csv"
RANKS_SUFFIX = "_ranks.npy"
ROW_SUFFIX = "_row.csv"
HIDDEN_FIT = "hid_fit"
MARGIN_COLUMN = "margin_percentage"
PROPENSITY_COLUMN = "margin_propensity"


def describe_split(
    dataset: pd.DataFrame,
    masks: SplitMasks,
    label_column: str,
) -> pd.DataFrame:
    """Counts the rows, sessions and bookings in every part of the split."""
    parts: dict[str, BoolArray] = {
        "development": masks.development,
        "test": masks.test,
    }
    rows: list[dict[str, float | str]] = []
    for name, mask in parts.items():
        part = dataset[mask]
        rows.append(
            {
                "part": name,
                "rows": float(len(part)),
                "sessions": float(part["session_id"].nunique()),
                "bookings": float(part[label_column].sum()),
                "cold_hotel_rows": float(masks.cold[mask].sum()),
            }
        )
    return pd.DataFrame(rows)


def read_tables(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reads the needed columns of the dataset, the truth and the features."""
    modeling = settings.modeling
    directory = dataset_directory(settings)
    columns: list[str] = [
        "session_id",
        modeling.hotel_column,
        modeling.label_column,
        MARGIN_COLUMN,
        PROPENSITY_COLUMN,
    ]
    truth_columns: list[str] = [HIDDEN_FIT, modeling.register_column]
    return (
        pd.read_parquet(directory / DATASET_FILE, columns=columns),
        pd.read_parquet(directory / TRUTH_FILE, columns=truth_columns),
        pd.read_parquet(features_directory(settings) / FEATURE_FILE),
    )


def save_split_summary(
    dataset: pd.DataFrame,
    masks: SplitMasks,
    settings: Settings,
) -> None:
    """Writes the size of every part of the split and prints it."""
    output = results_directory(settings)
    output.mkdir(parents=True, exist_ok=True)
    summary = describe_split(dataset, masks, settings.modeling.label_column)
    summary.to_csv(output / SPLIT_FILE, index=False)
    print(summary.to_string(index=False))


def load_split_masks(settings: Settings) -> SplitMasks:
    """Rebuilds the row masks of the split from the saved dataset."""
    modeling = settings.modeling
    columns: list[str] = ["session_id", modeling.hotel_column, modeling.label_column]
    directory = dataset_directory(settings)
    dataset = pd.read_parquet(directory / DATASET_FILE, columns=columns)
    truth = pd.read_parquet(directory / TRUTH_FILE, columns=[modeling.register_column])
    return build_masks(
        dataset,
        truth[modeling.register_column],
        modeling.split,
        modeling.label_column,
        modeling.hotel_column,
    )


def load_inputs(settings: Settings) -> ModelInputs:
    """Reads the saved tables and builds the inputs shared by all experiments."""
    modeling = settings.modeling
    dataset, truth, table = read_tables(settings)
    masks = build_masks(
        dataset,
        truth[modeling.register_column],
        modeling.split,
        modeling.label_column,
        modeling.hotel_column,
    )
    save_split_summary(dataset, masks, settings)
    return ModelInputs(
        table=table,
        labels=dataset[modeling.label_column].to_numpy(dtype=np.bool_),
        sessions=read_integers(dataset, "session_id"),
        hotels=dataset[modeling.hotel_column],
        hidden_fit=read_floats(truth, HIDDEN_FIT),
        masks=masks,
        weights=compute_propensity_weights(
            read_floats(dataset, MARGIN_COLUMN),
            read_floats(dataset, PROPENSITY_COLUMN),
        ),
    )


def experiment_folder(settings: Settings) -> Path:
    """Returns the folder where the result of every experiment is saved."""
    folder = results_directory(settings) / EXPERIMENT_FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_experiment(
    folder: Path,
    name: str,
    row: ResultRow,
    ranks: FloatArray,
) -> None:
    """Saves the metrics and the booked ranks of one experiment."""
    pd.DataFrame([row]).to_csv(folder / f"{name}{ROW_SUFFIX}", index=False)
    np.save(folder / f"{name}{RANKS_SUFFIX}", ranks)


def save_references(
    folder: Path,
    inputs: ModelInputs,
    settings: Settings,
) -> None:
    """Saves the random and hidden-fit reference rows."""
    (oracle, chance), ranks = reference_rows(inputs, settings)
    save_experiment(folder, ORACLE_NAME, oracle, ranks[ORACLE_NAME])
    pd.DataFrame([chance]).to_csv(folder / f"{RANDOM_NAME}{ROW_SUFFIX}", index=False)


def train_pending(
    inputs: ModelInputs,
    folder: Path,
    only: list[str],
    settings: Settings,
) -> None:
    """Cross-validates the experiments that have no saved result yet."""
    for experiment in settings.modeling.experiments:
        done = (folder / f"{experiment.name}{ROW_SUFFIX}").exists()
        if done or (only and experiment.name not in only):
            continue
        try:
            row, ranks = run_experiment(experiment, inputs, settings)
        except Exception as error:
            print(experiment.name, "failed:", error, flush=True)
            continue
        save_experiment(folder, experiment.name, row, ranks)
        print(experiment.name, row["auc"], row["hit_rate_at_3"], flush=True)


def run_cross_validation(settings: Settings, only: list[str]) -> pd.DataFrame:
    """Cross-validates the pending experiments and gathers all results."""
    inputs = load_inputs(settings)
    folder = experiment_folder(settings)
    save_references(folder, inputs, settings)
    train_pending(inputs, folder, only, settings)
    return collect_results(settings)


def collect_results(settings: Settings) -> pd.DataFrame:
    """Gathers the saved metrics of all experiments into one table."""
    folder = experiment_folder(settings)
    frames: list[pd.DataFrame] = [
        pd.read_csv(path) for path in sorted(folder.glob(f"*{ROW_SUFFIX}"))
    ]
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(results_directory(settings) / CROSS_VALIDATION_FILE, index=False)
    return results


def load_ranks(settings: Settings) -> dict[str, FloatArray]:
    """Reads the saved booked-session ranks of all trained experiments."""
    folder = experiment_folder(settings)
    ranks: dict[str, FloatArray] = {}
    for path in sorted(folder.glob(f"*{RANKS_SUFFIX}")):
        ranks[path.name.removesuffix(RANKS_SUFFIX)] = np.load(path)
    return ranks


def run_bootstrap(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes confidence intervals and paired differences from saved ranks."""
    modeling = settings.modeling
    ranks = load_ranks(settings)
    pool = modeling.split.pool_size
    intervals = bootstrap_table(ranks, pool, modeling.top_k_values, modeling.bootstrap)
    paired = paired_table(ranks, pool, modeling.top_k_values, modeling.bootstrap)
    output = results_directory(settings)
    intervals.to_csv(output / INTERVAL_FILE, index=False)
    paired.to_csv(output / PAIRED_FILE, index=False)
    return intervals, paired
