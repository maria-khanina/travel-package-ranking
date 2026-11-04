"""Reads and writes the generated tables and text files on disk."""

import json
from pathlib import Path

import pandas as pd

from src.dataset import DatasetBundle
from src.settings.loader import PROJECT_ROOT, Settings
from src.texts.models import JOB_NAMES, TextJobs
from src.world.models import WorldTables

WORLD_TABLES: list[str] = ["destinations", "hotels", "rooms", "sessions", "rows"]
FILE_SUFFIX = ".parquet"
JOBS_PREFIX = "jobs_"
DATASET_FILE = "dataset.parquet"
TRUTH_FILE = "truth.parquet"
INTENT_TEXT_FILE = "intent_texts.parquet"
HOTEL_TEXT_FILE = "hotel_texts.parquet"
ROOM_TEXT_FILE = "room_texts.parquet"
MATCH_FILE = "match_scores.parquet"
SIMULATION_FILE = "simulation.json"
TRUE_FILE = "true_probabilities.parquet"


def data_directory(settings: Settings) -> Path:
    """Returns the root directory of all generated data."""
    return PROJECT_ROOT / settings.run.paths.data_dir


def world_directory(settings: Settings) -> Path:
    """Returns the directory holding the world tables and text briefs."""
    return data_directory(settings) / settings.run.paths.world_dir


def texts_directory(settings: Settings) -> Path:
    """Returns the directory holding the generated texts."""
    return data_directory(settings) / settings.run.paths.texts_dir


def dataset_directory(settings: Settings) -> Path:
    """Returns the directory holding the final datasets."""
    return data_directory(settings) / settings.run.paths.dataset_dir


def features_directory(settings: Settings) -> Path:
    """Returns the directory holding cached vectors and feature tables."""
    return data_directory(settings) / settings.run.paths.features_dir


def results_directory(settings: Settings) -> Path:
    """Returns the directory holding evaluation tables."""
    return data_directory(settings) / settings.run.paths.results_dir


def models_directory(settings: Settings) -> Path:
    """Returns the directory holding trained models."""
    return data_directory(settings) / settings.run.paths.models_dir


def build_text_tables(public: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Collects each distinct text of the dataset once, with its identifier."""
    intents = public.drop_duplicates("session_id")[["session_id", "user_intent"]]
    hotels = public.drop_duplicates("hotel_id")[["hotel_id", "hotel_description"]]
    rooms = public.drop_duplicates("room_id")[["room_id", "room_description"]]
    return {
        INTENT_TEXT_FILE: intents,
        HOTEL_TEXT_FILE: hotels,
        ROOM_TEXT_FILE: rooms,
    }


def save_dataset(bundle: DatasetBundle, directory: Path) -> None:
    """Writes the public table, the hidden truth and the distinct texts."""
    directory.mkdir(parents=True, exist_ok=True)
    bundle.public.to_parquet(directory / DATASET_FILE, index=False)
    bundle.truth.to_parquet(directory / TRUTH_FILE, index=False)
    constant = {"outside_constant": bundle.outside_constant}
    (directory / SIMULATION_FILE).write_text(json.dumps(constant), encoding="utf-8")
    for name, table in build_text_tables(bundle.public).items():
        table.to_parquet(directory / name, index=False)


def load_text_jobs(directory: Path) -> TextJobs:
    """Reads only the text briefs from a world directory."""
    briefs: dict[str, pd.DataFrame] = {}
    for job in JOB_NAMES:
        briefs[job] = pd.read_parquet(directory / f"{JOBS_PREFIX}{job}{FILE_SUFFIX}")
    return TextJobs(**briefs)


def save_world(world: WorldTables, jobs: TextJobs, directory: Path) -> None:
    """Writes the world tables and the text briefs to a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in WORLD_TABLES:
        table: pd.DataFrame = getattr(world, name)
        table.to_parquet(directory / f"{name}{FILE_SUFFIX}", index=False)
    for job in JOB_NAMES:
        path = directory / f"{JOBS_PREFIX}{job}{FILE_SUFFIX}"
        jobs.get(job).to_parquet(path, index=False)


def load_world(directory: Path) -> tuple[WorldTables, TextJobs]:
    """Reads the world tables and the text briefs from a directory."""
    tables: dict[str, pd.DataFrame] = {}
    for name in WORLD_TABLES:
        tables[name] = pd.read_parquet(directory / f"{name}{FILE_SUFFIX}")
    briefs: dict[str, pd.DataFrame] = {}
    for job in JOB_NAMES:
        briefs[job] = pd.read_parquet(directory / f"{JOBS_PREFIX}{job}{FILE_SUFFIX}")
    return WorldTables(**tables), TextJobs(**briefs)
