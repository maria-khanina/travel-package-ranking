"""Computes and saves the open match and stated match scores of the dataset."""

import numpy as np
import pandas as pd

from src.behavior.taste_terms import compute_taste_term
from src.behavior.terms import compute_fit_term
from src.features.evaluation import sessions_correlation
from src.matching.stated import (
    build_characteristics,
    build_phrase_book,
    build_statements,
    build_visibility,
    stated_scores,
)
from src.settings.loader import Settings
from src.storage import (
    DATASET_FILE,
    MATCH_FILE,
    TRUTH_FILE,
    dataset_directory,
    load_text_jobs,
    results_directory,
    world_directory,
)
from src.world.columns import TASTE_PREFIX
from src.world.helpers import read_floats, read_integers
from src.world.models import FloatArray

REFERENCE_FILE = "match_reference.csv"
ASPECT_FIT = "hid_aspect_fit"
OPEN_MATCH = "hid_open_match"
STATED_MATCH = "hid_stated_match"
REGISTER = "hid_intent_register"


def open_columns(settings: Settings) -> list[str]:
    """Lists the open dataset columns behind the characteristics people state."""
    names = [record.column for record in settings.amenities.tastes]
    return list(dict.fromkeys(names))


def hidden_columns(settings: Settings) -> list[str]:
    """Lists the hidden columns needed for the open match."""
    tastes = [TASTE_PREFIX + record.name for record in settings.amenities.tastes]
    return ["hid_fit"] + tastes + [REGISTER]


def load_rows(settings: Settings) -> pd.DataFrame:
    """Joins the open and the hidden columns needed for the match scores."""
    directory = dataset_directory(settings)
    identifiers = ["session_id", "room_id", "hotel_id"]
    public = pd.read_parquet(
        directory / DATASET_FILE, columns=identifiers + open_columns(settings)
    )
    hidden = pd.read_parquet(directory / TRUTH_FILE, columns=hidden_columns(settings))
    return pd.concat([public, hidden], axis=1)


def compute_open_match(rows: pd.DataFrame, settings: Settings) -> FloatArray:
    """Adds the aspect part and the taste part of the traveler's utility."""
    return compute_fit_term(rows, settings) + compute_taste_term(rows, settings)


def compute_stated_match(rows: pd.DataFrame, settings: Settings) -> FloatArray:
    """Scores the rows by what the briefs state and what the packages show."""
    jobs = load_text_jobs(world_directory(settings))
    book = build_phrase_book(settings)
    sessions = read_integers(rows, "session_id")
    statements = build_statements(
        (jobs.intents, jobs.practical_intents),
        int(sessions.max()) + 1,
        book,
        settings,
    )
    visibility = build_visibility(jobs.hotels, book, settings)
    return stated_scores(
        sessions,
        read_integers(rows, "hotel_id"),
        build_characteristics(rows, settings),
        (statements, visibility),
        settings.matching.chunk_rows,
    )


def build_match_table(rows: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Builds the table with the three match scores of every row."""
    return pd.DataFrame(
        {
            "session_id": rows["session_id"],
            "room_id": rows["room_id"],
            ASPECT_FIT: read_floats(rows, "hid_fit"),
            OPEN_MATCH: compute_open_match(rows, settings),
            STATED_MATCH: compute_stated_match(rows, settings),
        }
    )


def sample_rows(
    table: pd.DataFrame,
    rows: pd.DataFrame,
    settings: Settings,
) -> pd.Series:
    """Marks the rows of a reproducible random sample of sessions."""
    matching = settings.matching
    generator = np.random.default_rng(matching.reference_seed)
    sessions = table["session_id"].unique()
    chosen = generator.choice(sessions, matching.reference_sessions, replace=False)
    return rows["session_id"].isin(chosen)


PAIRS: list[tuple[str, str]] = [
    (OPEN_MATCH, STATED_MATCH),
    (OPEN_MATCH, ASPECT_FIT),
    (STATED_MATCH, ASPECT_FIT),
]
GROUPS: list[str] = ["all", "practical", "vibe"]


def correlate_group(
    sample: pd.DataFrame,
    registers: pd.Series,
    group: str,
    pair: tuple[str, str],
) -> float:
    """Correlates two scores inside sessions for one register group."""
    part = sample if group == "all" else sample[(registers == group).to_numpy()]
    first, second = read_floats(part, pair[0]), read_floats(part, pair[1])
    return sessions_correlation(first, second, part["session_id"])


def reference_table(
    table: pd.DataFrame,
    rows: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Correlates the three scores with each other inside sessions."""
    mask = sample_rows(table, rows, settings)
    sample = table[mask]
    registers = rows.loc[mask, REGISTER].astype(str)
    records: list[dict[str, float | str]] = []
    for pair in PAIRS:
        record: dict[str, float | str] = {"first": pair[0], "second": pair[1]}
        for group in GROUPS:
            record[group] = correlate_group(sample, registers, group, pair)
        records.append(record)
    return pd.DataFrame(records)


def run_matches(settings: Settings) -> pd.DataFrame:
    """Computes the match scores, saves them and reports their correlations."""
    rows = load_rows(settings)
    table = build_match_table(rows, settings)
    table.to_parquet(dataset_directory(settings) / MATCH_FILE, index=False)
    reference = reference_table(table, rows, settings)
    output = results_directory(settings)
    output.mkdir(parents=True, exist_ok=True)
    reference.to_csv(output / REFERENCE_FILE, index=False)
    return reference
