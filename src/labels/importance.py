"""Measures from the true formulas how much each taste matters for booking."""

import numpy as np
import pandas as pd

from src.behavior.taste_terms import encode_feature
from src.behavior.utility import compute_outside_utilities, compute_utilities
from src.matching.stated import build_phrase_book, build_statements
from src.modeling.truth import chunk_rows, integrate_noise, read_constant
from src.settings.loader import Settings
from src.storage import load_text_jobs, results_directory, world_directory
from src.world.columns import TASTE_PREFIX
from src.world.helpers import read_floats, read_integers
from src.world.models import BoolArray, FloatArray

IMPORTANCE_FILE = "label_importance.csv"
MEASURES: list[str] = [
    "choice_shift",
    "session_spread",
    "mention_rate",
    "choice_shift_when_stated",
    "visible_choice_shift",
]


def probabilities(
    rows: pd.DataFrame,
    constant: float,
    settings: Settings,
) -> FloatArray:
    """Computes the true booking probabilities with a fixed noise sample."""
    pool = settings.pool.size
    margin = read_floats(rows, "margin_percentage")
    utilities = compute_utilities(rows, settings, margin, False).reshape(-1, pool)
    outside = compute_outside_utilities(rows.iloc[::pool], settings, constant)
    generator = np.random.default_rng(settings.modeling.truth.seed)
    return integrate_noise(outside, utilities, settings, generator)


def without_taste(rows: pd.DataFrame, name: str) -> pd.DataFrame:
    """Returns the rows with one taste switched off for every traveler."""
    changed = rows.copy()
    changed[TASTE_PREFIX + name] = 0.0
    return changed


def choice_shift(full: FloatArray, reduced: FloatArray, mask: BoolArray) -> float:
    """Averages the change of the booking distribution over chosen sessions."""
    distance = 0.5 * np.abs(full - reduced).sum(axis=1)
    return float(distance[mask].mean()) if mask.any() else 0.0


def stated_sessions(rows: pd.DataFrame, settings: Settings) -> FloatArray:
    """Marks, for every session of the rows, which tastes its brief states."""
    jobs = load_text_jobs(world_directory(settings))
    book = build_phrase_book(settings)
    count = settings.run.session_count
    briefs = (jobs.intents, jobs.practical_intents)
    statements = build_statements(briefs, count, book, settings)
    sessions = read_integers(rows.iloc[:: settings.pool.size], "session_id")
    return statements.tastes[sessions]


def measure_taste(
    name: str,
    index: int,
    context: tuple[pd.DataFrame, FloatArray, float],
    stated: FloatArray,
    settings: Settings,
) -> dict[str, float | str]:
    """Computes the importance measures of one taste."""
    rows, full, constant = context
    record = settings.amenities.tastes[index]
    reduced = probabilities(without_taste(rows, name), constant, settings)
    mentioned = stated[:, index] > 0
    everyone = np.ones(len(full), dtype=np.bool_)
    presence = encode_feature(rows, record).reshape(-1, settings.pool.size)
    weight = read_floats(rows, TASTE_PREFIX + name)[:: settings.pool.size]
    rate = float(mentioned.mean())
    when_stated = choice_shift(full, reduced, mentioned)
    return {
        "taste": name,
        "choice_shift": choice_shift(full, reduced, everyone),
        "session_spread": float(np.mean(np.abs(weight) * presence.std(axis=1))),
        "mention_rate": rate,
        "choice_shift_when_stated": when_stated,
        "visible_choice_shift": rate * when_stated,
    }


def rank_tastes(table: pd.DataFrame) -> pd.DataFrame:
    """Adds, for every measure, the rank of each taste (1 is the most important)."""
    ranked = table.copy()
    for measure in MEASURES:
        ranked[f"rank_{measure}"] = ranked[measure].rank(ascending=False).astype(int)
    return ranked.sort_values("choice_shift", ascending=False)


def run_importance(settings: Settings) -> pd.DataFrame:
    """Measures and ranks the importance of all 14 tastes on a session sample."""
    count = settings.matching.reference_sessions
    rows = chunk_rows(settings, 0, count)
    constant = read_constant(settings)
    full = probabilities(rows, constant, settings)
    stated = stated_sessions(rows, settings)
    records: list[dict[str, float | str]] = []
    for index, record in enumerate(settings.amenities.tastes):
        context = (rows, full, constant)
        records.append(measure_taste(record.name, index, context, stated, settings))
    table = rank_tastes(pd.DataFrame(records))
    output = results_directory(settings)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / IMPORTANCE_FILE, index=False)
    return table
