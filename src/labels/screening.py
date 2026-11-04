"""Tests which query labels really change what travelers book."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.behavior.taste_terms import encode_feature
from src.features.build import FEATURE_FILE
from src.features.extras import LABEL_PREFIX
from src.modeling.ranking import locate_booked
from src.modeling.split import build_masks
from src.settings.loader import Settings
from src.settings.screening_models import ScreeningSettings
from src.storage import (
    DATASET_FILE,
    TRUTH_FILE,
    dataset_directory,
    features_directory,
    results_directory,
)
from src.world.columns import TASTE_PREFIX
from src.world.helpers import read_floats
from src.world.models import BoolArray, FloatArray, IntArray

REGISTER = "hid_intent_register"
SUMMARY_FILE = "label_screening_summary.csv"
GRADED_FILE = "label_screening_graded.csv"


@dataclass
class TruthMeasures:
    weight_spread: float
    session_effect: float
    visible_effect: float


@dataclass
class LabelEvidence:
    similarity: FloatArray
    difference: FloatArray
    truth: TruthMeasures


@dataclass
class BookedSessions:
    fold: IntArray
    evidence: dict[str, LabelEvidence]


def standardize(values: FloatArray) -> FloatArray:
    """Centers values and divides them by their spread."""
    return (values - values.mean()) / values.std()


def null_exceedances(
    similarity: FloatArray,
    difference: FloatArray,
    observed: float,
    settings: ScreeningSettings,
) -> int:
    """Counts the random re-pairings that co-move at least as strongly."""
    generator = np.random.default_rng(settings.seed)
    exceed = 0
    for start in range(0, settings.permutations, settings.permutation_chunk):
        size = min(settings.permutation_chunk, settings.permutations - start)
        shuffled = generator.permuted(np.tile(similarity, (size, 1)), axis=1)
        null = (shuffled * difference).mean(axis=1)
        exceed += int(np.sum(np.abs(null) >= abs(observed)))
    return exceed


def permutation_test(
    similarity: FloatArray,
    difference: FloatArray,
    settings: ScreeningSettings,
) -> tuple[float, float]:
    """Compares the observed co-movement with random re-pairings of sessions."""
    observed = float(np.mean(similarity * difference))
    exceed = null_exceedances(similarity, difference, observed, settings)
    return observed, (1.0 + exceed) / (settings.permutations + 1.0)


def benjamini_hochberg(p_values: FloatArray) -> FloatArray:
    """Converts p-values to false discovery rate q-values."""
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / (np.arange(len(p_values)) + 1.0)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(len(p_values))
    result[order] = np.minimum(adjusted, 1.0)
    return result


Tables = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]


def read_tables(settings: Settings) -> Tables:
    """Reads the open columns, the hidden columns and the label similarities."""
    directory = dataset_directory(settings)
    labels = settings.features.label_characteristics
    modeling = settings.modeling
    columns = ["session_id", modeling.hotel_column, modeling.label_column]
    columns += list(dict.fromkeys(record.column for record in labels.values()))
    tastes = [TASTE_PREFIX + name for name in settings.screening.truth_taste.values()]
    similarity = [f"{LABEL_PREFIX}{name}" for name in labels]
    feature_path = features_directory(settings) / FEATURE_FILE
    return (
        pd.read_parquet(directory / DATASET_FILE, columns=columns),
        pd.read_parquet(directory / TRUTH_FILE, columns=[REGISTER] + tastes),
        pd.read_parquet(feature_path, columns=similarity),
    )


def mention_probability(name: str, settings: Settings) -> float:
    """Gives how often a message spells out the taste behind a label."""
    taste = settings.screening.truth_taste.get(name)
    for record in settings.amenities.mentions.details:
        if record.taste == taste:
            return record.probability
    return 0.0


def truth_measures(
    name: str,
    presence: FloatArray,
    weight: FloatArray | None,
    settings: Settings,
) -> TruthMeasures:
    """Measures how important a label truly is, using the hidden tastes."""
    if weight is None:
        return TruthMeasures(0.0, 0.0, 0.0)
    effect = float(np.mean(weight**2 * presence.var(axis=1)))
    visible = effect * mention_probability(name, settings)
    return TruthMeasures(float(weight.std()), effect, visible)


def read_weight(
    truth: pd.DataFrame,
    taste: str | None,
    development: IntArray,
    pool: int,
) -> FloatArray | None:
    """Reads the hidden taste weight of every development session."""
    if taste is None:
        return None
    weights = read_floats(truth, TASTE_PREFIX + taste)
    return weights[development][::pool]


def build_evidence(
    name: str,
    tables: Tables,
    booked: tuple[IntArray, IntArray, IntArray],
    settings: Settings,
) -> LabelEvidence:
    """Collects, for one label, what the booked sessions show about it."""
    dataset, truth, similarity = tables
    development, sessions, positions = booked
    pool = settings.modeling.split.pool_size
    record = settings.features.label_characteristics[name]
    presence = encode_feature(dataset.iloc[development], record).reshape(-1, pool)
    first = read_floats(similarity, f"{LABEL_PREFIX}{name}")
    taste = settings.screening.truth_taste.get(name)
    weight = read_weight(truth, taste, development, pool)
    difference = presence[sessions, positions] - presence[sessions].mean(axis=1)
    z = standardize(first[development][::pool])
    measures = truth_measures(name, presence, weight, settings)
    return LabelEvidence(z[sessions], difference, measures)


def prepare_sessions(settings: Settings) -> BookedSessions:
    """Builds the label evidence of every booked development session."""
    modeling = settings.modeling
    tables = read_tables(settings)
    dataset, truth, _ = tables
    masks = build_masks(
        dataset,
        truth[REGISTER],
        modeling.split,
        modeling.label_column,
        modeling.hotel_column,
    )
    development = np.flatnonzero(masks.development)
    pool = modeling.split.pool_size
    bookings = dataset[modeling.label_column].to_numpy(dtype=np.bool_)[development]
    sessions, positions = locate_booked(bookings, pool)
    evidence: dict[str, LabelEvidence] = {}
    for name in settings.features.label_characteristics:
        booked = (development, sessions, positions)
        evidence[name] = build_evidence(name, tables, booked, settings)
    return BookedSessions(masks.fold[development][::pool][sessions], evidence)


def test_folds(
    evidence: LabelEvidence,
    fold: IntArray,
    folds: list[int],
    settings: ScreeningSettings,
) -> tuple[float, float]:
    """Runs the permutation test on the booked sessions of some folds."""
    mask: BoolArray = np.isin(fold, folds)
    similarity, difference = evidence.similarity[mask], evidence.difference[mask]
    return permutation_test(similarity, difference, settings)


def screen_label(
    name: str,
    booked: BookedSessions,
    settings: ScreeningSettings,
) -> dict[str, float | str | bool | None]:
    """Tests one label on the selection folds and on the confirmation folds."""
    evidence = booked.evidence[name]
    selection = test_folds(evidence, booked.fold, settings.selection_folds, settings)
    confirmation = test_folds(
        evidence, booked.fold, settings.confirmation_folds, settings
    )
    return {
        "label": name,
        "statistic": selection[0],
        "p_value": selection[1],
        "confirmation_statistic": confirmation[0],
        "confirmation_p_value": confirmation[1],
        "weight_spread": evidence.truth.weight_spread,
        "session_effect": evidence.truth.session_effect,
        "visible_effect": evidence.truth.visible_effect,
        "is_true_taste": evidence.truth.weight_spread > 0.0,
    }


def screen_labels(booked: BookedSessions, settings: ScreeningSettings) -> pd.DataFrame:
    """Tests every label, corrects for many tests and marks the chosen ones."""
    records = [screen_label(name, booked, settings) for name in booked.evidence]
    table = pd.DataFrame(records)
    table["q_value"] = benjamini_hochberg(table["p_value"].to_numpy(dtype=np.float64))
    table["selected"] = table["q_value"] <= settings.false_discovery_rate
    same_sign = table["statistic"] * table["confirmation_statistic"] > 0
    significant = table["confirmation_p_value"] <= settings.confirmation_level
    table["confirmed"] = table["selected"] & same_sign & significant
    return table


def selection_quality(table: pd.DataFrame, column: str) -> dict[str, float | str]:
    """Compares the chosen labels with the labels that truly have a taste."""
    chosen = table[column].to_numpy(dtype=np.bool_)
    truth = table["is_true_taste"].to_numpy(dtype=np.bool_)
    hits = int(np.logical_and(chosen, truth).sum())
    false_hits = int(np.logical_and(chosen, ~truth).sum())
    return {
        "selection": column,
        "chosen": float(chosen.sum()),
        "true_tastes": float(truth.sum()),
        "precision": hits / max(1, hits + false_hits),
        "recall": hits / max(1, int(truth.sum())),
        "false_discovery_share": false_hits / max(1, hits + false_hits),
    }


MEASURES: list[str] = ["weight_spread", "session_effect", "visible_effect"]


def ordering_gain(table: pd.DataFrame, measure: str, top: int) -> float:
    """Scores how much true importance the test puts into its first labels."""
    ordered = table.sort_values("p_value")[measure].to_numpy(dtype=np.float64)
    ideal = np.sort(table[measure].to_numpy(dtype=np.float64))[::-1]
    discount = 1.0 / np.log2(np.arange(top) + 2.0)
    return float((ordered[:top] * discount).sum() / (ideal[:top] * discount).sum())


def ranking_quality(table: pd.DataFrame, top: int) -> pd.DataFrame:
    """Compares the order given by the test with each measure of true importance."""
    strength = pd.Series(-np.log10(table["p_value"].to_numpy(dtype=np.float64)))
    rows: list[dict[str, float | str]] = []
    for measure in MEASURES:
        truth = pd.Series(table[measure].to_numpy(dtype=np.float64))
        correlation = float(strength.corr(truth, method="spearman"))
        rows.append(
            {
                "true_importance": measure,
                "spearman_with_significance": correlation,
                "ndcg_of_test_order": ordering_gain(table, measure, top),
            }
        )
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, top: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds the binary selection quality and the graded ordering quality."""
    chosen = [selection_quality(table, name) for name in ("selected", "confirmed")]
    return pd.DataFrame(chosen), ranking_quality(table, top)


def run_screening(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Screens all labels and saves the per-label and the summary tables."""
    booked = prepare_sessions(settings)
    table = screen_labels(booked, settings.screening)
    binary, graded = summarize(table, settings.screening.top_labels)
    output = results_directory(settings)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / settings.screening.output_file, index=False)
    binary.to_csv(output / SUMMARY_FILE, index=False)
    graded.to_csv(output / GRADED_FILE, index=False)
    return table, graded
