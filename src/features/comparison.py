"""Compares text models by how well their scores follow the match scores."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.catalog import TextCatalog, build_text_catalog, cache_file, locate
from src.features.documents import TEXT_VARIANTS, Documents, build_documents, task_for
from src.features.encoders import CrossPairScorer, PairScorer, SentenceEncoder
from src.features.evaluation import sessions_correlation
from src.features.text_features import build_queries, encode_cached, row_cosine
from src.modeling.split import build_masks
from src.settings.feature_models import EmbeddingVariant
from src.settings.loader import Settings
from src.storage import (
    DATASET_FILE,
    HOTEL_TEXT_FILE,
    INTENT_TEXT_FILE,
    MATCH_FILE,
    ROOM_TEXT_FILE,
    TRUTH_FILE,
    dataset_directory,
    features_directory,
    results_directory,
)
from src.world.helpers import read_floats, read_integers
from src.world.models import BoolArray, FloatArray, IntArray

REGISTER = "hid_intent_register"
GROUPS: list[str] = ["all", "practical", "vibe"]
Record = dict[str, float | str]


@dataclass
class ComparisonInputs:
    rows: pd.DataFrame
    catalog: TextCatalog
    documents: dict[str, Documents]


def select_sessions(settings: Settings) -> IntArray:
    """Draws the random development sessions used for the comparison."""
    modeling = settings.modeling
    directory = dataset_directory(settings)
    columns = ["session_id", modeling.hotel_column, modeling.label_column]
    dataset = pd.read_parquet(directory / DATASET_FILE, columns=columns)
    truth = pd.read_parquet(directory / TRUTH_FILE, columns=[REGISTER])
    masks = build_masks(
        dataset,
        truth[REGISTER],
        modeling.split,
        modeling.label_column,
        modeling.hotel_column,
    )
    development = np.unique(read_integers(dataset, "session_id")[masks.development])
    generator = np.random.default_rng(settings.comparison.sample_seed)
    count = settings.comparison.sample_sessions
    return np.sort(generator.choice(development, size=count, replace=False))


def load_rows(settings: Settings, sessions: IntArray) -> pd.DataFrame:
    """Reads the rows, registers and match scores of the sampled sessions."""
    directory = dataset_directory(settings)
    columns = ["session_id", "room_id", "hotel_id", "city", "country"]
    rows = pd.read_parquet(directory / DATASET_FILE, columns=columns)
    registers = pd.read_parquet(directory / TRUTH_FILE, columns=[REGISTER])
    rows[REGISTER] = registers[REGISTER].to_numpy()
    scores = pd.read_parquet(directory / MATCH_FILE)
    for target in settings.comparison.targets:
        rows[target] = scores[target].to_numpy()
    return rows[rows["session_id"].isin(sessions)].reset_index(drop=True)


def load_catalog(settings: Settings, sessions: IntArray) -> TextCatalog:
    """Builds the text catalog with the intents of the sampled sessions only."""
    directory = dataset_directory(settings)
    intents = pd.read_parquet(directory / INTENT_TEXT_FILE)
    intents = intents[intents["session_id"].isin(sessions)]
    return build_text_catalog(
        intents,
        pd.read_parquet(directory / HOTEL_TEXT_FILE),
        pd.read_parquet(directory / ROOM_TEXT_FILE),
    )


def build_inputs(settings: Settings) -> ComparisonInputs:
    """Loads the sampled sessions, the texts and the five document variants."""
    sessions = select_sessions(settings)
    rows = load_rows(settings, sessions)
    catalog = load_catalog(settings, sessions)
    features = settings.features
    documents = build_documents(
        rows, catalog, features.document_separator, features.location_prefix
    )
    return ComparisonInputs(rows=rows, catalog=catalog, documents=documents)


def group_mask(registers: pd.Series, group: str) -> BoolArray:
    """Selects the rows of one register group, or all rows."""
    if group == "all":
        return np.ones(len(registers), dtype=np.bool_)
    return (registers == group).to_numpy()


def correlations(scores: FloatArray, rows: pd.DataFrame, target: str) -> Record:
    """Correlates a score with a target inside sessions, per register group."""
    values = read_floats(rows, target)
    registers = rows[REGISTER].astype(str)
    result: Record = {}
    for group in GROUPS:
        mask = group_mask(registers, group)
        sessions = rows["session_id"][mask]
        result[group] = sessions_correlation(scores[mask], values[mask], sessions)
    return result


def report_scores(
    model: str,
    text_variant: str,
    scores: FloatArray,
    rows: pd.DataFrame,
    settings: Settings,
) -> list[Record]:
    """Builds one report record per target for a model and a text variant."""
    records: list[Record] = []
    for target in settings.comparison.targets:
        base: Record = {"model": model, "text_variant": text_variant, "target": target}
        base.update(correlations(scores, rows, target))
        base["sessions"] = float(rows["session_id"].nunique())
        records.append(base)
    return records


def score_embedding(
    variant: EmbeddingVariant,
    name: str,
    encoder: SentenceEncoder,
    inputs: ComparisonInputs,
    settings: Settings,
) -> FloatArray:
    """Computes the cosine of every row's query and document for one variant."""
    directory = features_directory(settings) / settings.comparison.cache_folder
    length = settings.features.cache_hash_length
    texts = build_queries(variant, task_for(variant, name), inputs.catalog.intent_texts)
    documents = inputs.documents[name]
    queries = encode_cached(
        encoder, texts, cache_file(directory, variant.repository, texts, length)
    )
    vectors = encode_cached(
        encoder,
        documents.texts,
        cache_file(directory, variant.repository, documents.texts, length),
    )
    ids = read_integers(inputs.rows, "session_id")
    sessions = locate(inputs.catalog.intent_ids, ids)
    chunk = settings.features.chunk_rows
    return row_cosine(queries, sessions, vectors, documents.positions, chunk)


def evaluate_embedding(
    variant: EmbeddingVariant,
    inputs: ComparisonInputs,
    settings: Settings,
) -> list[Record]:
    """Reports the agreement of one embedding model for every text variant."""
    features = settings.features
    encoder = SentenceEncoder(variant, features.device, features.half_precision)
    records: list[Record] = []
    for name in TEXT_VARIANTS:
        scores = score_embedding(variant, name, encoder, inputs, settings)
        records += report_scores(variant.key, name, scores, inputs.rows, settings)
    return records


def reranker_scores(
    scorer: PairScorer,
    intents: list[str],
    sessions: IntArray,
    documents: Documents,
    positions: IntArray,
) -> FloatArray:
    """Scores the distinct query-document pairs and maps them back to rows."""
    pairs = np.unique(np.stack([sessions, positions], axis=1), axis=0)
    first: list[int] = pairs[:, 0].tolist()
    second: list[int] = pairs[:, 1].tolist()
    scores = scorer.score(
        [intents[at] for at in first], [documents.texts[at] for at in second]
    )
    width = len(documents.texts)
    keys = pairs[:, 0] * width + pairs[:, 1]
    return scores[np.searchsorted(keys, sessions * width + positions)]


def reranker_rows(
    inputs: ComparisonInputs,
    settings: Settings,
) -> tuple[pd.DataFrame, BoolArray]:
    """Takes the rows of the sessions on which the reranker is evaluated."""
    count = settings.comparison.reranker_sessions
    chosen = inputs.rows["session_id"].unique()[:count]
    mask = inputs.rows["session_id"].isin(chosen).to_numpy()
    return inputs.rows[mask].reset_index(drop=True), mask


def reranker_records(
    scorer: PairScorer,
    inputs: ComparisonInputs,
    subset: tuple[pd.DataFrame, BoolArray],
    name: str,
    settings: Settings,
) -> list[Record]:
    """Scores one text variant with the reranker and reports the agreement."""
    rows, mask = subset
    ids = read_integers(rows, "session_id")
    sessions = locate(inputs.catalog.intent_ids, ids)
    documents = inputs.documents[name]
    intents = inputs.catalog.intent_texts
    positions = documents.positions[mask]
    scores = reranker_scores(scorer, intents, sessions, documents, positions)
    key = settings.comparison.reranker.key
    return report_scores(key, name, scores, rows, settings)


def evaluate_reranker(inputs: ComparisonInputs, settings: Settings) -> list[Record]:
    """Reports the agreement of the reranker on a smaller set of sessions."""
    features = settings.features
    scorer = CrossPairScorer(
        settings.comparison.reranker, features.device, features.half_precision
    )
    subset = reranker_rows(inputs, settings)
    records: list[Record] = []
    for name in TEXT_VARIANTS:
        records += reranker_records(scorer, inputs, subset, name, settings)
    return records


def run_comparison(settings: Settings) -> pd.DataFrame:
    """Runs the whole comparison and saves the long-format result table."""
    inputs = build_inputs(settings)
    records: list[Record] = []
    for variant in settings.comparison.embeddings:
        records += evaluate_embedding(variant, inputs, settings)
        print(variant.key, "done", flush=True)
    records += evaluate_reranker(inputs, settings)
    return save_table(pd.DataFrame(records), settings)


def save_table(table: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Writes the comparison table to the results folder."""
    output = results_directory(settings)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / settings.comparison.output_file, index=False)
    return table


def pivot_report(table: pd.DataFrame, group: str) -> dict[str, pd.DataFrame]:
    """Builds, per target, a model-by-text-variant table of one register group."""
    reports: dict[str, pd.DataFrame] = {}
    for target, part in table.groupby("target"):
        reports[str(target)] = part.pivot(
            index="model", columns="text_variant", values=group
        )
    return reports
