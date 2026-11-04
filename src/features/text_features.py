"""Computes similarity features between intents and offers."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.catalog import (
    TextCatalog,
    cache_file,
    locate,
)
from src.features.documents import (
    HOTEL_PLACE,
    HOTEL_PLACE_ROOM,
    build_documents,
    task_for,
)
from src.features.encoders import SentenceEncoder, TextEncoder
from src.features.extras import extra_columns
from src.features.vectors import VariantVectors
from src.settings.feature_models import EmbeddingVariant, FeatureSettings
from src.world.helpers import read_integers, read_strings
from src.world.models import FloatArray, IntArray, VectorArray

HOTEL_PART = "hotel"
ROOM_PART = "room"


def fill_template(template: str, task: str, text: str) -> str:
    """Wraps a query text into the model's query template."""
    return template.replace("{task}", task).replace("{query}", text)


def build_queries(variant: EmbeddingVariant, task: str, texts: list[str]) -> list[str]:
    """Wraps every intent into the query template of a variant and a task."""
    return [fill_template(variant.query_template, task, text) for text in texts]


def encode_cached(
    encoder: TextEncoder,
    texts: list[str],
    path: Path,
) -> VectorArray:
    """Encodes texts or loads the vectors saved by an earlier run."""
    if path.exists():
        return np.load(path)
    vectors = encoder.encode(texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def row_cosine(
    query_vectors: VectorArray,
    query_positions: IntArray,
    document_vectors: VectorArray,
    document_positions: IntArray,
    chunk_rows: int,
) -> FloatArray:
    """Computes the cosine of every row's query and document in chunks."""
    result = np.empty(len(query_positions), dtype=np.float64)
    for start in range(0, len(query_positions), chunk_rows):
        stop = start + chunk_rows
        queries = query_vectors[query_positions[start:stop]]
        documents = document_vectors[document_positions[start:stop]]
        result[start:stop] = np.einsum("ij,ij->i", queries, documents)
    return result


def place_positions(dataset: pd.DataFrame) -> tuple[list[str], IntArray]:
    """Names the destination of every row as city and country."""
    cities = read_strings(dataset, "city")
    countries = read_strings(dataset, "country")
    labels = [f"{city}, {country}" for city, country in zip(cities, countries)]
    places = sorted(set(labels))
    index = {place: position for position, place in enumerate(places)}
    return places, np.array([index[label] for label in labels], dtype=np.int64)


def place_feature(
    dataset: pd.DataFrame,
    catalog: TextCatalog,
    variant: EmbeddingVariant,
    encoder: TextEncoder,
    settings: FeatureSettings,
    directory: Path,
) -> FloatArray:
    """Computes the similarity between a query and the row's destination."""
    places, positions = place_positions(dataset)
    length = settings.cache_hash_length
    queries = build_queries(variant, variant.place_task, catalog.intent_texts)
    query_path = cache_file(directory, variant.repository, queries, length)
    place_path = cache_file(directory, variant.repository, places, length)
    sessions = locate(catalog.intent_ids, read_integers(dataset, "session_id"))
    return row_cosine(
        encode_cached(encoder, queries, query_path),
        sessions,
        encode_cached(encoder, places, place_path),
        positions,
        settings.chunk_rows,
    )


def encode_part(
    encoder: TextEncoder,
    variant: EmbeddingVariant,
    documents: list[str],
    queries: list[str],
    settings_and_directory: tuple[FeatureSettings, Path],
) -> tuple[VectorArray, VectorArray]:
    """Encodes the queries and the documents of one part, using the cache."""
    settings, directory = settings_and_directory
    length = settings.cache_hash_length
    query_path = cache_file(directory, variant.repository, queries, length)
    document_path = cache_file(directory, variant.repository, documents, length)
    return (
        encode_cached(encoder, queries, query_path),
        encode_cached(encoder, documents, document_path),
    )


def encode_variant(
    encoder: TextEncoder,
    variant: EmbeddingVariant,
    catalog: TextCatalog,
    settings: FeatureSettings,
    directory: Path,
) -> VariantVectors:
    """Encodes the queries and the documents of one variant, using the cache."""
    tasks = {HOTEL_PART: variant.hotel_task, ROOM_PART: variant.room_task}
    texts = {HOTEL_PART: catalog.hotel_texts, ROOM_PART: catalog.room_texts}
    result = VariantVectors(queries={}, documents={})
    for part, task in tasks.items():
        queries = build_queries(variant, task, catalog.intent_texts)
        pair = (settings, directory)
        encoded = encode_part(encoder, variant, texts[part], queries, pair)
        result.queries[part], result.documents[part] = encoded
    return result


def stay_columns(
    dataset: pd.DataFrame,
    catalog: TextCatalog,
    variant: EmbeddingVariant,
    vectors: VariantVectors,
    settings: FeatureSettings,
) -> dict[str, FloatArray]:
    """Computes the query similarity to the hotel and to the room of every row."""
    sessions = locate(catalog.intent_ids, read_integers(dataset, "session_id"))
    identifiers = {HOTEL_PART: catalog.hotel_ids, ROOM_PART: catalog.room_ids}
    columns: dict[str, FloatArray] = {}
    for part, ids in identifiers.items():
        positions = locate(ids, read_integers(dataset, f"{part}_id"))
        columns[f"sim_{part}_{variant.key}"] = row_cosine(
            vectors.queries[part],
            sessions,
            vectors.documents[part],
            positions,
            settings.chunk_rows,
        )
    return columns


COMPOSITE_NAMES: dict[str, str] = {
    HOTEL_PLACE: "sim_stay",
    HOTEL_PLACE_ROOM: "sim_full",
}


def composite_columns(
    dataset: pd.DataFrame,
    catalog: TextCatalog,
    variant: EmbeddingVariant,
    encoder: TextEncoder,
    settings: FeatureSettings,
    directory: Path,
) -> dict[str, FloatArray]:
    """Compares queries with the hotel plus place and the full stay texts."""
    documents = build_documents(
        dataset, catalog, settings.document_separator, settings.location_prefix
    )
    sessions = locate(catalog.intent_ids, read_integers(dataset, "session_id"))
    length = settings.cache_hash_length
    columns: dict[str, FloatArray] = {}
    for name, prefix in COMPOSITE_NAMES.items():
        texts = build_queries(variant, task_for(variant, name), catalog.intent_texts)
        query_path = cache_file(directory, variant.repository, texts, length)
        document_path = cache_file(
            directory, variant.repository, documents[name].texts, length
        )
        columns[f"{prefix}_{variant.key}"] = row_cosine(
            encode_cached(encoder, texts, query_path),
            sessions,
            encode_cached(encoder, documents[name].texts, document_path),
            documents[name].positions,
            settings.chunk_rows,
        )
    return columns


def embedding_features(
    dataset: pd.DataFrame,
    catalog: TextCatalog,
    variant: EmbeddingVariant,
    settings: FeatureSettings,
    directory: Path,
) -> pd.DataFrame:
    """Builds the similarity columns and the extra inputs of one variant."""
    encoder = SentenceEncoder(variant, settings.device, settings.half_precision)
    vectors = encode_variant(encoder, variant, catalog, settings, directory)
    columns: dict[str, FloatArray | VectorArray] = {}
    columns.update(stay_columns(dataset, catalog, variant, vectors, settings))
    columns[f"sim_place_{variant.key}"] = place_feature(
        dataset, catalog, variant, encoder, settings, directory
    )
    columns.update(
        composite_columns(dataset, catalog, variant, encoder, settings, directory)
    )
    columns.update(extra_columns(dataset, catalog, variant, vectors, encoder, settings))
    return pd.DataFrame(columns, index=dataset.index)


def build_text_features(
    dataset: pd.DataFrame,
    catalog: TextCatalog,
    settings: FeatureSettings,
    directory: Path,
) -> pd.DataFrame:
    """Builds every configured similarity feature column."""
    frames: list[pd.DataFrame] = []
    for variant in settings.embeddings:
        frame = embedding_features(dataset, catalog, variant, settings, directory)
        frames.append(frame)
    return pd.concat(frames, axis=1)
