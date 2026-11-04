"""Extra text inputs: similarity to characteristic labels and direct embeddings."""

import numpy as np
import pandas as pd

from src.features.catalog import TextCatalog, locate
from src.features.encoders import TextEncoder
from src.features.vectors import VariantVectors
from src.settings.feature_models import EmbeddingVariant, FeatureSettings
from src.world.helpers import read_integers
from src.world.models import FloatArray, IntArray, VectorArray

HOTEL_PART = "hotel"
ROOM_PART = "room"
LABEL_PREFIX = "sim_label_"
EMBEDDING_PREFIX = "emb_"


def label_columns(
    queries: VectorArray,
    sessions: IntArray,
    encoder: TextEncoder,
    settings: FeatureSettings,
) -> dict[str, VectorArray]:
    """Computes the similarity of every query to each characteristic label."""
    names = list(settings.label_phrases)
    labels = encoder.encode([settings.label_phrases[name] for name in names])
    similarity = np.asarray(queries @ labels.T, dtype=np.float32)
    columns: dict[str, VectorArray] = {}
    for index, name in enumerate(names):
        columns[f"{LABEL_PREFIX}{name}"] = similarity[sessions, index]
    return columns


def fit_projection(
    vectors: VectorArray,
    components: int,
    rows: int,
) -> tuple[FloatArray, FloatArray]:
    """Finds the mean and the main directions of a set of vectors."""
    sample = np.asarray(vectors[:rows], dtype=np.float64)
    mean = sample.mean(axis=0)
    _, _, directions = np.linalg.svd(sample - mean, full_matrices=False)
    return mean, np.asarray(directions[:components])


def project_rows(
    vectors: VectorArray,
    positions: IntArray,
    components: int,
    rows: int,
) -> VectorArray:
    """Projects vectors on their main directions and spreads them to the rows."""
    mean, directions = fit_projection(vectors, components, rows)
    projected = (np.asarray(vectors, dtype=np.float64) - mean) @ directions.T
    return np.asarray(projected[positions], dtype=np.float32)


def direct_columns(
    dataset: pd.DataFrame,
    catalog: TextCatalog,
    key: str,
    vectors: VariantVectors,
    sessions: IntArray,
    settings: FeatureSettings,
) -> dict[str, VectorArray]:
    """Gives the main directions of the query, hotel and room embeddings."""
    hotels = locate(catalog.hotel_ids, read_integers(dataset, "hotel_id"))
    rooms = locate(catalog.room_ids, read_integers(dataset, "room_id"))
    sources = {
        "query": (vectors.queries[HOTEL_PART], sessions),
        "hotel": (vectors.documents[HOTEL_PART], hotels),
        "room": (vectors.documents[ROOM_PART], rooms),
    }
    columns: dict[str, VectorArray] = {}
    for name, (source, positions) in sources.items():
        projected = project_rows(
            source, positions, settings.direct_components, settings.pca_fit_rows
        )
        for component in range(settings.direct_components):
            label = f"{EMBEDDING_PREFIX}{key}_{name}_{component}"
            columns[label] = projected[:, component]
    return columns


def extra_columns(
    dataset: pd.DataFrame,
    catalog: TextCatalog,
    variant: EmbeddingVariant,
    vectors: VariantVectors,
    encoder: TextEncoder,
    settings: FeatureSettings,
) -> dict[str, VectorArray]:
    """Builds the label and direct-embedding columns for the chosen variants."""
    sessions = locate(catalog.intent_ids, read_integers(dataset, "session_id"))
    columns: dict[str, VectorArray] = {}
    if variant.key == settings.label_variant_key:
        queries = vectors.queries[HOTEL_PART]
        columns.update(label_columns(queries, sessions, encoder, settings))
    if variant.key in settings.direct_variant_keys:
        columns.update(
            direct_columns(dataset, catalog, variant.key, vectors, sessions, settings)
        )
    return columns
