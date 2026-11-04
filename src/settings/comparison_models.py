"""Typed models describing the comparison of text models."""

from pydantic import BaseModel

from src.settings.feature_models import EmbeddingVariant, RerankerVariant


class ComparisonSettings(BaseModel):
    embeddings: list[EmbeddingVariant]
    reranker: RerankerVariant
    sample_sessions: int
    reranker_sessions: int
    sample_seed: int
    targets: list[str]
    cache_folder: str
    output_file: str
