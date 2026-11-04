"""Typed models describing the feature building stage."""

from pydantic import BaseModel

from src.settings.amenity_models import CharacteristicRecord


class EmbeddingVariant(BaseModel):
    key: str
    repository: str
    query_template: str
    hotel_task: str
    room_task: str
    place_task: str
    combined_task: str
    batch_size: int
    maximum_tokens: int


class RerankerVariant(BaseModel):
    key: str
    repository: str
    batch_size: int
    maximum_tokens: int


class FeatureColumns(BaseModel):
    categorical: list[str]
    numeric: list[str]
    derived: list[str]


class FeatureSettings(BaseModel):
    embeddings: list[EmbeddingVariant]
    columns: FeatureColumns
    device: str
    half_precision: bool
    chunk_rows: int
    cache_hash_length: int
    id_columns: list[str]
    document_separator: str
    location_prefix: str
    label_phrases: dict[str, str]
    label_characteristics: dict[str, CharacteristicRecord]
    label_variant_key: str
    direct_variant_keys: list[str]
    direct_components: int
    pca_fit_rows: int
    relative_columns: list[str]
    evaluation_pool_size: int
