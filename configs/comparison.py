"""Text models compared by their agreement with the match scores."""

from configs.features import (
    BGE_QUERY_PREFIX,
    COMBINED_TASK,
    HOTEL_TASK,
    PLACE_TASK,
    QWEN_QUERY_TEMPLATE,
    ROOM_TASK,
)
from src.settings.comparison_models import ComparisonSettings
from src.settings.feature_models import EmbeddingVariant, RerankerVariant

PLAIN_TEMPLATE = "{query}"


def variant(
    key: str,
    repository: str,
    template: str,
    batch_and_tokens: tuple[int, int],
) -> EmbeddingVariant:
    """Builds an embedding variant that uses the shared task descriptions."""
    return EmbeddingVariant(
        key=key,
        repository=repository,
        query_template=template,
        hotel_task=HOTEL_TASK,
        room_task=ROOM_TASK,
        place_task=PLACE_TASK,
        combined_task=COMBINED_TASK,
        batch_size=batch_and_tokens[0],
        maximum_tokens=batch_and_tokens[1],
    )


QWEN = "Qwen/Qwen3-Embedding-0.6B"
BGE = "BAAI/bge-large-en-v1.5"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"

COMPARISON = ComparisonSettings(
    embeddings=[
        variant("qwen3_instruct", QWEN, QWEN_QUERY_TEMPLATE, (64, 512)),
        variant("qwen3_plain", QWEN, PLAIN_TEMPLATE, (64, 512)),
        variant("bge_large_instruct", BGE, BGE_QUERY_PREFIX, (64, 512)),
        variant("bge_large_plain", BGE, PLAIN_TEMPLATE, (64, 512)),
        variant("minilm", MINILM, PLAIN_TEMPLATE, (256, 256)),
    ],
    reranker=RerankerVariant(
        key="bge_reranker_base",
        repository="BAAI/bge-reranker-base",
        batch_size=128,
        maximum_tokens=512,
    ),
    sample_sessions=20000,
    reranker_sessions=5000,
    sample_seed=17,
    targets=["hid_open_match", "hid_stated_match", "hid_aspect_fit"],
    cache_folder="comparison",
    output_file="embedding_comparison.csv",
)
