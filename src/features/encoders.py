"""Wraps the text models that turn texts into vectors."""

from typing import Any, Protocol, cast

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch

from src.settings.feature_models import EmbeddingVariant, RerankerVariant
from src.world.models import FloatArray, VectorArray


class TextEncoder(Protocol):
    def encode(self, texts: list[str]) -> VectorArray: ...


class PairScorer(Protocol):
    def score(self, queries: list[str], documents: list[str]) -> FloatArray: ...


class SentenceEncoder:
    def __init__(
        self,
        variant: EmbeddingVariant,
        device: str,
        half_precision: bool,
    ) -> None:
        dtype = torch.float16 if half_precision else torch.float32
        self.model = SentenceTransformer(
            variant.repository,
            device=device,
            model_kwargs={"torch_dtype": dtype},
        )
        self.model.max_seq_length = variant.maximum_tokens
        self.batch_size = variant.batch_size

    def encode(self, texts: list[str]) -> VectorArray:
        """Encodes texts into unit-length vectors."""
        vectors = cast(Any, self.model).encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


class CrossPairScorer:
    def __init__(
        self,
        variant: RerankerVariant,
        device: str,
        half_precision: bool,
    ) -> None:
        dtype = torch.float16 if half_precision else torch.float32
        self.model = CrossEncoder(
            variant.repository,
            device=device,
            max_length=variant.maximum_tokens,
            activation_fn=torch.nn.Identity(),
            model_kwargs={"torch_dtype": dtype},
        )
        self.batch_size = variant.batch_size

    def score(self, queries: list[str], documents: list[str]) -> FloatArray:
        """Scores every query-document pair with the raw model logit."""
        pairs = list(zip(queries, documents))
        scores = cast(Any, self.model).predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype=np.float64)
