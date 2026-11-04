"""Containers for the text vectors computed by one embedding variant."""

from dataclasses import dataclass

from src.world.models import VectorArray


@dataclass
class VariantVectors:
    queries: dict[str, VectorArray]
    documents: dict[str, VectorArray]
