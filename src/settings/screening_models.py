"""Typed models describing the statistical screening of query labels."""

from pydantic import BaseModel


class ScreeningSettings(BaseModel):
    permutations: int
    top_labels: int
    permutation_chunk: int
    seed: int
    false_discovery_rate: float
    confirmation_level: float
    selection_folds: list[int]
    confirmation_folds: list[int]
    truth_taste: dict[str, str]
    output_file: str
