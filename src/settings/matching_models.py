"""Typed models describing the match scores between travelers and packages."""

from pydantic import BaseModel


class MatchSettings(BaseModel):
    strong_visibility: float
    medium_visibility: float
    chunk_rows: int
    reference_sessions: int
    reference_seed: int
