"""Typed settings of the simulated A/B tests of pricing policies."""

from pydantic import BaseModel


class PolicyTestRecord(BaseModel):
    name: str
    control: str
    treatment: str


class SegmentSettings(BaseModel):
    device_column: str
    children_column: str
    family_label: str
    adults_label: str
    minimum_sessions: int


class AbTestSettings(BaseModel):
    tests: list[PolicyTestRecord]
    alpha: float
    power: float
    replications: int
    bootstrap_resamples: int
    seed: int
    segments: SegmentSettings
    results_folder: str
    comparison_file: str
    power_file: str
    validation_file: str
    segment_file: str
