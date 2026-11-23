"""Typed settings of the hidden demand shock scenario and the weighted model."""

from pydantic import BaseModel

from src.settings.modeling_models import ExperimentRecord


class ConfounderSettings(BaseModel):
    enabled: bool
    shock_share: float
    utility_shift: float
    margin_shift: float
    seed: int
    data_folder: str
    shock_column: str
    weighted_experiment: ExperimentRecord
