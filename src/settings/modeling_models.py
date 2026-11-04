"""Typed models describing the training and evaluation of the booking model."""

from pydantic import BaseModel


class SplitSettings(BaseModel):
    seed: int
    test_share: float
    fold_count: int
    cold_hotel_share: float
    pool_size: int


class BootstrapSettings(BaseModel):
    replicates: int
    seed: int
    lower_percentile: float
    upper_percentile: float
    baseline: str


class TruthSettings(BaseModel):
    draws: int
    chunk_sessions: int
    seed: int
    output_file: str


class CalibrationSettings(BaseModel):
    experiment: str
    bins: int
    session_bins: int
    confidence_z: float
    price_quantiles: list[float]
    newton_iterations: int
    newton_tolerance: float
    probability_floor: float
    figure_folder: str
    model_folder: str


class ExplainSettings(BaseModel):
    experiment: str
    sessions_per_fold: int
    seed: int
    top_features: int
    dependence_points: int
    exact_groups: dict[str, list[str]]
    prefix_groups: dict[str, str]
    no_effect_columns: list[str]
    figure_name: str
    importance_file: str
    group_file: str


class MarginSettings(BaseModel):
    experiment: str
    sample_sessions: int
    window_sessions: int
    seed: int
    slate_size: int
    fixed_margin_policies: bool
    draws: int
    chunk_sessions: int
    probability_floor: float
    bins: int
    policy_file: str
    paired_file: str
    response_file: str
    response_price_file: str


class CatBoostSettings(BaseModel):
    iterations: int
    learning_rate: float
    depth: int
    l2_leaf_reg: float
    early_stopping_rounds: int
    thread_count: int
    task_type: str
    random_seed: int
    verbose: int


class ExperimentRecord(BaseModel):
    name: str
    text_columns: list[str]
    dropped_columns: list[str]
    include_hotel_id: bool
    extra_prefixes: list[str]
    relative_columns: list[str]
    loss: str


class ModelingSettings(BaseModel):
    split: SplitSettings
    bootstrap: BootstrapSettings
    truth: TruthSettings
    calibration: CalibrationSettings
    explain: ExplainSettings
    margin: MarginSettings
    catboost: CatBoostSettings
    experiments: list[ExperimentRecord]
    top_k_values: list[int]
    diagnostic_rows: int
    label_column: str
    hotel_column: str
    register_column: str
