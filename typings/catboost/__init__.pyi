"""Type stubs for the parts of CatBoost the project uses."""

import numpy as np
import numpy.typing as npt
import pandas as pd


class Pool:
    def __init__(
        self,
        data: pd.DataFrame,
        label: npt.NDArray[np.int64] | None = ...,
        cat_features: list[str] | None = ...,
        group_id: npt.NDArray[np.int64] | None = ...,
        weight: npt.NDArray[np.float64] | None = ...,
    ) -> None: ...


class CatBoost:
    def fit(
        self,
        X: Pool,
        eval_set: Pool | None = ...,
        use_best_model: bool | None = ...,
    ) -> CatBoost: ...
    def predict(self, data: Pool) -> npt.NDArray[np.float64]: ...
    def get_best_iteration(self) -> int | None: ...
    def save_model(self, fname: str) -> None: ...
    def load_model(self, fname: str) -> CatBoost: ...
    def get_feature_importance(
        self,
        data: Pool | None = ...,
        type: str = ...,
    ) -> npt.NDArray[np.float64]: ...


class CatBoostClassifier(CatBoost):
    def __init__(
        self,
        iterations: int | None = ...,
        learning_rate: float | None = ...,
        depth: int | None = ...,
        l2_leaf_reg: float | None = ...,
        early_stopping_rounds: int | None = ...,
        thread_count: int | None = ...,
        task_type: str | None = ...,
        random_seed: int | None = ...,
        verbose: int | None = ...,
        loss_function: str | None = ...,
        eval_metric: str | None = ...,
    ) -> None: ...
    def predict_proba(self, data: Pool) -> npt.NDArray[np.float64]: ...


class CatBoostRanker(CatBoost):
    def __init__(
        self,
        iterations: int | None = ...,
        learning_rate: float | None = ...,
        depth: int | None = ...,
        l2_leaf_reg: float | None = ...,
        early_stopping_rounds: int | None = ...,
        thread_count: int | None = ...,
        task_type: str | None = ...,
        random_seed: int | None = ...,
        verbose: int | None = ...,
        loss_function: str | None = ...,
        eval_metric: str | None = ...,
    ) -> None: ...
