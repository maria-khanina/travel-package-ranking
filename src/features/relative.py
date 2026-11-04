"""Features that compare a package with the other packages of its session."""

import numpy as np
import pandas as pd

from src.world.helpers import read_floats
from src.world.models import FloatArray, VectorArray

RANK_SUFFIX = "_pool_rank"


def compute_pool_rank(values: FloatArray, pool_size: int) -> VectorArray:
    """Gives the share of the session's packages that are not above a value."""
    pools = values.reshape(-1, pool_size)
    ranks = pools.argsort(axis=1).argsort(axis=1) + 1
    return np.asarray((ranks / pool_size).reshape(-1), dtype=np.float32)


def build_relative_features(
    table: pd.DataFrame,
    columns: list[str],
    pool_size: int,
) -> pd.DataFrame:
    """Adds the rank inside the session for each column."""
    created: dict[str, VectorArray] = {}
    for column in columns:
        values = read_floats(table, column)
        created[f"{column}{RANK_SUFFIX}"] = compute_pool_rank(values, pool_size)
    return pd.DataFrame(created, index=table.index)
