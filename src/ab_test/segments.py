"""Reports the experiment effect within each traveler segment."""

import pandas as pd

from src.world.models import BoolArray, FloatArray, TextArray


def segment_effects(
    metric: FloatArray,
    treated: BoolArray,
    segments: TextArray,
    alpha: float,
) -> pd.DataFrame:
    """Compares the arms separately inside every segment."""
    raise NotImplementedError
