"""Scores how well text features agree with the hidden fit of the truth."""

import numpy as np
import pandas as pd

from src.world.helpers import read_floats, read_integers
from src.world.models import FloatArray

ALL_SESSIONS = "all"


def centered_ranks(values: FloatArray, sessions: pd.Series) -> FloatArray:
    """Ranks values inside each session and subtracts the session mean rank."""
    frame = pd.DataFrame({"session": sessions.to_numpy(), "value": values})
    ranks = frame.groupby("session")["value"].rank()
    means = ranks.groupby(frame["session"]).transform("mean")
    return (ranks - means).to_numpy(dtype=np.float64)


def sessions_correlation(
    first: FloatArray,
    second: FloatArray,
    sessions: pd.Series,
) -> float:
    """Averages the within-session Spearman correlation of two scores."""
    left = centered_ranks(first, sessions)
    right = centered_ranks(second, sessions)
    keys = read_integers(pd.DataFrame({"key": sessions}), "key")
    _, inverse = np.unique(keys, return_inverse=True)
    products = np.bincount(inverse, left * right)
    left_norm = np.bincount(inverse, left * left)
    right_norm = np.bincount(inverse, right * right)
    valid = (left_norm > 0) & (right_norm > 0)
    return float(np.mean(products[valid] / np.sqrt(left_norm * right_norm)[valid]))


def score_feature(
    score: FloatArray,
    column: str,
    truth: pd.DataFrame,
) -> dict[str, float | str]:
    """Scores one feature on all sessions and on each message register."""
    fit = read_floats(truth, "hid_fit")
    registers = truth["hid_intent_register"].astype(str)
    sessions = truth["session_id"]
    row: dict[str, float | str] = {"feature": column}
    row[ALL_SESSIONS] = sessions_correlation(score, fit, sessions)
    for register in sorted(registers.unique()):
        mask = (registers == register).to_numpy()
        row[register] = sessions_correlation(score[mask], fit[mask], sessions[mask])
    return row


def evaluate_text_features(
    features: pd.DataFrame,
    truth: pd.DataFrame,
    text_columns: list[str],
) -> pd.DataFrame:
    """Reports each text feature's agreement with the hidden fit per register."""
    rows: list[dict[str, float | str]] = []
    for column in text_columns:
        rows.append(score_feature(read_floats(features, column), column, truth))
    return pd.DataFrame(rows)
