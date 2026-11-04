"""Small sampling, vector and table-reading helpers for the world builders."""

import numpy as np
import pandas as pd

from src.world.models import FloatArray, IntArray, TextArray


def sample_categorical(
    random_generator: np.random.Generator,
    probabilities: FloatArray,
) -> IntArray:
    """Draws one category index per row of a probability matrix."""
    cumulative = probabilities.cumsum(axis=1)
    draws = random_generator.random((probabilities.shape[0], 1))
    indices = (draws > cumulative).sum(axis=1)
    return np.minimum(indices, probabilities.shape[1] - 1)


def sample_index(
    random_generator: np.random.Generator,
    probabilities: FloatArray,
) -> int:
    """Draws one category index from a single probability vector."""
    return int(random_generator.choice(len(probabilities), p=probabilities))


def normalize(weights: FloatArray) -> FloatArray:
    """Scales a non-negative weight vector so that it sums to one."""
    return weights / weights.sum()


def normalize_rows(weights: FloatArray) -> FloatArray:
    """Scales every row of a weight matrix so that it sums to one."""
    return weights / weights.sum(axis=1, keepdims=True)


def build_aspect_vector(
    values: dict[str, float],
    aspect_names: list[str],
) -> FloatArray:
    """Turns an aspect-to-value mapping into a vector in aspect order."""
    vector = np.zeros(len(aspect_names))
    for aspect, value in values.items():
        vector[aspect_names.index(aspect)] = value
    return vector


def read_floats(table: pd.DataFrame, column: str) -> FloatArray:
    """Reads a table column as a float array."""
    return table[column].to_numpy(dtype=np.float64)


def read_integers(table: pd.DataFrame, column: str) -> IntArray:
    """Reads a table column as an integer array."""
    return table[column].to_numpy(dtype=np.int64)


def read_texts(table: pd.DataFrame, column: str) -> TextArray:
    """Reads a table column as an array of Python objects."""
    return table[column].to_numpy(dtype=np.object_)


def read_strings(table: pd.DataFrame, column: str) -> list[str]:
    """Reads a table column as a list of strings."""
    return [str(value) for value in table[column].tolist()]
