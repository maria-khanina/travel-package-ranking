"""Estimates what a traveler spends on meals the package does not cover."""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.world.helpers import read_floats, read_integers, read_texts
from src.world.models import FloatArray


def count_persons(rows: pd.DataFrame, settings: Settings) -> FloatArray:
    """Counts the travelers, with children weighted below adults."""
    children = read_floats(rows, "n_children")
    return read_floats(rows, "n_adults") + settings.choice.child_person_share * children


def compute_meal_cost(
    rows: pd.DataFrame,
    destinations: pd.DataFrame,
    settings: Settings,
) -> FloatArray:
    """Computes the extra cost of meals not included in the board type."""
    meals = settings.amenities.meals
    destination = read_integers(rows, "hid_destination")
    price_level = read_floats(destinations, "price_level")[destination]
    boards = read_texts(rows, "board_type")
    shares = np.array([meals.uncovered_share[str(board)] for board in boards])
    per_person_night = meals.cost_per_price_level * price_level * shares
    persons = count_persons(rows, settings)
    return per_person_night * persons * read_floats(rows, "nights")
