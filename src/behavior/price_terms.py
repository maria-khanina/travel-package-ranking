"""Price-related components of the utility a traveler gets from a package."""

import numpy as np
import pandas as pd

from src.settings.models import ChoiceSettings
from src.world.columns import MEAL_COST_COLUMN
from src.world.helpers import read_floats, read_integers
from src.world.models import FloatArray


def compute_price_ratio(
    net_cost: FloatArray,
    margin: FloatArray,
    competitor_price: FloatArray,
) -> FloatArray:
    """Compares our selling price with the competitor retail price."""
    return net_cost * (1.0 + margin) / competitor_price


def compute_softplus(values: FloatArray, sharpness: float) -> FloatArray:
    """Smooth version of max(0, x) with sharpness controlled by a scale."""
    return sharpness * np.logaddexp(0.0, values / sharpness)


def compute_urgency(
    days_to_departure: FloatArray,
    choice: ChoiceSettings,
) -> FloatArray:
    """Lowers price sensitivity for last-minute trips."""
    gap = days_to_departure - choice.urgency_center_days
    scaled = gap / choice.urgency_scale_days
    return choice.urgency_floor + choice.urgency_gain / (1.0 + np.exp(-scaled))


def compute_price_pressure(ratio: FloatArray, choice: ChoiceSettings) -> FloatArray:
    """Penalizes prices above competitors much harder than below them."""
    gap = ratio - 1.0
    above = compute_softplus(gap, choice.price_sharpness)
    return choice.price_slope_below * gap + choice.price_slope_above * above


def compute_trust_penalty(ratio: FloatArray, choice: ChoiceSettings) -> FloatArray:
    """Penalizes prices so low that they look suspicious."""
    shortfall = np.maximum(0.0, choice.trust_threshold - ratio)
    return choice.trust_coefficient * shortfall


def compute_budget_penalty(
    rows: pd.DataFrame,
    price: FloatArray,
    choice: ChoiceSettings,
) -> FloatArray:
    """Penalizes prices per person and night above the traveler budget."""
    children = read_floats(rows, "n_children")
    persons = read_floats(rows, "n_adults") + choice.child_person_share * children
    total = price + read_floats(rows, MEAL_COST_COLUMN)
    per_night = total / (persons * read_floats(rows, "nights"))
    budget = choice.budget_base_per_night * read_floats(rows, "hid_budget_scale")
    return choice.budget_coefficient * np.maximum(0.0, np.log(per_night / budget))


def compute_sensitivity(rows: pd.DataFrame, choice: ChoiceSettings) -> FloatArray:
    """Combines the hidden price sensitivity with urgency and fit effects."""
    shift = read_floats(rows, "hid_alpha_shift")
    alpha = np.exp(choice.alpha_z_coefficient * read_floats(rows, "hid_z") + shift)
    days = read_integers(rows, "days_to_departure").astype(np.float64)
    fit = np.clip(read_floats(rows, "hid_fit"), 0.0, 1.0)
    moderation = 1.0 - choice.fit_price_moderation * fit
    return alpha * compute_urgency(days, choice) * moderation


def compute_price_term(
    rows: pd.DataFrame,
    margin: FloatArray,
    choice: ChoiceSettings,
) -> FloatArray:
    """Combines price sensitivity, urgency, trust and budget effects."""
    net_cost = read_floats(rows, "net_cost")
    competitor = read_floats(rows, "competitor_retail_price")
    ratio = compute_price_ratio(net_cost, margin, competitor)
    pressure = compute_sensitivity(rows, choice) * compute_price_pressure(ratio, choice)
    budget = compute_budget_penalty(rows, net_cost * (1.0 + margin), choice)
    return -pressure - compute_trust_penalty(ratio, choice) - budget
