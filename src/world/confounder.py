"""Adds a hidden demand shock that both the travelers and the old pricing rule see.

The shock stands for demand the pricing team knew about but the logs do not record,
such as a local event. It is drawn per session and destination, so no model feature
can reconstruct it. Where it is present, travelers find the packages more
attractive and the historical rule charges a higher margin, which confounds the
logged relation between margin and booking.
"""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.world.helpers import read_floats, read_integers
from src.world.margins import compute_target_margin, draw_logged_margins
from src.world.models import FloatArray, WorldTables


def draw_demand_shock(
    random_generator: np.random.Generator,
    rows: pd.DataFrame,
    settings: Settings,
) -> FloatArray:
    """Marks the rows whose session and destination pair has a demand shock."""
    sessions = read_integers(rows, "session_id")
    destinations = read_integers(rows, "hid_destination")
    shape = (int(sessions.max()) + 1, len(settings.destinations))
    shocked = random_generator.random(shape) < settings.confounder.shock_share
    return shocked[sessions, destinations].astype(np.float64)


def compute_shocked_target(
    rows: pd.DataFrame,
    destinations: pd.DataFrame,
    shock: FloatArray,
    settings: Settings,
) -> FloatArray:
    """Raises the target margin of the old pricing rule where demand is shocked."""
    destination = read_integers(rows, "hid_destination")
    price_level = read_floats(destinations, "price_level")[destination]
    target = compute_target_margin(rows, price_level, settings)
    return target + settings.confounder.margin_shift * shock


def apply_confounder(world: WorldTables, settings: Settings) -> None:
    """Adds the hidden shock and redraws the logged margins it influenced."""
    random_generator = np.random.default_rng(settings.confounder.seed)
    rows = world.rows
    shock = draw_demand_shock(random_generator, rows, settings)
    rows[settings.confounder.shock_column] = shock
    target = compute_shocked_target(rows, world.destinations, shock, settings)
    draw_logged_margins(random_generator, rows, target, settings)
