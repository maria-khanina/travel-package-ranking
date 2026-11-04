"""Chooses which sessions write their message in the practical register."""

import numpy as np

from src.settings.loader import Settings
from src.world.models import BoolArray


def select_practical_sessions(session_count: int, settings: Settings) -> BoolArray:
    """Chooses, reproducibly, the sessions that write in a practical register."""
    run = settings.run
    generator = np.random.default_rng(run.seed + run.practical_seed_offset)
    return generator.random(session_count) < run.practical_intent_share
