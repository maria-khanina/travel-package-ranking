"""Builds the briefs of every text-writing job of the dataset."""

import numpy as np

from src.settings.loader import Settings
from src.texts.models import TextJobs
from src.texts.practical_specs import build_practical_jobs
from src.texts.specs import build_hotel_jobs, build_intent_jobs, build_room_jobs
from src.world.models import WorldTables

INTENT_STREAM = 1
PRACTICAL_STREAM = 2
HOTEL_STREAM = 3
ROOM_STREAM = 4


def build_job_generator(seed: int, stream: int) -> np.random.Generator:
    """Creates the random stream of one job, independent of the other jobs."""
    return np.random.default_rng([seed, stream])


def build_text_jobs(seed: int, world: WorldTables, settings: Settings) -> TextJobs:
    """Builds the briefs for all texts the world needs."""
    intents = build_job_generator(seed, INTENT_STREAM)
    practical = build_job_generator(seed, PRACTICAL_STREAM)
    hotels = build_job_generator(seed, HOTEL_STREAM)
    rooms = build_job_generator(seed, ROOM_STREAM)
    return TextJobs(
        intents=build_intent_jobs(intents, world.sessions, settings),
        hotels=build_hotel_jobs(hotels, world.hotels, settings),
        rooms=build_room_jobs(rooms, world.rooms, world.hotels, settings),
        practical_intents=build_practical_jobs(practical, world.sessions, settings),
    )
