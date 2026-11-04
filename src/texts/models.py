"""Typed specifications, records and job tables used by text generation."""

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel

JOB_INTENTS = "intents"
JOB_HOTELS = "hotels"
JOB_ROOMS = "rooms"
JOB_PRACTICAL = "practical_intents"
JOB_NAMES: list[str] = [JOB_INTENTS, JOB_PRACTICAL, JOB_HOTELS, JOB_ROOMS]


class IntentSpec(BaseModel):
    wants: list[str]
    avoids: list[str]
    budget: str | None
    examples: list[str]
    maximum_words: int
    styles: dict[str, str]
    labels: dict[str, str]


class PracticalSpec(BaseModel):
    requirements: list[str]
    avoids: list[str]
    budget: str | None
    examples: list[str]
    maximum_words: int
    styles: dict[str, str]
    labels: dict[str, str]


class HotelSpec(BaseModel):
    name: str
    hotel_type: str
    stars: int
    city: str
    country: str
    emphasize: list[str]
    mention: list[str]
    facts: list[str]
    styles: dict[str, str]
    labels: dict[str, str]


class RoomSpec(BaseModel):
    hotel_type: str
    stars: int
    city: str
    category: str
    view: str
    facts: list[str]
    styles: dict[str, str]
    labels: dict[str, str]


class TextRecord(BaseModel):
    id: int
    text: str | None
    model: str
    attempts: int


@dataclass
class TextBundle:
    intents: dict[int, str]
    practical_intents: dict[int, str]
    hotels: dict[int, str]
    rooms: dict[int, str]


@dataclass
class TextJobs:
    intents: pd.DataFrame
    practical_intents: pd.DataFrame
    hotels: pd.DataFrame
    rooms: pd.DataFrame

    def get(self, job: str) -> pd.DataFrame:
        """Returns the brief table of one job by its name."""
        tables: dict[str, pd.DataFrame] = {
            JOB_INTENTS: self.intents,
            JOB_PRACTICAL: self.practical_intents,
            JOB_HOTELS: self.hotels,
            JOB_ROOMS: self.rooms,
        }
        return tables[job]
