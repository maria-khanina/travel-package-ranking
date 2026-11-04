"""Assembles the final dataset from the world, the texts and the bookings."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavior.simulation import calibrate_outside_constant, simulate_bookings
from src.settings.loader import Settings
from src.texts.models import (
    JOB_HOTELS,
    JOB_INTENTS,
    JOB_PRACTICAL,
    JOB_ROOMS,
    HotelSpec,
    IntentSpec,
    PracticalSpec,
    RoomSpec,
    TextBundle,
    TextJobs,
)
from src.texts.selection import select_practical_sessions
from src.texts.store import read_text_map
from src.world.helpers import read_floats, read_integers
from src.world.models import BoolArray, FloatArray, WorldTables

HIDDEN_PREFIX = "hid_"
TEXT_FILE_SUFFIX = ".jsonl"


@dataclass
class DatasetBundle:
    public: pd.DataFrame
    truth: pd.DataFrame
    outside_constant: float
    probabilities: FloatArray


def load_texts(texts_directory: Path) -> TextBundle:
    """Reads the generated texts of all jobs from a directory."""
    return TextBundle(
        intents=read_text_map(texts_directory / f"{JOB_INTENTS}{TEXT_FILE_SUFFIX}"),
        practical_intents=read_text_map(
            texts_directory / f"{JOB_PRACTICAL}{TEXT_FILE_SUFFIX}"
        ),
        hotels=read_text_map(texts_directory / f"{JOB_HOTELS}{TEXT_FILE_SUFFIX}"),
        rooms=read_text_map(texts_directory / f"{JOB_ROOMS}{TEXT_FILE_SUFFIX}"),
    )


def write_placeholder_intent(spec: IntentSpec) -> str:
    """Writes a plain message listing the wishes of a traveler."""
    text = "I want " + " and ".join(spec.wants)
    if spec.avoids:
        text += ", " + ", ".join(spec.avoids)
    if spec.budget:
        text += ", " + spec.budget
    return text


def write_placeholder_practical(spec: PracticalSpec) -> str:
    """Writes a plain list of the practical requirements."""
    return ", ".join(spec.requirements)


def write_placeholder_hotel(spec: HotelSpec) -> str:
    """Writes a plain hotel description listing its features."""
    text = f"{spec.name} is a {spec.stars}-star {spec.hotel_type} in {spec.city}."
    features = spec.emphasize + spec.mention + spec.facts
    if features:
        text += " It offers " + ", ".join(features) + "."
    return text


def write_placeholder_room(spec: RoomSpec) -> str:
    """Writes a plain room description."""
    text = f"{spec.category.capitalize()} with {spec.view}."
    return text + "".join(f" {fact.capitalize()}." for fact in spec.facts)


def write_placeholder(job: str, spec_json: str) -> str:
    """Writes the placeholder text of one brief, whatever the job."""
    if job == JOB_INTENTS:
        return write_placeholder_intent(IntentSpec.model_validate_json(spec_json))
    if job == JOB_PRACTICAL:
        return write_placeholder_practical(PracticalSpec.model_validate_json(spec_json))
    if job == JOB_HOTELS:
        return write_placeholder_hotel(HotelSpec.model_validate_json(spec_json))
    return write_placeholder_room(RoomSpec.model_validate_json(spec_json))


def build_placeholder_map(jobs: TextJobs, job: str) -> dict[int, str]:
    """Creates a stand-in text for every brief of one job."""
    briefs = jobs.get(job)
    texts: dict[int, str] = {}
    for item_id, spec_json in zip(briefs["id"], briefs["spec"]):
        texts[int(item_id)] = write_placeholder(job, str(spec_json))
    return texts


def build_placeholder_texts(jobs: TextJobs) -> TextBundle:
    """Creates deterministic stand-in texts for every brief."""
    return TextBundle(
        intents=build_placeholder_map(jobs, JOB_INTENTS),
        practical_intents=build_placeholder_map(jobs, JOB_PRACTICAL),
        hotels=build_placeholder_map(jobs, JOB_HOTELS),
        rooms=build_placeholder_map(jobs, JOB_ROOMS),
    )


def attach_intents(
    rows: pd.DataFrame,
    texts: TextBundle,
    practical_sessions: BoolArray,
) -> None:
    """Adds the traveler message, practical for the selected sessions."""
    sessions = read_integers(rows, "session_id")
    vibe = rows["session_id"].map(texts.intents)
    practical = rows["session_id"].map(texts.practical_intents)
    use_practical = practical_sessions[sessions] & practical.notna().to_numpy()
    rows["user_intent"] = np.where(use_practical, practical, vibe)
    rows["hid_intent_register"] = np.where(use_practical, "practical", "vibe")


def attach_texts(
    rows: pd.DataFrame,
    texts: TextBundle,
    practical_sessions: BoolArray,
) -> None:
    """Adds the intent, hotel and room texts to the row-level table."""
    attach_intents(rows, texts, practical_sessions)
    rows["hotel_description"] = rows["hotel_id"].map(texts.hotels)
    rows["room_description"] = rows["room_id"].map(texts.rooms)


def extract_truth(rows: pd.DataFrame) -> pd.DataFrame:
    """Keeps the hidden columns that only the simulator knows."""
    hidden: list[str] = []
    for column in rows.columns:
        if str(column).startswith(HIDDEN_PREFIX):
            hidden.append(str(column))
    return rows[["session_id", "room_id"] + hidden].copy()


def drop_incomplete_sessions(rows: pd.DataFrame) -> pd.DataFrame:
    """Removes the sessions whose message could not be generated."""
    missing = rows.loc[rows["user_intent"].isna(), "session_id"].unique()
    return rows[~rows["session_id"].isin(missing)].reset_index(drop=True)


def build_dataset(
    world: WorldTables,
    texts: TextBundle,
    settings: Settings,
    outside_constant: float | None,
) -> DatasetBundle:
    """Simulates the bookings and returns the public and the hidden tables."""
    rows = world.rows.copy()
    practical_sessions = select_practical_sessions(len(world.sessions), settings)
    attach_texts(rows, texts, practical_sessions)
    rows = drop_incomplete_sessions(rows)
    if outside_constant is None:
        target = settings.run.target_conversion
        outside_constant = calibrate_outside_constant(rows, settings, target)
    outcome = simulate_bookings(
        rows, settings, outside_constant, read_floats(rows, "margin_percentage")
    )
    rows["is_booked"] = outcome.is_booked
    return DatasetBundle(
        public=rows[settings.run.dataset_columns].copy(),
        truth=extract_truth(rows),
        outside_constant=outside_constant,
        probabilities=outcome.probabilities,
    )
