"""Checks of the open match and stated match scores."""

import numpy as np
import pandas as pd

from src.dataset import DatasetBundle
from src.features.evaluation import sessions_correlation
from src.matching.pipeline import compute_open_match, hidden_columns, open_columns
from src.matching.stated import (
    Statements,
    build_characteristics,
    build_phrase_book,
    build_statements,
    build_visibility,
    stated_scores,
)
from src.settings.loader import Settings
from src.texts.models import TextJobs
from src.world.helpers import read_integers

MINIMUM_AGREEMENT = 0.2
CHUNK = 7


def build_rows(bundle: DatasetBundle, settings: Settings) -> pd.DataFrame:
    identifiers = ["session_id", "room_id", "hotel_id"]
    public = bundle.public[identifiers + open_columns(settings)]
    hidden = bundle.truth[hidden_columns(settings)]
    parts = [public.reset_index(drop=True), hidden.reset_index(drop=True)]
    return pd.concat(parts, axis=1)


def test_a_stated_detail_scores_the_matching_characteristic(
    bundle: DatasetBundle,
    settings: Settings,
) -> None:
    rows = build_rows(bundle, settings)
    names = [record.name for record in settings.amenities.tastes]
    sessions = read_integers(rows, "session_id")
    count = int(sessions.max()) + 1
    wants = np.zeros((count, len(settings.catalogue.aspects.names)))
    tastes = np.zeros((count, len(names)))
    tastes[:, names.index("view_sea")] = 1.0
    statements = Statements(wants=wants, avoids=wants.copy(), tastes=tastes)
    visibility = np.zeros((int(rows["hotel_id"].max()) + 1, wants.shape[1]))
    scores = stated_scores(
        sessions,
        read_integers(rows, "hotel_id"),
        build_characteristics(rows, settings),
        (statements, visibility),
        CHUNK,
    )
    expected = (rows["room_view"] == "sea").to_numpy(dtype=np.float64)
    assert np.allclose(scores, expected)


def test_open_match_adds_aspect_and_taste_parts(
    bundle: DatasetBundle,
    settings: Settings,
) -> None:
    rows = build_rows(bundle, settings)
    match = compute_open_match(rows, settings)
    assert np.isfinite(match).all()
    assert match.std() > 0.0


def test_stated_and_open_match_agree_inside_sessions(
    bundle: DatasetBundle,
    jobs: TextJobs,
    settings: Settings,
) -> None:
    rows = build_rows(bundle, settings)
    sessions = read_integers(rows, "session_id")
    book = build_phrase_book(settings)
    statements = build_statements(
        (jobs.intents, jobs.practical_intents),
        int(sessions.max()) + 1,
        book,
        settings,
    )
    visibility = build_visibility(jobs.hotels, book, settings)
    stated = stated_scores(
        sessions,
        read_integers(rows, "hotel_id"),
        build_characteristics(rows, settings),
        (statements, visibility),
        settings.matching.chunk_rows,
    )
    opened = compute_open_match(rows, settings)
    agreement = sessions_correlation(stated, opened, rows["session_id"])
    assert agreement > MINIMUM_AGREEMENT
