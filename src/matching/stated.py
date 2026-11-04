"""Scores how well a package matches what the traveler message states."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.behavior.taste_terms import encode_feature
from src.settings.loader import Settings
from src.texts.models import HotelSpec, IntentSpec, PracticalSpec
from src.world.models import FloatArray, IntArray

PhraseLookup = dict[str, str]


@dataclass
class PhraseBook:
    want: PhraseLookup
    avoid: PhraseLookup
    feature: PhraseLookup
    detail: PhraseLookup


@dataclass
class Statements:
    wants: FloatArray
    avoids: FloatArray
    tastes: FloatArray


def invert_phrase(mapping: dict[str, str]) -> PhraseLookup:
    """Maps every phrase back to the key it stands for."""
    return {phrase: key for key, phrase in mapping.items()}


def invert_phrase_lists(mapping: dict[str, list[str]]) -> PhraseLookup:
    """Maps every phrase of every list back to the key of its list."""
    lookup: PhraseLookup = {}
    for key, phrases in mapping.items():
        for phrase in phrases:
            lookup[phrase] = key
    return lookup


def build_phrase_book(settings: Settings) -> PhraseBook:
    """Collects the phrase-to-meaning lookups used in the text briefs."""
    detail: PhraseLookup = {}
    for record in settings.amenities.mentions.details:
        detail.update({phrase: record.taste for phrase in record.phrases})
    want = invert_phrase(settings.phrases.want)
    want.update(invert_phrase_lists(settings.practical.aspect_phrases))
    avoid = invert_phrase(settings.phrases.avoid)
    avoid.update(invert_phrase_lists(settings.practical.avoid_phrases))
    return PhraseBook(
        want=want,
        avoid=avoid,
        feature=invert_phrase(settings.phrases.feature),
        detail=detail,
    )


def mark_phrases(
    phrases: list[str],
    lookup: PhraseLookup,
    names: list[str],
    vector: FloatArray,
) -> None:
    """Sets a one in the vector for every phrase that the lookup knows."""
    for phrase in phrases:
        if phrase in lookup:
            vector[names.index(lookup[phrase])] = 1.0


def split_statement(spec_json: str, is_practical: bool) -> tuple[list[str], list[str]]:
    """Reads the stated and the refused phrases of one traveler brief."""
    if is_practical:
        practical = PracticalSpec.model_validate_json(spec_json)
        return practical.requirements, practical.avoids
    intent = IntentSpec.model_validate_json(spec_json)
    return intent.wants, intent.avoids


def read_statement(
    spec_json: str,
    is_practical: bool,
    book: PhraseBook,
    names: tuple[list[str], list[str]],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Turns one traveler brief into wanted, avoided and detail indicators."""
    aspects, tastes = names
    stated, refused = split_statement(spec_json, is_practical)
    wants = np.zeros(len(aspects))
    avoids = np.zeros(len(aspects))
    details = np.zeros(len(tastes))
    mark_phrases(stated, book.want, aspects, wants)
    mark_phrases(stated, book.detail, tastes, details)
    mark_phrases(refused, book.avoid, aspects, avoids)
    return wants, avoids, details


def fill_statements(
    result: Statements,
    table: pd.DataFrame,
    is_practical: bool,
    book: PhraseBook,
    names: tuple[list[str], list[str]],
) -> None:
    """Stores what each session of a brief table states."""
    for session, spec in zip(table["id"], table["spec"]):
        wants, avoids, details = read_statement(str(spec), is_practical, book, names)
        index = int(session)
        result.wants[index] = wants
        result.avoids[index] = avoids
        result.tastes[index] = details


def build_statements(
    briefs: tuple[pd.DataFrame, pd.DataFrame],
    session_count: int,
    book: PhraseBook,
    settings: Settings,
) -> Statements:
    """Collects what every session states, indexed by session identifier."""
    aspects = settings.catalogue.aspects.names
    tastes = [record.name for record in settings.amenities.tastes]
    result = Statements(
        wants=np.zeros((session_count, len(aspects))),
        avoids=np.zeros((session_count, len(aspects))),
        tastes=np.zeros((session_count, len(tastes))),
    )
    fill_statements(result, briefs[0], False, book, (aspects, tastes))
    fill_statements(result, briefs[1], True, book, (aspects, tastes))
    return result


def show_phrases(
    row: FloatArray,
    phrases: list[str],
    book: PhraseBook,
    level: tuple[list[str], float],
) -> None:
    """Records the strength with which a hotel brief shows some aspects."""
    names, strength = level
    for phrase in phrases:
        row[names.index(book.feature[phrase])] = strength


def build_visibility(
    hotels: pd.DataFrame,
    book: PhraseBook,
    settings: Settings,
) -> FloatArray:
    """Gives each hotel the aspect strength that its description brief shows."""
    names = settings.catalogue.aspects.names
    matching = settings.matching
    visibility = np.zeros((int(hotels["id"].max()) + 1, len(names)))
    for hotel, spec_json in zip(hotels["id"], hotels["spec"]):
        spec = HotelSpec.model_validate_json(str(spec_json))
        row = visibility[int(hotel)]
        show_phrases(row, spec.mention, book, (names, matching.medium_visibility))
        show_phrases(row, spec.emphasize, book, (names, matching.strong_visibility))
    return visibility


def build_characteristics(rows: pd.DataFrame, settings: Settings) -> FloatArray:
    """Reads, for every row, the values of the characteristics people state."""
    columns = [encode_feature(rows, record) for record in settings.amenities.tastes]
    return np.stack(columns, axis=1)


def stated_scores(
    sessions: IntArray,
    hotels: IntArray,
    characteristics: FloatArray,
    context: tuple[Statements, FloatArray],
    chunk_rows: int,
) -> FloatArray:
    """Scores every row by the stated wishes it satisfies, minus those it breaks."""
    statements, visibility = context
    net = statements.wants - statements.avoids
    scores = np.empty(len(sessions), dtype=np.float64)
    for start in range(0, len(sessions), chunk_rows):
        part = slice(start, start + chunk_rows)
        aspect = np.einsum("ij,ij->i", net[sessions[part]], visibility[hotels[part]])
        taste = np.einsum(
            "ij,ij->i", statements.tastes[sessions[part]], characteristics[part]
        )
        scores[part] = aspect + taste
    return scores
