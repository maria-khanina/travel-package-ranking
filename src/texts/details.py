"""Chooses the concrete amenity details that the texts mention."""

import numpy as np
import pandas as pd

from src.settings.amenity_models import (
    DistanceFactRecord,
    MentionSettings,
)
from src.settings.loader import Settings
from src.world.models import FloatArray


def collect_fact_phrases(mentions: MentionSettings) -> list[str]:
    """Lists the phrases a hotel or room text may be told to state."""
    phrases = [record.phrase for record in mentions.hotel_facts]
    for distance in mentions.distance_facts:
        phrases += distance.phrases
    phrases += [mentions.balcony_phrase, mentions.size_small_phrase]
    return phrases + [mentions.size_large_phrase]


def collect_detail_phrases(mentions: MentionSettings) -> list[str]:
    """Lists the phrases a traveler message may be told to include."""
    phrases = list(mentions.noise_phrases)
    for detail in mentions.details:
        phrases += detail.phrases
    return phrases


def collect_required_phrases(settings: Settings) -> list[str]:
    """Lists every phrase that a text brief may require a text to say."""
    mentions = settings.amenities.mentions
    phrases = list(settings.phrases.feature.values())
    phrases += list(settings.phrases.want.values())
    return phrases + collect_fact_phrases(mentions) + collect_detail_phrases(mentions)


def choose_phrase(random_generator: np.random.Generator, options: list[str]) -> str:
    """Picks one phrase uniformly from a list."""
    return options[int(random_generator.integers(0, len(options)))]


def pick_noise_detail(
    random_generator: np.random.Generator,
    mentions: MentionSettings,
    factor: float,
) -> list[str]:
    """Sometimes adds a detail that has no effect on the traveler's choice."""
    if random_generator.random() >= mentions.noise_probability * factor:
        return []
    return [choose_phrase(random_generator, mentions.noise_phrases)]


def choose_details(
    random_generator: np.random.Generator,
    tastes: FloatArray,
    settings: Settings,
    factor: float,
) -> list[tuple[float, str]]:
    """Draws, for each taste, whether a message spells it out."""
    names = [record.name for record in settings.amenities.tastes]
    chosen: list[tuple[float, str]] = []
    for record in settings.amenities.mentions.details:
        weight = float(tastes[names.index(record.taste)])
        drawn = random_generator.random() < record.probability * factor
        if weight >= record.minimum_weight and drawn:
            chosen.append((weight, choose_phrase(random_generator, record.phrases)))
    return chosen


def pick_query_details(
    random_generator: np.random.Generator,
    tastes: FloatArray,
    settings: Settings,
    factor: float,
) -> list[str]:
    """Chooses which strongly held tastes a traveler message spells out."""
    mentions = settings.amenities.mentions
    chosen = choose_details(random_generator, tastes, settings, factor)
    chosen.sort(reverse=True)
    phrases = [phrase for _, phrase in chosen[: mentions.maximum_details]]
    return phrases + pick_noise_detail(random_generator, mentions, factor)


def bucket_phrase(meters: float, record: DistanceFactRecord) -> str | None:
    """Describes a distance in words, or nothing when it is too far."""
    for bound, phrase in zip(record.bounds, record.phrases):
        if meters <= bound:
            return phrase
    return None


def pick_hotel_facts(
    random_generator: np.random.Generator,
    hotel: pd.Series,
    mentions: MentionSettings,
) -> list[str]:
    """Chooses which real amenities and distances a hotel text states."""
    facts: list[str] = []
    for record in mentions.hotel_facts:
        present = bool(hotel[record.column])
        if present and random_generator.random() < record.probability:
            facts.append(record.phrase)
    for distance in mentions.distance_facts:
        phrase = bucket_phrase(float(hotel[distance.column]), distance)
        if phrase is not None and random_generator.random() < distance.probability:
            facts.append(phrase)
    return facts


def pick_balcony_fact(
    random_generator: np.random.Generator,
    room: pd.Series,
    settings: Settings,
) -> list[str]:
    """States the balcony of a room, sometimes."""
    mentions = settings.amenities.mentions
    has_balcony = bool(room[settings.amenities.rooms.balcony_column])
    if has_balcony and random_generator.random() < mentions.balcony_probability:
        return [mentions.balcony_phrase]
    return []


def pick_size_fact(
    random_generator: np.random.Generator,
    room: pd.Series,
    settings: Settings,
) -> list[str]:
    """States that a room is compact or spacious, sometimes."""
    mentions = settings.amenities.mentions
    size = float(room[settings.amenities.rooms.size_column])
    if random_generator.random() >= mentions.size_probability:
        return []
    if size <= mentions.size_small_maximum:
        return [mentions.size_small_phrase]
    if size >= mentions.size_large_minimum:
        return [mentions.size_large_phrase]
    return []


def pick_room_facts(
    random_generator: np.random.Generator,
    room: pd.Series,
    settings: Settings,
) -> list[str]:
    """Chooses which real features a room text states."""
    balcony = pick_balcony_fact(random_generator, room, settings)
    return balcony + pick_size_fact(random_generator, room, settings)
