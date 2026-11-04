"""Turns hidden traveler wishes and hotel aspects into text-writing briefs."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.settings.records import (
    FORMAT_DIMENSION,
    FormatOption,
    PhraseSettings,
    StyleOption,
    TextGenerationSettings,
)
from src.texts.details import pick_hotel_facts, pick_query_details, pick_room_facts
from src.texts.models import HotelSpec, IntentSpec, RoomSpec
from src.texts.selection import select_practical_sessions
from src.world.columns import level_columns, taste_columns, wish_columns
from src.world.helpers import normalize, read_floats, read_integers, read_strings
from src.world.models import FloatArray

BUDGET_LOW = "low"
BUDGET_HIGH = "high"
BUDGET_NEUTRAL = "neutral"

StyleChoice = tuple[dict[str, str], dict[str, str]]


def choose_option(
    random_generator: np.random.Generator,
    options: Sequence[StyleOption],
) -> StyleOption:
    """Draws one style option according to the configured probabilities."""
    weights: list[float] = []
    for option in options:
        weights.append(option.probability)
    probabilities = normalize(np.array(weights))
    return options[int(random_generator.choice(len(options), p=probabilities))]


def find_format(options: Sequence[FormatOption], label: str) -> FormatOption:
    """Finds the message format option with the given label."""
    for option in options:
        if option.label == label:
            return option
    raise KeyError(label)


def sample_styles(
    random_generator: np.random.Generator,
    dimensions: dict[str, list[StyleOption]],
    skipped_dimensions: list[str],
) -> StyleChoice:
    """Draws one option per style dimension, leaving out skipped ones."""
    instructions: dict[str, str] = {}
    labels: dict[str, str] = {}
    for dimension, options in dimensions.items():
        if dimension in skipped_dimensions:
            continue
        option = choose_option(random_generator, options)
        instructions[dimension] = option.instruction
        labels[dimension] = option.label
    return instructions, labels


def pick_examples(
    random_generator: np.random.Generator,
    option: FormatOption,
    count: int,
) -> list[str]:
    """Picks a few example messages written in the chosen format."""
    amount = min(count, len(option.examples))
    picked = random_generator.choice(len(option.examples), size=amount, replace=False)
    return [option.examples[int(index)] for index in picked]


def build_intent_styles(
    random_generator: np.random.Generator,
    option: FormatOption,
    dimensions: dict[str, list[StyleOption]],
    settings: Settings,
) -> tuple[StyleChoice, list[str]]:
    """Draws the style notes and the examples of one traveler message."""
    instructions, labels = sample_styles(
        random_generator,
        dimensions,
        option.skipped_dimensions,
    )
    instructions = {FORMAT_DIMENSION: option.instruction, **instructions}
    labels = {FORMAT_DIMENSION: option.label, **labels}
    examples = pick_examples(random_generator, option, settings.text.example_count)
    return (instructions, labels), examples


def pick_wanted_aspects(
    random_generator: np.random.Generator,
    wishes: FloatArray,
    text: TextGenerationSettings,
) -> list[int]:
    """Chooses which of the strongest wishes the message will mention."""
    order = np.argsort(-wishes)
    strong: list[int] = []
    for index in order[: text.maximum_wants]:
        if wishes[index] >= text.want_threshold:
            strong.append(int(index))
    count = int(
        random_generator.choice(
            text.mention_count_values,
            p=text.mention_count_probabilities,
        )
    )
    return (strong or [int(order[0])])[:count]


def pick_avoided_aspects(
    random_generator: np.random.Generator,
    wishes: FloatArray,
    text: TextGenerationSettings,
) -> list[int]:
    """Chooses which strong dislikes the message will mention."""
    avoided: list[int] = []
    for index in np.argsort(wishes)[: text.maximum_avoids]:
        mentioned = random_generator.random() < text.avoid_mention_probability
        if wishes[index] <= text.avoid_threshold and mentioned:
            avoided.append(int(index))
    return avoided


def pick_budget_phrase(
    random_generator: np.random.Generator,
    log_budget: float,
    text: TextGenerationSettings,
    phrases: PhraseSettings,
) -> str | None:
    """Chooses how, if at all, the message hints at the budget."""
    if random_generator.random() >= text.budget_mention_probability:
        return None
    if log_budget < -text.budget_threshold:
        return phrases.budget[BUDGET_LOW]
    if log_budget > text.budget_threshold:
        return phrases.budget[BUDGET_HIGH]
    if random_generator.random() < text.budget_neutral_probability:
        return phrases.budget[BUDGET_NEUTRAL]
    return None


def pick_intent_wants(
    random_generator: np.random.Generator,
    wishes: FloatArray,
    tastes: FloatArray,
    settings: Settings,
) -> list[str]:
    """Chooses the aspect wishes and concrete details a message mentions."""
    names = settings.catalogue.aspects.names
    wanted = pick_wanted_aspects(random_generator, wishes, settings.text)
    factor = settings.amenities.mentions.vibe_detail_factor
    details = pick_query_details(random_generator, tastes, settings, factor)
    room = max(1, settings.text.maximum_wants - len(details))
    aspects = [settings.phrases.want[names[index]] for index in wanted[:room]]
    return aspects + details


def build_intent_spec(
    random_generator: np.random.Generator,
    wishes: FloatArray,
    tastes: FloatArray,
    log_budget: float,
    format_label: str,
    settings: Settings,
) -> IntentSpec:
    """Builds the brief for one traveler message."""
    names = settings.catalogue.aspects.names
    phrases = settings.phrases
    wants = pick_intent_wants(random_generator, wishes, tastes, settings)
    avoided = pick_avoided_aspects(random_generator, wishes, settings.text)
    option = find_format(settings.text_styles.intent_formats, format_label)
    (styles, labels), examples = build_intent_styles(
        random_generator,
        option,
        settings.text_styles.intent,
        settings,
    )
    return IntentSpec(
        wants=wants,
        avoids=[phrases.avoid[names[index]] for index in avoided],
        budget=pick_budget_phrase(random_generator, log_budget, settings.text, phrases),
        examples=examples,
        maximum_words=option.maximum_words,
        styles=styles,
        labels=labels,
    )


def read_wish_matrix(sessions: pd.DataFrame, settings: Settings) -> FloatArray:
    """Reads the hidden wish weights of every session as a matrix."""
    columns = wish_columns(settings.catalogue.aspects.names)
    return sessions[columns].to_numpy(dtype=np.float64)


def read_taste_matrix(sessions: pd.DataFrame, settings: Settings) -> FloatArray:
    """Reads the hidden taste weights of every session as a matrix."""
    names = [record.name for record in settings.amenities.tastes]
    return sessions[taste_columns(names)].to_numpy(dtype=np.float64)


def build_intent_specs(
    random_generator: np.random.Generator,
    sessions: pd.DataFrame,
    settings: Settings,
) -> list[str]:
    """Builds the message briefs of the given sessions as JSON."""
    wishes = read_wish_matrix(sessions, settings)
    tastes = read_taste_matrix(sessions, settings)
    budgets = np.log(read_floats(sessions, "hid_budget_mult"))
    formats = read_strings(sessions, "hid_text_style")
    specs: list[str] = []
    for row in range(len(sessions)):
        spec = build_intent_spec(
            random_generator,
            wishes[row],
            tastes[row],
            float(budgets[row]),
            formats[row],
            settings,
        )
        specs.append(spec.model_dump_json())
    return specs


def build_intent_jobs(
    random_generator: np.random.Generator,
    sessions: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Creates a message brief for every session that writes in the vibe register."""
    practical = select_practical_sessions(len(sessions), settings)
    vibe = sessions.iloc[np.flatnonzero(~practical)]
    specs = build_intent_specs(random_generator, vibe, settings)
    return pd.DataFrame({"id": read_integers(vibe, "session_id"), "spec": specs})


def add_feature(
    random_generator: np.random.Generator,
    feature: str,
    level: float,
    groups: tuple[list[str], list[str]],
    settings: Settings,
) -> None:
    """Files a hotel feature as strong, medium or not mentioned at all."""
    text = settings.text
    strong, medium = groups
    if level >= text.strong_level and len(strong) < text.maximum_strong_features:
        strong.append(feature)
    elif text.medium_level <= level < text.strong_level:
        if random_generator.random() < text.medium_mention_probability:
            medium.append(feature)


def pick_features(
    random_generator: np.random.Generator,
    levels: FloatArray,
    settings: Settings,
) -> tuple[list[str], list[str]]:
    """Splits a hotel's aspects into emphasized and briefly mentioned ones."""
    names = settings.catalogue.aspects.names
    strong: list[str] = []
    medium: list[str] = []
    for index in np.argsort(-levels):
        feature = settings.phrases.feature[names[index]]
        level = float(levels[index])
        add_feature(random_generator, feature, level, (strong, medium), settings)
    return strong, medium[: settings.text.maximum_medium_features]


def build_hotel_spec(
    random_generator: np.random.Generator,
    hotel: pd.Series,
    levels: FloatArray,
    settings: Settings,
) -> HotelSpec:
    """Builds the brief for one hotel description."""
    strong, medium = pick_features(random_generator, levels, settings)
    styles, labels = sample_styles(random_generator, settings.text_styles.hotel, [])
    return HotelSpec(
        name=str(hotel["hotel_name"]),
        hotel_type=str(hotel["hotel_type"]).replace("_", " "),
        stars=int(hotel["stars"]),
        city=str(hotel["city"]),
        country=str(hotel["country"]),
        emphasize=strong,
        mention=medium,
        facts=pick_hotel_facts(random_generator, hotel, settings.amenities.mentions),
        styles=styles,
        labels=labels,
    )


def build_hotel_jobs(
    random_generator: np.random.Generator,
    hotels: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Creates one description brief per hotel."""
    levels = hotels[level_columns(settings.catalogue.aspects.names)]
    levels = levels.to_numpy(dtype=np.float64)
    specs: list[str] = []
    for position, (_, hotel) in enumerate(hotels.iterrows()):
        spec = build_hotel_spec(random_generator, hotel, levels[position], settings)
        specs.append(spec.model_dump_json())
    return pd.DataFrame({"id": read_integers(hotels, "hotel_id"), "spec": specs})


def build_room_spec(
    random_generator: np.random.Generator,
    room: pd.Series,
    hotel: pd.Series,
    settings: Settings,
) -> RoomSpec:
    """Builds the brief for one room description."""
    phrases = settings.phrases
    styles, labels = sample_styles(random_generator, settings.text_styles.room, [])
    return RoomSpec(
        hotel_type=str(hotel["hotel_type"]).replace("_", " "),
        stars=int(hotel["stars"]),
        city=str(hotel["city"]),
        category=phrases.room_category[str(room["room_category"])],
        view=phrases.room_view[str(room["room_view"])],
        facts=pick_room_facts(random_generator, room, settings),
        styles=styles,
        labels=labels,
    )


def build_room_jobs(
    random_generator: np.random.Generator,
    rooms: pd.DataFrame,
    hotels: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Creates one description brief per room."""
    by_hotel: dict[int, pd.Series] = {}
    for _, hotel_row in hotels.iterrows():
        by_hotel[int(hotel_row["hotel_id"])] = hotel_row
    specs: list[str] = []
    for _, room in rooms.iterrows():
        hotel = by_hotel[int(room["hotel_id"])]
        spec = build_room_spec(random_generator, room, hotel, settings)
        specs.append(spec.model_dump_json())
    return pd.DataFrame({"id": read_integers(rooms, "room_id"), "spec": specs})
