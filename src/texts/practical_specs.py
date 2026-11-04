"""Turns hidden traveler wishes into briefs for practical-register messages."""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.settings.records import (
    FORMAT_DIMENSION,
    PracticalFormatOption,
    PracticalSettings,
    TextGenerationSettings,
)
from src.texts.details import choose_phrase, pick_query_details
from src.texts.models import PracticalSpec
from src.texts.selection import select_practical_sessions
from src.texts.specs import (
    BUDGET_HIGH,
    BUDGET_LOW,
    BUDGET_NEUTRAL,
    build_intent_styles,
    choose_option,
    pick_avoided_aspects,
    pick_examples,
    pick_wanted_aspects,
    read_taste_matrix,
    read_wish_matrix,
)
from src.world.helpers import read_floats, read_integers
from src.world.models import FloatArray


def pick_generic_phrases(
    random_generator: np.random.Generator,
    taken: list[str],
    practical: PracticalSettings,
    limit: int,
) -> list[str]:
    """Picks requirements that do not depend on the traveler's wishes."""
    probabilities = practical.generic_count_probabilities
    values = practical.generic_count_values
    count = int(random_generator.choice(values, p=probabilities))
    pool: list[str] = []
    for phrase in practical.generic_phrases:
        if phrase not in taken:
            pool.append(phrase)
    amount = max(0, min(count, len(pool), limit))
    picked = random_generator.choice(len(pool), size=amount, replace=False)
    return [pool[int(index)] for index in picked]


def pick_requirements(
    random_generator: np.random.Generator,
    wanted: list[int],
    names: list[str],
    practical: PracticalSettings,
    details: list[str],
) -> list[str]:
    """Turns the strongest wishes and tastes into practical requirements."""
    capacity = max(option.maximum_requirements for option in practical.formats)
    requirements: list[str] = []
    for index in wanted:
        options = practical.aspect_phrases.get(names[index], [])
        if options:
            requirements.append(choose_phrase(random_generator, options))
    requirements = (requirements + details)[:capacity]
    limit = capacity - len(requirements)
    generic = pick_generic_phrases(random_generator, requirements, practical, limit)
    return requirements + generic


def pick_practical_avoids(
    random_generator: np.random.Generator,
    avoided: list[int],
    names: list[str],
    practical: PracticalSettings,
) -> list[str]:
    """Turns strong dislikes into practical exclusions."""
    avoids: list[str] = []
    for index in avoided:
        options = practical.avoid_phrases.get(names[index], [])
        if options:
            avoids.append(choose_phrase(random_generator, options))
    return avoids


def choose_budget_key(
    random_generator: np.random.Generator,
    log_budget: float,
    text: TextGenerationSettings,
) -> str | None:
    """Decides whether the budget is mentioned and how it is described."""
    if random_generator.random() >= text.budget_mention_probability:
        return None
    if log_budget < -text.budget_threshold:
        return BUDGET_LOW
    if log_budget > text.budget_threshold:
        return BUDGET_HIGH
    if random_generator.random() < text.budget_neutral_probability:
        return BUDGET_NEUTRAL
    return None


def pick_practical_budget(
    random_generator: np.random.Generator,
    log_budget: float,
    settings: Settings,
) -> str | None:
    """Picks a plain phrase about the budget, if one is mentioned."""
    key = choose_budget_key(random_generator, log_budget, settings.text)
    if key is None:
        return None
    return choose_phrase(random_generator, settings.practical.budget_phrases[key])


def choose_format(
    random_generator: np.random.Generator,
    formats: list[PracticalFormatOption],
) -> PracticalFormatOption:
    """Draws the format of a practical message."""
    label = choose_option(random_generator, formats).label
    return find_practical_format(formats, label)


def find_practical_format(
    formats: list[PracticalFormatOption],
    label: str,
) -> PracticalFormatOption:
    """Finds the practical format with the given label."""
    for option in formats:
        if option.label == label:
            return option
    raise KeyError(label)


def apply_format(
    spec: PracticalSpec,
    option: PracticalFormatOption,
    fallback_generator: np.random.Generator,
    settings: Settings,
) -> None:
    """Switches a practical brief to another format and its examples."""
    spec.styles[FORMAT_DIMENSION] = option.instruction
    spec.labels[FORMAT_DIMENSION] = option.label
    spec.maximum_words = option.maximum_words
    count = settings.text.example_count
    spec.examples = pick_examples(fallback_generator, option, count)


def fit_format_to_requirements(
    spec: PracticalSpec,
    fallback_generator: np.random.Generator,
    settings: Settings,
) -> PracticalSpec:
    """Moves a brief with many requirements to a format that can hold them."""
    formats = settings.practical.formats
    current = find_practical_format(formats, spec.labels[FORMAT_DIMENSION])
    count = len(spec.requirements)
    if count <= current.maximum_requirements:
        return spec
    allowed = [option for option in formats if option.maximum_requirements >= count]
    label = choose_option(fallback_generator, allowed).label
    option = find_practical_format(formats, label)
    apply_format(spec, option, fallback_generator, settings)
    return spec


def build_practical_spec(
    random_generator: np.random.Generator,
    wishes: FloatArray,
    tastes: FloatArray,
    log_budget: float,
    settings: Settings,
) -> PracticalSpec:
    """Builds the brief for one practical traveler message."""
    practical = settings.practical
    names = settings.catalogue.aspects.names
    wanted = pick_wanted_aspects(random_generator, wishes, settings.text)
    avoided = pick_avoided_aspects(random_generator, wishes, settings.text)
    option = choose_format(random_generator, practical.formats)
    (styles, labels), examples = build_intent_styles(
        random_generator,
        option,
        practical.dimensions,
        settings,
    )
    return PracticalSpec(
        requirements=pick_requirements(
            random_generator,
            wanted,
            names,
            practical,
            pick_query_details(random_generator, tastes, settings, 1.0),
        ),
        avoids=pick_practical_avoids(random_generator, avoided, names, practical),
        budget=pick_practical_budget(random_generator, log_budget, settings),
        examples=examples,
        maximum_words=option.maximum_words,
        styles=styles,
        labels=labels,
    )


def build_fallback_generator(settings: Settings) -> np.random.Generator:
    """Creates the separate random stream used to refit briefs to formats."""
    run = settings.run
    return np.random.default_rng(run.seed + run.practical_seed_offset + 1)


def build_practical_specs(
    random_generator: np.random.Generator,
    selected: pd.DataFrame,
    settings: Settings,
) -> list[str]:
    """Builds the practical briefs of the selected sessions as JSON."""
    fallback = build_fallback_generator(settings)
    wishes = read_wish_matrix(selected, settings)
    tastes = read_taste_matrix(selected, settings)
    budgets = np.log(read_floats(selected, "hid_budget_mult"))
    specs: list[str] = []
    for row in range(len(selected)):
        spec = build_practical_spec(
            random_generator,
            wishes[row],
            tastes[row],
            float(budgets[row]),
            settings,
        )
        fitted = fit_format_to_requirements(spec, fallback, settings)
        specs.append(fitted.model_dump_json())
    return specs


def build_practical_jobs(
    random_generator: np.random.Generator,
    sessions: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Creates a practical message brief for the selected sessions."""
    mask = select_practical_sessions(len(sessions), settings)
    selected = sessions.iloc[np.flatnonzero(mask)]
    specs = build_practical_specs(random_generator, selected, settings)
    return pd.DataFrame({"id": read_integers(selected, "session_id"), "spec": specs})
