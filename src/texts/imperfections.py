"""Adds realistic writing imperfections to generated texts after the fact."""

from collections.abc import Callable
import json

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.texts.models import TextBundle, TextJobs

IMPERFECTION_DIMENSION = "imperfections"
WORD_SEPARATOR = " "


def swap_letters(word: str, position: int) -> str:
    """Swaps the letter at a position with the next one."""
    letters = list(word)
    letters[position], letters[position + 1] = letters[position + 1], letters[position]
    return "".join(letters)


def drop_letter(word: str, position: int) -> str:
    """Removes the letter at a position."""
    following = position + 1
    return word[:position] + word[following:]


def double_letter(word: str, position: int) -> str:
    """Repeats the letter at a position."""
    return word[:position] + word[position] + word[position:]


LETTER_EDITS: list[Callable[[str, int], str]] = [
    swap_letters,
    drop_letter,
    double_letter,
]


def corrupt_word(random_generator: np.random.Generator, word: str) -> str:
    """Applies one random letter-level mistake inside a word."""
    position = int(random_generator.integers(1, len(word) - 1))
    edit = LETTER_EDITS[int(random_generator.integers(0, len(LETTER_EDITS)))]
    return edit(word, position)


def pick_typo_positions(
    random_generator: np.random.Generator,
    words: list[str],
    settings: Settings,
) -> list[int]:
    """Chooses one or two longer words of a text to corrupt."""
    limit = settings.text.typo_minimum_word_length
    eligible = [index for index, word in enumerate(words) if len(word) >= limit]
    count = int(
        random_generator.choice(
            settings.text.typo_count_values,
            p=settings.text.typo_count_probabilities,
        )
    )
    amount = min(count, len(eligible))
    chosen = random_generator.choice(eligible, size=amount, replace=False)
    return [int(index) for index in chosen]


def inject_typos(
    random_generator: np.random.Generator,
    text: str,
    settings: Settings,
) -> str:
    """Corrupts one or two longer words, however long the text is."""
    words = text.split(WORD_SEPARATOR)
    if not any(len(word) >= settings.text.typo_minimum_word_length for word in words):
        return text
    for index in pick_typo_positions(random_generator, words, settings):
        words[index] = corrupt_word(random_generator, words[index])
    return WORD_SEPARATOR.join(words)


def strip_punctuation(text: str, settings: Settings) -> str:
    """Removes sentence punctuation so the text reads as one run-on line."""
    return text.translate(str.maketrans("", "", settings.text.punctuation_symbols))


def apply_operation(
    random_generator: np.random.Generator,
    operation: str,
    text: str,
    settings: Settings,
) -> str:
    """Applies one named imperfection to a text."""
    if operation == "typos":
        return inject_typos(random_generator, text, settings)
    if operation == "lowercase":
        return text.lower()
    return strip_punctuation(text, settings)


def apply_imperfection(
    random_generator: np.random.Generator,
    text: str,
    label: str,
    settings: Settings,
) -> str:
    """Applies all operations configured for an imperfection label."""
    operations = settings.text.imperfection_operations.get(label, [])
    for operation in operations:
        text = apply_operation(random_generator, operation, text, settings)
    return text


def degrade_job_texts(
    random_generator: np.random.Generator,
    briefs: pd.DataFrame,
    texts: dict[int, str],
    settings: Settings,
) -> dict[int, str]:
    """Applies the imperfection chosen in each brief to its text."""
    result: dict[int, str] = {}
    for identifier, spec_json in zip(briefs["id"], briefs["spec"]):
        if int(identifier) not in texts:
            continue
        label = json.loads(str(spec_json))["labels"].get(IMPERFECTION_DIMENSION, "")
        text = texts[int(identifier)]
        result[int(identifier)] = apply_imperfection(
            random_generator,
            text,
            label,
            settings,
        )
    return result


def apply_imperfections(
    texts: TextBundle,
    jobs: TextJobs,
    settings: Settings,
) -> TextBundle:
    """Returns the texts with their planned writing imperfections applied."""
    generator = np.random.default_rng(settings.run.seed)
    return TextBundle(
        intents=degrade_job_texts(generator, jobs.intents, texts.intents, settings),
        practical_intents=degrade_job_texts(
            generator,
            jobs.practical_intents,
            texts.practical_intents,
            settings,
        ),
        hotels=degrade_job_texts(generator, jobs.hotels, texts.hotels, settings),
        rooms=degrade_job_texts(generator, jobs.rooms, texts.rooms, settings),
    )
