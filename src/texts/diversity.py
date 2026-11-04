"""Keeps generated texts varied by rejecting overused word sequences."""

from collections import Counter
from collections.abc import Callable
import re

from src.settings.records import TextGenerationSettings

WORD_PATTERN = re.compile(r"[a-z']+")
WordSequence = tuple[str, ...]


def split_words(text: str) -> WordSequence:
    """Splits a text into lowercase words."""
    return tuple(WORD_PATTERN.findall(text.lower()))


def contains_run(words: WordSequence, run: WordSequence) -> bool:
    """Checks whether a run of words appears inside a longer word sequence."""
    for start in range(len(words) - len(run) + 1):
        stop = start + len(run)
        if words[start:stop] == run:
            return True
    return False


def is_exempt(sequence: WordSequence, exempt: list[WordSequence]) -> bool:
    """Checks whether a word sequence overlaps a phrase the brief requires."""
    for phrase in exempt:
        if contains_run(sequence, phrase) or contains_run(phrase, sequence):
            return True
    return False


def extract_sequences(text: str, size: int) -> set[WordSequence]:
    """Lists the distinct runs of consecutive words of a given length."""
    words = WORD_PATTERN.findall(text.lower())
    sequences: set[WordSequence] = set()
    for start in range(len(words) - size + 1):
        end = start + size
        sequences.add(tuple(words[start:end]))
    return sequences


class DiversityFilter:
    def __init__(
        self,
        text: TextGenerationSettings,
        exempt: list[WordSequence],
    ) -> None:
        self.text = text
        self.exempt = exempt
        self.counts: Counter[WordSequence] = Counter()
        self.accepted = 0

    def limit(self) -> float:
        """Computes how many accepted texts may share one word sequence."""
        allowed_share = self.text.repeat_share * self.accepted
        return max(self.text.repeat_minimum_count, allowed_share)

    def sequences(self, value: str) -> set[WordSequence]:
        """Lists the word sequences of a text that the filter watches."""
        found = extract_sequences(value, self.text.repeat_sequence_size)
        return {item for item in found if not is_exempt(item, self.exempt)}

    def allows(self, value: str) -> bool:
        """Checks that a text does not reuse a word sequence too often."""
        if self.accepted < self.text.repeat_warmup_count:
            return True
        limit = self.limit()
        for sequence in self.sequences(value):
            if self.counts[sequence] >= limit:
                return False
        return True

    def register(self, value: str) -> None:
        """Records the word sequences of an accepted text."""
        self.counts.update(self.sequences(value))
        self.accepted += 1

    def wrap(self, validate: Callable[[str], bool]) -> Callable[[str], bool]:
        """Combines a validation rule with the diversity requirement."""

        def check(value: str) -> bool:
            return validate(value) and self.allows(value)

        return check
