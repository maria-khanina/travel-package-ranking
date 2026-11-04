"""Checks that generated texts follow the rules of the generation brief."""

import re

from src.settings.records import TextGenerationSettings, WordLimits

DIGIT_PATTERN = re.compile(r"\d")
PARAGRAPH_BREAK = "\n\n"
LIST_MARKER = "\n- "


def build_word_pattern(words: list[str]) -> re.Pattern[str]:
    """Compiles a whole-word matcher for a list of words."""
    escaped = sorted(re.escape(word) for word in words)
    return re.compile(r"\b(" + "|".join(escaped) + r")\b")


def build_party_pattern(text: TextGenerationSettings) -> re.Pattern[str]:
    """Compiles a matcher for explicit party sizes such as two adults."""
    numbers = "|".join(text.number_words)
    nouns = "|".join(text.party_nouns)
    gap = text.party_number_gap
    return re.compile(rf"\b({numbers})\b(?:\s+\w+){{0,{gap}}}?\s+({nouns})\b")


class TextValidator:
    def __init__(
        self,
        text: TextGenerationSettings,
        place_words: list[str],
    ) -> None:
        self.text = text
        self.place_pattern = build_word_pattern(place_words + text.banned_place_words)
        self.date_pattern = build_word_pattern(text.banned_date_words)
        self.party_pattern = build_party_pattern(text)
        self.phrase_pattern = build_word_pattern(text.party_phrases)
        self.alphabet_pattern = re.compile(text.disallowed_characters)
        self.poetic_pattern = build_word_pattern(text.poetic_words)
        prefixes = "|".join(text.chatter_prefixes)
        self.chatter_pattern = re.compile(rf"^({prefixes})\b", re.IGNORECASE)

    def within_limits(self, value: str, limits: WordLimits) -> bool:
        """Checks that the number of words is inside the allowed range."""
        return limits.minimum <= len(value.split()) <= limits.maximum

    def has_symbols(self, value: str, symbols: list[str]) -> bool:
        """Checks whether any of the given symbols appears in the text."""
        return any(symbol in value for symbol in symbols)

    def is_clean(self, value: str) -> bool:
        """Checks the rules shared by every kind of text."""
        no_markdown = not self.has_symbols(value, self.text.markdown_symbols)
        latin_only = not self.alphabet_pattern.search(value)
        return no_markdown and latin_only and not self.chatter_pattern.search(value)

    def has_leaks(self, value: str) -> bool:
        """Checks a traveler message for places, dates, party sizes or digits."""
        lowered = value.lower()
        return bool(
            DIGIT_PATTERN.search(value)
            or self.place_pattern.search(lowered)
            or self.date_pattern.search(lowered)
            or self.party_pattern.search(lowered)
            or self.phrase_pattern.search(lowered)
        )

    def validate_intent(self, value: str, maximum_words: int) -> bool:
        """Checks a traveler message: vibe only, no hard filters."""
        limits = WordLimits(
            minimum=self.text.intent_words.minimum,
            maximum=maximum_words,
        )
        in_limits = self.within_limits(value, limits)
        one_block = PARAGRAPH_BREAK not in value
        clean = self.is_clean(value) and not self.has_leaks(value)
        return in_limits and one_block and clean

    def validate_practical(self, value: str, maximum_words: int) -> bool:
        """Checks a practical message: an intent without poetic wording."""
        plain = not self.poetic_pattern.search(value.lower())
        return plain and self.validate_intent(value, maximum_words)

    def validate_hotel(self, value: str) -> bool:
        """Checks a hotel description: no prices and no list formatting."""
        no_prices = not self.has_symbols(value, self.text.price_symbols)
        in_limits = self.within_limits(value, self.text.hotel_words)
        no_list = LIST_MARKER not in value
        return in_limits and no_prices and no_list and self.is_clean(value)

    def validate_room(self, value: str) -> bool:
        """Checks a room description: no prices and no digits."""
        no_prices = not self.has_symbols(value, self.text.price_symbols)
        in_limits = self.within_limits(value, self.text.room_words)
        no_digits = not DIGIT_PATTERN.search(value)
        return in_limits and no_prices and no_digits and self.is_clean(value)
