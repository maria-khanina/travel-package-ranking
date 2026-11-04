"""Typed models for the reference tables and the text-related settings."""

from pydantic import BaseModel


class DestinationRecord(BaseModel):
    city: str
    country: str
    latitude: float
    longitude: float
    price_level: float
    competition: float
    climate_offset: float
    ski: bool
    aspect_levels: dict[str, float]


class HubRecord(BaseModel):
    name: str
    latitude: float
    longitude: float
    weight: float


class OriginRecord(BaseModel):
    weight: float
    budget: float
    alpha: float
    hubs: list[HubRecord]


class ArchetypeRecord(BaseModel):
    budget_shift: float
    wishes: dict[str, float]
    multipliers: dict[str, float]


class HotelTypeRecord(BaseModel):
    name: str
    key_aspect: str | None
    base_weight: float
    requires_ski: bool
    requires_aspect: str | None
    requires_level: float
    shift: dict[str, float]
    stars: list[int]
    boards: list[str]
    name_cores: list[str]


class RoomCategoryRecord(BaseModel):
    price_multiplier: float
    weight: float
    minimum_stars: int
    hotel_types: list[str] | None
    gate_aspect: str | None
    gate_level: float
    weight_when_gate_unmet: float
    shift: dict[str, float]


class RoomViewRecord(BaseModel):
    price_multiplier: float
    weight: float
    gate_aspect: str | None
    gate_level: float
    weight_when_gate_unmet: float
    shift: dict[str, float]


class RoomSettings(BaseModel):
    count_values: list[int]
    count_probabilities: list[float]
    maximum_tries: int
    categories: dict[str, RoomCategoryRecord]
    views: dict[str, RoomViewRecord]


class WordLimits(BaseModel):
    minimum: int
    maximum: int


class TextGenerationSettings(BaseModel):
    base_url: str
    temperature: float
    top_p: float
    attempts: int
    concurrency: int
    request_timeout_seconds: float
    retry_backoff_seconds: float
    progress_every: int
    api_key: str
    maximum_tokens: dict[str, int]
    mention_count_values: list[int]
    mention_count_probabilities: list[float]
    want_threshold: float
    maximum_wants: int
    avoid_threshold: float
    maximum_avoids: int
    avoid_mention_probability: float
    budget_mention_probability: float
    budget_threshold: float
    budget_neutral_probability: float
    strong_level: float
    maximum_strong_features: int
    medium_level: float
    medium_mention_probability: float
    maximum_medium_features: int
    intent_words: WordLimits
    hotel_words: WordLimits
    room_words: WordLimits
    banned_place_words: list[str]
    banned_date_words: list[str]
    number_words: list[str]
    party_nouns: list[str]
    party_phrases: list[str]
    party_number_gap: int
    chatter_prefixes: list[str]
    price_symbols: list[str]
    markdown_symbols: list[str]
    poetic_words: list[str]
    imperfection_operations: dict[str, list[str]]
    typo_count_values: list[int]
    typo_count_probabilities: list[float]
    example_count: int
    disallowed_characters: str
    repeat_sequence_size: int
    repeat_share: float
    repeat_warmup_count: int
    repeat_minimum_count: int
    typo_minimum_word_length: int
    punctuation_symbols: str


class PhraseSettings(BaseModel):
    want: dict[str, str]
    avoid: dict[str, str]
    feature: dict[str, str]
    room_category: dict[str, str]
    room_view: dict[str, str]
    budget: dict[str, str]


FORMAT_DIMENSION = "format"


class StyleOption(BaseModel):
    label: str
    probability: float
    instruction: str


class FormatOption(StyleOption):
    maximum_words: int
    skipped_dimensions: list[str]
    examples: list[str]


class PracticalFormatOption(FormatOption):
    maximum_requirements: int


class PracticalSettings(BaseModel):
    aspect_phrases: dict[str, list[str]]
    avoid_phrases: dict[str, list[str]]
    generic_phrases: list[str]
    budget_phrases: dict[str, list[str]]
    generic_count_values: list[int]
    generic_count_probabilities: list[float]
    formats: list[PracticalFormatOption]
    dimensions: dict[str, list[StyleOption]]


class TextStyleSettings(BaseModel):
    intent_formats: list[FormatOption]
    intent: dict[str, list[StyleOption]]
    hotel: dict[str, list[StyleOption]]
    room: dict[str, list[StyleOption]]


class IntentPrompt(BaseModel):
    system: str
    user: str
    wants_line: str
    avoids_line: str
    budget_line: str
    examples_header: str
    example_line: str


class PracticalPrompt(BaseModel):
    system: str
    user: str
    requirements_line: str
    avoids_line: str
    budget_line: str
    examples_header: str
    example_line: str


class HotelPrompt(BaseModel):
    system: str
    user: str
    emphasize_line: str
    mention_line: str
    facts_line: str


class RoomPrompt(BaseModel):
    system: str
    user: str
    facts_line: str


class PromptSettings(BaseModel):
    style_line: str
    intent: IntentPrompt
    practical: PracticalPrompt
    hotel: HotelPrompt
    room: RoomPrompt


class TestingSettings(BaseModel):
    seed: int
    session_count: int
    small_session_count: int
    conversion_tolerance: float
    maximum_low_star_share: float
    low_star_limit: int
    probe_row_count: int
    cheap_row_count: int
    margin_low: float
    margin_high: float
    cheap_ratio: float
    fair_ratio: float
    cheap_win_without_penalty: float
    cheap_win_with_penalty: float
    sum_tolerance: float
    accepted_intents: list[str]
    rejected_intents: list[str]
    accepted_practical: list[str]
    rejected_practical: list[str]
