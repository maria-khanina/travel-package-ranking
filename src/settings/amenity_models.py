"""Typed models of hotel amenities, room features, tastes and meal costs."""

from pydantic import BaseModel


class FlagRecord(BaseModel):
    column: str
    base: float
    star_weight: float
    aspect_weights: dict[str, float]
    type_boosts: dict[str, float]


class DistanceRecord(BaseModel):
    column: str
    aspect: str
    near_meters: float
    far_meters: float
    noise_standard_deviation: float
    type_factors: dict[str, float]


class RoomFeatureSettings(BaseModel):
    size_column: str
    balcony_column: str
    size_by_category: dict[str, float]
    size_noise_standard_deviation: float
    balcony_by_category: dict[str, float]
    balcony_view_bonus: dict[str, float]


class CharacteristicRecord(BaseModel):
    column: str
    category: str | None
    low: float
    high: float
    invert: bool


class TasteRecord(CharacteristicRecord):
    name: str
    mean: float
    archetype_shifts: dict[str, float]
    kids_shift: float
    couple_shift: float
    standard_deviation: float


class MealSettings(BaseModel):
    cost_per_price_level: float
    uncovered_share: dict[str, float]


class DetailRecord(BaseModel):
    taste: str
    minimum_weight: float
    probability: float
    phrases: list[str]


class FactRecord(BaseModel):
    column: str
    probability: float
    phrase: str


class DistanceFactRecord(BaseModel):
    column: str
    probability: float
    bounds: list[float]
    phrases: list[str]


class MentionSettings(BaseModel):
    details: list[DetailRecord]
    maximum_details: int
    vibe_detail_factor: float
    noise_probability: float
    noise_phrases: list[str]
    hotel_facts: list[FactRecord]
    distance_facts: list[DistanceFactRecord]
    balcony_probability: float
    balcony_phrase: str
    size_probability: float
    size_small_maximum: float
    size_small_phrase: str
    size_large_minimum: float
    size_large_phrase: str


class AmenitySettings(BaseModel):
    random_stream: int
    star_reference: int
    minimum_probability: float
    maximum_probability: float
    rounding_meters: float
    flags: list[FlagRecord]
    distances: list[DistanceRecord]
    rooms: RoomFeatureSettings
    tastes: list[TasteRecord]
    meals: MealSettings
    mentions: MentionSettings
