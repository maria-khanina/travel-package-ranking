"""Plain-language phrases describing aspects, rooms and budgets."""

from src.settings.records import PhraseSettings


PHRASES = PhraseSettings(
    want={
        "beach_sea": "beach and sea",
        "mountains_ski": "mountains and skiing",
        "city_culture": "city life and culture",
        "nightlife": "nightlife",
        "quiet_relax": "quiet and relaxation",
        "family_friendly": "family friendly",
        "romantic": "romance",
        "wellness_spa": "spa and wellness",
        "active_adventure": "active adventure",
        "luxury": "luxury",
    },
    avoid={
        "beach_sea": "beaches",
        "mountains_ski": "mountains",
        "city_culture": "city sightseeing",
        "nightlife": "nightlife and parties",
        "quiet_relax": "too much quiet",
        "family_friendly": "lots of kids",
        "romantic": "romance",
        "wellness_spa": "spa",
        "active_adventure": "being active",
        "luxury": "luxury and fanciness",
    },
    feature={
        "beach_sea": "beach access",
        "mountains_ski": "mountain setting",
        "city_culture": "central location",
        "nightlife": "nightlife",
        "quiet_relax": "peaceful atmosphere",
        "family_friendly": "family facilities",
        "romantic": "romantic setting",
        "wellness_spa": "spa and wellness",
        "active_adventure": "outdoor activities",
        "luxury": "luxury service",
    },
    room_category={
        "standard": "a comfortable, well-equipped standard room",
        "superior": "a superior room with extra space and upgraded furnishings",
        "family_room": "a spacious family room with extra beds or a sofa bed",
        "junior_suite": "a junior suite with an open-plan sitting area",
        "suite": "a suite with a separate living room and bedroom",
        "villa": "a private villa with its own terrace",
    },
    room_view={
        "sea": "a balcony with sea views",
        "mountain": "a balcony with mountain views",
        "city": "views over the city",
        "pool": "a view of the pool area",
        "garden": "a quiet garden outlook",
        "none": "no particular view",
    },
    budget={
        "low": "tight",
        "high": "generous",
        "neutral": "moderate",
    },
)
