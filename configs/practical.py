"""Practical, everyday-language requirements travelers state about a holiday."""

from src.settings.records import (
    PracticalFormatOption,
    PracticalSettings,
    StyleOption,
)

PRACTICAL = PracticalSettings(
    aspect_phrases={
        "beach_sea": [
            "beach a short walk away",
            "sunbeds on the beach",
            "calm water for swimming",
            "sandy beach",
            "sea within walking distance",
            "beach with umbrellas",
        ],
        "mountains_ski": [
            "ski lift nearby",
            "ski rental close by",
            "slopes for beginners",
            "shuttle to the slopes",
            "slopes close to the hotel",
            "ski storage",
        ],
        "city_culture": [
            "center within walking distance",
            "main sights nearby",
            "bus or metro nearby",
            "restaurants and cafes around",
            "museums close by",
            "shops nearby",
        ],
        "nightlife": [
            "bars nearby",
            "clubs nearby",
            "places to go out in the evening",
            "restaurants open late",
            "close to the nightlife area",
            "easy taxi back at night",
        ],
        "quiet_relax": [
            "quiet at night",
            "not on a busy road",
            "quiet area",
            "no loud music nearby",
            "a quiet room",
            "not too crowded",
        ],
        "family_friendly": [
            "activities for kids",
            "kids pool",
            "playground",
            "kids menu",
            "family room",
            "babysitting",
        ],
        "romantic": [
            "dinner for two",
            "restaurant with a view",
            "private balcony",
            "private pool",
            "wine bar nearby",
            "couples massage",
        ],
        "wellness_spa": [
            "spa",
            "sauna",
            "massage",
            "gym",
            "indoor pool",
            "yoga classes",
        ],
        "active_adventure": [
            "excursions",
            "water sports",
            "bike rental",
            "hiking trails nearby",
            "diving center",
            "sports activities",
        ],
        "luxury": [
            "high-end hotel",
            "good restaurant",
            "large rooms",
            "good service",
            "top-rated hotel",
            "private pool",
        ],
    },
    avoid_phrases={
        "nightlife": ["no loud bars nearby", "far from clubs", "no late-night noise"],
        "family_friendly": ["no kids", "adults only"],
        "quiet_relax": ["not too quiet"],
        "city_culture": ["far from the center"],
        "wellness_spa": ["no spa needed"],
        "luxury": ["nothing fancy"],
        "mountains_ski": ["no mountains"],
    },
    generic_phrases=[
        "good food",
        "breakfast included",
        "wifi",
        "air conditioning",
        "airport transfer",
        "free parking",
        "pool",
        "balcony",
        "all inclusive",
        "restaurant nearby",
        "clean rooms",
        "late check-out",
        "gym",
        "bar",
        "close to the center",
    ],
    budget_phrases={
        "low": ["cheap", "not too expensive", "good price", "on a budget"],
        "high": ["price does not matter", "ready to pay for quality"],
        "neutral": ["fair price", "reasonable price"],
    },
    generic_count_values=[0, 1, 2],
    generic_count_probabilities=[0.4, 0.4, 0.2],
    formats=[
        PracticalFormatOption(
            label="checklist",
            probability=0.35,
            instruction=(
                "a short list of what you want in everyday words, three to "
                "five items separated by commas, no full sentences"
            ),
            maximum_words=24,
            maximum_requirements=6,
            skipped_dimensions=[],
            examples=[
                "beach a short walk away, kids pool, good food",
                "quiet at night, pool, breakfast included",
                "center within walking distance, restaurants around, not too expensive",
                "ski lift nearby, sauna",
                "bars nearby, clubs nearby, cheap",
                "activities for kids, sandy beach, all inclusive",
                "massage, quiet area, good food",
            ],
        ),
        PracticalFormatOption(
            label="plus_list",
            probability=0.15,
            instruction=(
                "what you want joined with plus signs or slashes, very terse"
            ),
            maximum_words=24,
            maximum_requirements=4,
            skipped_dimensions=[],
            examples=[
                "beach + cheap + good food",
                "center + quiet + balcony",
                "pool + kids stuff + all inclusive",
                "ski lift + sauna + not expensive",
                "bars + sea + budget",
                "massage / quiet / breakfast",
            ],
        ),
        PracticalFormatOption(
            label="short_phrase",
            probability=0.25,
            instruction="one short plain phrase of four to nine words",
            maximum_words=12,
            maximum_requirements=2,
            skipped_dimensions=[],
            examples=[
                "cheap hotel by the sea with activities for kids",
                "quiet hotel within walking distance of the center",
                "all inclusive with things to do in the evening",
                "ski holiday on a budget",
                "hotel with a spa and a good restaurant",
                "family hotel with a playground and a pool",
            ],
        ),
        PracticalFormatOption(
            label="plain_sentence",
            probability=0.25,
            instruction=(
                "one or two plain sentences in everyday words, the way you "
                "would text a friend, with no feelings and no decoration"
            ),
            maximum_words=28,
            maximum_requirements=3,
            skipped_dimensions=[],
            examples=[
                "I want a beach within walking distance and good food, nothing fancy.",
                "Somewhere close to the center but quiet at night.",
                "Need activities for kids and a beach close by.",
                "Ski lift nearby and a sauna would be great.",
                "Good food, a pool and not too expensive.",
                "Bars within walking distance, no need for taxis.",
            ],
        ),
    ],
    dimensions={
        "imperfections": [
            StyleOption(
                label="clean",
                probability=0.70,
                instruction="correct spelling and normal punctuation",
            ),
            StyleOption(
                label="typos",
                probability=0.10,
                instruction="contains one or two natural typos",
            ),
            StyleOption(
                label="lowercase",
                probability=0.15,
                instruction="written entirely in lowercase",
            ),
            StyleOption(
                label="run_on",
                probability=0.05,
                instruction="missing punctuation",
            ),
        ],
    },
)
