"""What texts say about amenities, and how often."""

from src.settings.amenity_models import (
    DetailRecord,
    DistanceFactRecord,
    FactRecord,
    MentionSettings,
)

DETAILS: list[DetailRecord] = [
    DetailRecord(
        taste="board_all_inclusive",
        minimum_weight=0.5,
        probability=0.25,
        phrases=[
            "all inclusive",
            "all inclusive board",
            "everything included",
            "all meals included",
        ],
    ),
    DetailRecord(
        taste="board_breakfast",
        minimum_weight=0.4,
        probability=0.15,
        phrases=["breakfast included", "breakfast buffet", "breakfast in the hotel"],
    ),
    DetailRecord(
        taste="board_half_board",
        minimum_weight=0.4,
        probability=0.12,
        phrases=["half board", "dinner included", "breakfast and dinner"],
    ),
    DetailRecord(
        taste="view_sea",
        minimum_weight=0.4,
        probability=0.5,
        phrases=["sea view", "room with a sea view", "view of the sea"],
    ),
    DetailRecord(
        taste="view_mountain",
        minimum_weight=0.4,
        probability=0.4,
        phrases=["mountain view", "view of the mountains", "room with mountain view"],
    ),
    DetailRecord(
        taste="view_city",
        minimum_weight=0.35,
        probability=0.2,
        phrases=["city view", "view over the city"],
    ),
    DetailRecord(
        taste="beach_closeness",
        minimum_weight=0.5,
        probability=0.5,
        phrases=[
            "right on the beach",
            "first line from the sea",
            "beach a few minutes away",
        ],
    ),
    DetailRecord(
        taste="center_closeness",
        minimum_weight=0.4,
        probability=0.35,
        phrases=["close to the center", "in the city center", "central location"],
    ),
    DetailRecord(
        taste="kids_club",
        minimum_weight=0.5,
        probability=0.45,
        phrases=["kids club", "club for kids", "kids animation"],
    ),
    DetailRecord(
        taste="pool",
        minimum_weight=0.45,
        probability=0.3,
        phrases=["swimming pool", "pool", "big pool"],
    ),
    DetailRecord(
        taste="spa",
        minimum_weight=0.4,
        probability=0.35,
        phrases=["spa", "spa center", "sauna and spa"],
    ),
    DetailRecord(
        taste="balcony",
        minimum_weight=0.4,
        probability=0.3,
        phrases=["balcony", "room with a balcony", "private balcony"],
    ),
    DetailRecord(
        taste="room_size",
        minimum_weight=0.5,
        probability=0.03,
        phrases=["spacious room", "big room"],
    ),
    DetailRecord(
        taste="transfer",
        minimum_weight=0.4,
        probability=0.08,
        phrases=["airport transfer", "free transfer", "transfer included"],
    ),
]


HOTEL_FACTS: list[FactRecord] = [
    FactRecord(column="has_kids_club", probability=0.8, phrase="has a kids club"),
    FactRecord(column="has_pool", probability=0.7, phrase="has a swimming pool"),
    FactRecord(column="has_spa", probability=0.8, phrase="has a spa center"),
    FactRecord(
        column="free_transfer",
        probability=0.5,
        phrase="offers airport transfer",
    ),
    FactRecord(column="free_wifi", probability=0.4, phrase="has free wifi"),
    FactRecord(column="has_gym", probability=0.4, phrase="has a gym"),
    FactRecord(column="has_parking", probability=0.4, phrase="has free parking"),
    FactRecord(column="pets_allowed", probability=0.5, phrase="allows pets"),
]


DISTANCE_FACTS: list[DistanceFactRecord] = [
    DistanceFactRecord(
        column="beach_distance_m",
        probability=0.7,
        bounds=[100.0, 400.0, 1000.0],
        phrases=[
            "right on the beach",
            "a short walk from the beach",
            "about ten minutes on foot from the beach",
        ],
    ),
    DistanceFactRecord(
        column="center_distance_m",
        probability=0.5,
        bounds=[300.0, 1000.0, 2500.0],
        phrases=[
            "in the heart of the town",
            "an easy walk from the center",
            "a short ride from the center",
        ],
    ),
]


MENTIONS = MentionSettings(
    details=DETAILS,
    maximum_details=2,
    vibe_detail_factor=0.5,
    noise_probability=0.08,
    noise_phrases=["free wifi", "gym", "parking", "pets allowed", "pet friendly"],
    hotel_facts=HOTEL_FACTS,
    distance_facts=DISTANCE_FACTS,
    balcony_probability=0.8,
    balcony_phrase="has a balcony",
    size_probability=0.6,
    size_small_maximum=22.0,
    size_small_phrase="a compact room",
    size_large_minimum=40.0,
    size_large_phrase="a spacious room",
)
