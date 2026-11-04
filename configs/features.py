"""Settings of the feature building stage: text models and model inputs."""

from src.settings.amenity_models import CharacteristicRecord
from src.settings.feature_models import (
    EmbeddingVariant,
    FeatureColumns,
    FeatureSettings,
)

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: {query}"
QWEN_QUERY_TEMPLATE = "Instruct: {task}\nQuery:{query}"
COMBINED_TASK = (
    "Given a traveler's description of the holiday they want, retrieve the "
    "hotel, location and room descriptions that match what they want"
)
PLACE_TASK = (
    "Given a traveler's wish for a holiday, retrieve the destination that "
    "suits it"
)
HOTEL_TASK = (
    "Given a traveler's description of the holiday they want, retrieve the "
    "hotel descriptions that match what they want"
)
ROOM_TASK = (
    "Given a traveler's description of the holiday they want, retrieve the "
    "room descriptions that match what they want"
)


def category_label(column: str, category: str) -> CharacteristicRecord:
    """Describes a label that is a value of a categorical column."""
    return CharacteristicRecord(
        column=column, category=category, low=0.0, high=1.0, invert=False
    )


def numeric_label(
    column: str,
    low: float,
    high: float,
    invert: bool,
) -> CharacteristicRecord:
    """Describes a label that is a scaled numeric column."""
    return CharacteristicRecord(
        column=column, category=None, low=low, high=high, invert=invert
    )


LABEL_CHARACTERISTICS: dict[str, CharacteristicRecord] = {
    "all_inclusive": category_label("board_type", "AI"),
    "breakfast": category_label("board_type", "BB"),
    "half_board": category_label("board_type", "HB"),
    "sea_view": category_label("room_view", "sea"),
    "mountain_view": category_label("room_view", "mountain"),
    "city_view": category_label("room_view", "city"),
    "beach_close": numeric_label("beach_distance_m", 0.0, 1500.0, True),
    "center_close": numeric_label("center_distance_m", 0.0, 3000.0, True),
    "kids_club": numeric_label("has_kids_club", 0.0, 1.0, False),
    "pool": numeric_label("has_pool", 0.0, 1.0, False),
    "spa": numeric_label("has_spa", 0.0, 1.0, False),
    "balcony": numeric_label("has_balcony", 0.0, 1.0, False),
    "spacious_room": numeric_label("room_size_sqm", 15.0, 60.0, False),
    "transfer": numeric_label("free_transfer", 0.0, 1.0, False),
    "wifi": numeric_label("free_wifi", 0.0, 1.0, False),
    "gym": numeric_label("has_gym", 0.0, 1.0, False),
    "parking": numeric_label("has_parking", 0.0, 1.0, False),
    "pets": numeric_label("pets_allowed", 0.0, 1.0, False),
}

LABEL_PHRASES: dict[str, str] = {
    "all_inclusive": "all inclusive",
    "breakfast": "breakfast included",
    "half_board": "half board",
    "sea_view": "sea view",
    "mountain_view": "mountain view",
    "city_view": "city view",
    "beach_close": "right on the beach",
    "center_close": "close to the center",
    "kids_club": "kids club",
    "pool": "swimming pool",
    "spa": "spa",
    "balcony": "balcony",
    "spacious_room": "spacious room",
    "transfer": "airport transfer",
    "wifi": "free wifi",
    "gym": "gym",
    "parking": "parking",
    "pets": "pets allowed",
}

RELATIVE_COLUMNS: list[str] = [
    "price_ratio_vs_competitor",
    "selling_price",
    "price_per_person_night",
    "net_cost",
    "stars",
    "rating",
    "flight_duration_hours",
    "number_of_stops",
    "beach_distance_m",
    "center_distance_m",
    "room_size_sqm",
    "sim_hotel_qwen3_small_instruct",
    "sim_place_qwen3_small_instruct",
    "sim_room_qwen3_small_instruct",
    "sim_stay_qwen3_small_instruct",
    "sim_hotel_bge_large_instruct",
    "sim_full_bge_large_instruct",
    "label_match",
]

FEATURES = FeatureSettings(
    embeddings=[
        EmbeddingVariant(
            key="qwen3_small_instruct",
            repository="Qwen/Qwen3-Embedding-0.6B",
            query_template=QWEN_QUERY_TEMPLATE,
            hotel_task=HOTEL_TASK,
            room_task=ROOM_TASK,
            place_task=PLACE_TASK,
            combined_task=COMBINED_TASK,
            batch_size=64,
            maximum_tokens=512,
        ),
        EmbeddingVariant(
            key="bge_large_instruct",
            repository="BAAI/bge-large-en-v1.5",
            query_template=BGE_QUERY_PREFIX,
            hotel_task=HOTEL_TASK,
            room_task=ROOM_TASK,
            place_task=PLACE_TASK,
            combined_task=COMBINED_TASK,
            batch_size=64,
            maximum_tokens=512,
        ),
    ],
    columns=FeatureColumns(
        categorical=[
            "user_device",
            "user_country",
            "country",
            "city",
            "board_type",
            "room_category",
            "room_view",
        ],
        numeric=[
            "search_hour",
            "search_dow",
            "days_to_departure",
            "nights",
            "n_adults",
            "n_children",
            "departure_month",
            "stars",
            "rating",
            "free_cancellation",
            "number_of_stops",
            "flight_duration_hours",
            "net_cost",
            "margin_percentage",
            "beach_distance_m",
            "center_distance_m",
            "room_size_sqm",
            "has_kids_club",
            "has_pool",
            "has_spa",
            "free_transfer",
            "free_wifi",
            "has_gym",
            "has_parking",
            "pets_allowed",
            "has_balcony",
        ],
        derived=[
            "price_ratio_vs_competitor",
            "selling_price",
            "margin_dollars",
            "price_per_person_night",
            "intent_word_count",
        ],
    ),
    device="cuda",
    half_precision=True,
    chunk_rows=50000,
    cache_hash_length=12,
    id_columns=["session_id", "room_id"],
    document_separator=" ",
    location_prefix="Location:",
    label_phrases=LABEL_PHRASES,
    label_characteristics=LABEL_CHARACTERISTICS,
    label_variant_key="qwen3_small_instruct",
    direct_variant_keys=["qwen3_small_instruct", "bge_large_instruct"],
    direct_components=16,
    pca_fit_rows=20000,
    relative_columns=RELATIVE_COLUMNS,
    evaluation_pool_size=30,
)
