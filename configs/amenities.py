"""Hotel amenities, room features, traveler tastes and meal costs."""

from configs.amenity_mentions import MENTIONS
from src.settings.amenity_models import (
    AmenitySettings,
    DistanceRecord,
    FlagRecord,
    MealSettings,
    RoomFeatureSettings,
    TasteRecord,
)


def category_taste(
    name: str,
    column: str,
    category: str,
    mean: float,
    archetype_shifts: dict[str, float],
    shifts: tuple[float, float, float],
) -> TasteRecord:
    """Builds a taste for one value of a categorical column."""
    return TasteRecord(
        name=name,
        column=column,
        category=category,
        low=0.0,
        high=1.0,
        invert=False,
        mean=mean,
        archetype_shifts=archetype_shifts,
        kids_shift=shifts[0],
        couple_shift=shifts[1],
        standard_deviation=shifts[2],
    )


def numeric_taste(
    name: str,
    column: str,
    bounds: tuple[float, float, bool],
    mean: float,
    archetype_shifts: dict[str, float],
    shifts: tuple[float, float, float],
) -> TasteRecord:
    """Builds a taste for a numeric column scaled between two bounds."""
    return TasteRecord(
        name=name,
        column=column,
        category=None,
        low=bounds[0],
        high=bounds[1],
        invert=bounds[2],
        mean=mean,
        archetype_shifts=archetype_shifts,
        kids_shift=shifts[0],
        couple_shift=shifts[1],
        standard_deviation=shifts[2],
    )


FLAGS: list[FlagRecord] = [
    FlagRecord(
        column="has_kids_club",
        base=0.0,
        star_weight=0.02,
        aspect_weights={"family_friendly": 0.7},
        type_boosts={
            "family_club": 0.35,
            "beach_resort": 0.1,
            "city_hotel": -0.1,
            "boutique": -0.1,
            "party_resort": -0.1,
            "spa_retreat": -0.1,
        },
    ),
    FlagRecord(
        column="has_pool",
        base=0.3,
        star_weight=0.08,
        aspect_weights={"beach_sea": 0.15},
        type_boosts={
            "beach_resort": 0.35,
            "family_club": 0.3,
            "luxury_resort": 0.3,
            "spa_retreat": 0.25,
            "party_resort": 0.3,
            "city_hotel": -0.15,
            "ski_lodge": -0.1,
            "budget_inn": -0.15,
        },
    ),
    FlagRecord(
        column="has_spa",
        base=-0.1,
        star_weight=0.06,
        aspect_weights={"wellness_spa": 0.5, "luxury": 0.2},
        type_boosts={
            "spa_retreat": 0.5,
            "luxury_resort": 0.3,
            "ski_lodge": 0.1,
            "budget_inn": -0.1,
        },
    ),
    FlagRecord(
        column="free_transfer",
        base=0.2,
        star_weight=0.03,
        aspect_weights={},
        type_boosts={
            "beach_resort": 0.2,
            "family_club": 0.2,
            "luxury_resort": 0.25,
            "party_resort": 0.1,
            "city_hotel": -0.1,
            "budget_inn": -0.1,
        },
    ),
    FlagRecord(
        column="free_wifi",
        base=0.9,
        star_weight=0.02,
        aspect_weights={},
        type_boosts={},
    ),
    FlagRecord(
        column="has_gym",
        base=0.25,
        star_weight=0.08,
        aspect_weights={},
        type_boosts={"city_hotel": 0.1, "luxury_resort": 0.2},
    ),
    FlagRecord(
        column="has_parking",
        base=0.5,
        star_weight=0.0,
        aspect_weights={},
        type_boosts={"ski_lodge": 0.3, "city_hotel": -0.2, "adventure_lodge": 0.2},
    ),
    FlagRecord(
        column="pets_allowed",
        base=0.15,
        star_weight=0.0,
        aspect_weights={},
        type_boosts={
            "adventure_lodge": 0.15,
            "budget_inn": 0.1,
            "luxury_resort": -0.1,
        },
    ),
]


DISTANCES: list[DistanceRecord] = [
    DistanceRecord(
        column="beach_distance_m",
        aspect="beach_sea",
        near_meters=20.0,
        far_meters=4000.0,
        noise_standard_deviation=0.6,
        type_factors={
            "beach_resort": 0.8,
            "party_resort": 0.9,
            "family_club": 1.0,
            "spa_retreat": 1.3,
            "adventure_lodge": 1.5,
            "city_hotel": 1.6,
            "ski_lodge": 2.0,
        },
    ),
    DistanceRecord(
        column="center_distance_m",
        aspect="city_culture",
        near_meters=50.0,
        far_meters=6000.0,
        noise_standard_deviation=0.6,
        type_factors={
            "city_hotel": 0.5,
            "boutique": 0.7,
            "family_club": 1.5,
            "beach_resort": 1.8,
            "spa_retreat": 2.0,
            "adventure_lodge": 2.0,
        },
    ),
]


ROOM_FEATURES = RoomFeatureSettings(
    size_column="room_size_sqm",
    balcony_column="has_balcony",
    size_by_category={
        "standard": 20.0,
        "superior": 25.0,
        "family_room": 35.0,
        "junior_suite": 40.0,
        "suite": 55.0,
        "villa": 110.0,
    },
    size_noise_standard_deviation=0.15,
    balcony_by_category={
        "standard": 0.45,
        "superior": 0.6,
        "family_room": 0.55,
        "junior_suite": 0.75,
        "suite": 0.9,
        "villa": 0.95,
    },
    balcony_view_bonus={
        "sea": 0.2,
        "mountain": 0.2,
        "city": 0.1,
        "pool": 0.05,
        "garden": 0.0,
        "none": -0.2,
    },
)


TASTES: list[TasteRecord] = [
    category_taste(
        "board_all_inclusive",
        "board_type",
        "AI",
        0.2,
        {
            "family_beach": 0.4,
            "family_adventure": 0.2,
            "party_break": 0.2,
            "budget_sun": 0.3,
            "luxury_escape": -0.1,
            "culture_city": -0.4,
            "quiet_nature": -0.1,
        },
        (0.3, 0.0, 0.5),
    ),
    category_taste(
        "board_breakfast",
        "board_type",
        "BB",
        0.15,
        {"culture_city": 0.2, "romantic_getaway": 0.1},
        (-0.1, 0.0, 0.35),
    ),
    category_taste(
        "board_half_board",
        "board_type",
        "HB",
        0.15,
        {"wellness_retreat": 0.2, "ski_trip": 0.2, "quiet_nature": 0.1},
        (0.0, 0.0, 0.35),
    ),
    category_taste(
        "view_sea",
        "room_view",
        "sea",
        0.1,
        {"romantic_getaway": 0.4, "family_beach": 0.1, "luxury_escape": 0.3},
        (0.0, 0.15, 0.4),
    ),
    category_taste(
        "view_mountain",
        "room_view",
        "mountain",
        0.1,
        {"ski_trip": 0.3, "quiet_nature": 0.2, "romantic_getaway": 0.2},
        (0.0, 0.0, 0.4),
    ),
    category_taste(
        "view_city",
        "room_view",
        "city",
        0.05,
        {"culture_city": 0.3, "friends_getaway": 0.1, "romantic_getaway": 0.1},
        (0.0, 0.0, 0.3),
    ),
    numeric_taste(
        "beach_closeness",
        "beach_distance_m",
        (0.0, 1500.0, True),
        0.15,
        {
            "family_beach": 0.7,
            "budget_sun": 0.4,
            "party_break": 0.3,
            "friends_getaway": 0.2,
        },
        (0.3, 0.0, 0.5),
    ),
    numeric_taste(
        "center_closeness",
        "center_distance_m",
        (0.0, 3000.0, True),
        0.1,
        {
            "culture_city": 0.6,
            "party_break": 0.4,
            "friends_getaway": 0.3,
            "wellness_retreat": -0.3,
            "quiet_nature": -0.3,
            "family_beach": -0.1,
        },
        (0.0, 0.0, 0.4),
    ),
    numeric_taste(
        "kids_club",
        "has_kids_club",
        (0.0, 1.0, False),
        0.0,
        {},
        (0.9, 0.0, 0.3),
    ),
    numeric_taste(
        "pool",
        "has_pool",
        (0.0, 1.0, False),
        0.25,
        {"family_beach": 0.2},
        (0.3, 0.0, 0.4),
    ),
    numeric_taste(
        "spa",
        "has_spa",
        (0.0, 1.0, False),
        0.0,
        {
            "wellness_retreat": 0.8,
            "romantic_getaway": 0.3,
            "luxury_escape": 0.3,
            "quiet_nature": 0.1,
        },
        (0.0, 0.0, 0.35),
    ),
    numeric_taste(
        "balcony",
        "has_balcony",
        (0.0, 1.0, False),
        0.2,
        {"romantic_getaway": 0.3, "luxury_escape": 0.2},
        (-0.1, 0.0, 0.35),
    ),
    numeric_taste(
        "room_size",
        "room_size_sqm",
        (15.0, 60.0, False),
        0.2,
        {"luxury_escape": 0.3, "budget_sun": -0.2},
        (0.5, 0.0, 0.45),
    ),
    numeric_taste(
        "transfer",
        "free_transfer",
        (0.0, 1.0, False),
        0.2,
        {"ski_trip": 0.2, "budget_sun": -0.05},
        (0.3, 0.0, 0.4),
    ),
]


AMENITIES = AmenitySettings(
    random_stream=1,
    star_reference=3,
    minimum_probability=0.02,
    maximum_probability=0.98,
    rounding_meters=10.0,
    flags=FLAGS,
    distances=DISTANCES,
    rooms=ROOM_FEATURES,
    tastes=TASTES,
    meals=MealSettings(
        cost_per_price_level=0.3,
        uncovered_share={"RO": 1.0, "BB": 0.7, "HB": 0.35, "AI": 0.0},
    ),
    mentions=MENTIONS,
)
