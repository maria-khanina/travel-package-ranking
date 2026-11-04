"""Settings of the label screening test."""

from src.settings.screening_models import ScreeningSettings

SCREENING = ScreeningSettings(
    permutations=2000,
    top_labels=7,
    permutation_chunk=250,
    seed=23,
    false_discovery_rate=0.05,
    confirmation_level=0.05,
    selection_folds=[0, 1, 2],
    confirmation_folds=[3, 4],
    truth_taste={
        "all_inclusive": "board_all_inclusive",
        "breakfast": "board_breakfast",
        "half_board": "board_half_board",
        "sea_view": "view_sea",
        "mountain_view": "view_mountain",
        "city_view": "view_city",
        "beach_close": "beach_closeness",
        "center_close": "center_closeness",
        "kids_club": "kids_club",
        "pool": "pool",
        "spa": "spa",
        "balcony": "balcony",
        "spacious_room": "room_size",
        "transfer": "transfer",
    },
    output_file="label_screening.csv",
)
