"""Natural constants and column-name helpers shared by the world builders."""

MONTHS_PER_YEAR = 12
HOURS_PER_DAY = 24
EARTH_RADIUS_KILOMETERS = 6371.0
DESTINATION_PREFIX = "dest_"
LEVEL_PREFIX = "lvl_"
SHIFT_PREFIX = "rsh_"
WISH_PREFIX = "hid_w_"
BOARD_PREFIX = "board_"
TASTE_PREFIX = "hid_t_"
MEAL_COST_COLUMN = "hid_meal_cost"


def prefixed_columns(prefix: str, names: list[str]) -> list[str]:
    """Returns the column names obtained by prefixing every given name."""
    columns: list[str] = []
    for name in names:
        columns.append(f"{prefix}{name}")
    return columns


def destination_columns(aspect_names: list[str]) -> list[str]:
    """Returns the destination aspect-level column names."""
    return prefixed_columns(DESTINATION_PREFIX, aspect_names)


def level_columns(aspect_names: list[str]) -> list[str]:
    """Returns the hotel aspect-level column names."""
    return prefixed_columns(LEVEL_PREFIX, aspect_names)


def shift_columns(aspect_names: list[str]) -> list[str]:
    """Returns the room aspect-shift column names."""
    return prefixed_columns(SHIFT_PREFIX, aspect_names)


def wish_columns(aspect_names: list[str]) -> list[str]:
    """Returns the hidden traveler wish-weight column names."""
    return prefixed_columns(WISH_PREFIX, aspect_names)


def taste_columns(taste_names: list[str]) -> list[str]:
    """Returns the hidden traveler taste-weight column names."""
    return prefixed_columns(TASTE_PREFIX, taste_names)


def board_columns(board_names: list[str]) -> list[str]:
    """Returns the hotel board-availability column names."""
    return prefixed_columns(BOARD_PREFIX, board_names)
