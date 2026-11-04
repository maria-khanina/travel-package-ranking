"""Generates the synthetic hotel catalogue."""

import numpy as np
import pandas as pd

from src.settings.loader import Settings
from src.settings.models import CatalogueSettings
from src.settings.records import HotelTypeRecord
from src.world.columns import BOARD_PREFIX, LEVEL_PREFIX
from src.world.destinations import read_destination_levels
from src.world.helpers import build_aspect_vector, normalize, sample_index
from src.world.models import FloatArray

HotelRow = dict[str, float | str | bool]


def read_key_level(
    hotel_type: HotelTypeRecord,
    levels: FloatArray,
    aspect_names: list[str],
    default_level: float,
) -> float:
    """Reads the destination level that drives a hotel type's popularity."""
    if hotel_type.key_aspect is None:
        return default_level
    return float(levels[aspect_names.index(hotel_type.key_aspect)])


def is_type_allowed(
    hotel_type: HotelTypeRecord,
    levels: FloatArray,
    is_ski: bool,
    aspect_names: list[str],
) -> bool:
    """Checks whether a hotel type can exist in a destination."""
    if hotel_type.requires_ski:
        return is_ski
    if hotel_type.requires_aspect is None:
        return True
    level = levels[aspect_names.index(hotel_type.requires_aspect)]
    return bool(level >= hotel_type.requires_level)


def compute_type_weight(
    hotel_type: HotelTypeRecord,
    levels: FloatArray,
    is_ski: bool,
    catalogue: CatalogueSettings,
) -> float:
    """Computes how likely one hotel type is in a destination."""
    names = catalogue.aspects.names
    if not is_type_allowed(hotel_type, levels, is_ski, names):
        return 0.0
    level = read_key_level(hotel_type, levels, names, catalogue.default_type_level)
    return hotel_type.base_weight * level**catalogue.type_level_exponent


def compute_type_probabilities(
    levels: FloatArray,
    is_ski: bool,
    settings: Settings,
) -> FloatArray:
    """Gives the probability of every hotel type in a destination."""
    weights: list[float] = []
    for hotel_type in settings.hotel_types:
        weight = compute_type_weight(hotel_type, levels, is_ski, settings.catalogue)
        weights.append(weight)
    return normalize(np.array(weights))


def sample_stars(
    random_generator: np.random.Generator,
    hotel_type: HotelTypeRecord,
    catalogue: CatalogueSettings,
) -> int:
    """Draws a star rating allowed for the hotel type."""
    stars = np.arange(1, len(catalogue.star_probabilities) + 1)
    allowed = np.isin(stars, hotel_type.stars)
    probabilities = np.array(catalogue.star_probabilities) * allowed
    return int(random_generator.choice(stars, p=normalize(probabilities)))


def compute_luxury_level(
    random_generator: np.random.Generator,
    destination_level: float,
    type_shift: float,
    stars: int,
    catalogue: CatalogueSettings,
) -> float:
    """Computes the luxury level, which grows with the star rating."""
    base = catalogue.luxury_destination_weight * destination_level
    base += catalogue.luxury_star_weight * (stars - 1) + type_shift
    noise = random_generator.normal(0.0, catalogue.luxury_noise_standard_deviation)
    return base + float(noise)


def compute_hotel_levels(
    random_generator: np.random.Generator,
    destination_levels: FloatArray,
    shift: FloatArray,
    stars: int,
    catalogue: CatalogueSettings,
) -> FloatArray:
    """Derives the aspect levels of a hotel from its destination and type."""
    names = catalogue.aspects.names
    spread = catalogue.level_noise_standard_deviation
    noise = random_generator.normal(0.0, spread, len(names))
    levels = destination_levels + shift + noise
    luxury = names.index(catalogue.aspects.luxury)
    levels[luxury] = compute_luxury_level(
        random_generator,
        destination_levels[luxury],
        shift[luxury],
        stars,
        catalogue,
    )
    return np.clip(levels, 0.0, 1.0)


def compute_rating(
    random_generator: np.random.Generator,
    stars: int,
    hotel_effect: float,
    catalogue: CatalogueSettings,
) -> float:
    """Computes the guest rating from stars and the hidden hotel effect."""
    spread = catalogue.rating_noise_standard_deviation
    noise = random_generator.normal(0.0, spread)
    raw = catalogue.rating_base + catalogue.rating_star_step * (stars - 1)
    raw += catalogue.rating_effect_weight * hotel_effect + noise
    clipped = float(np.clip(raw, catalogue.rating_minimum, catalogue.rating_maximum))
    return round(clipped, catalogue.rating_decimals)


def sample_boards(
    random_generator: np.random.Generator,
    hotel_type: HotelTypeRecord,
    keep_probability: float,
) -> list[str]:
    """Chooses which board options a hotel supports."""
    kept: list[str] = []
    for board in hotel_type.boards:
        if random_generator.random() < keep_probability:
            kept.append(board)
    if not kept:
        kept.append(str(random_generator.choice(hotel_type.boards)))
    return kept


def compose_name(
    random_generator: np.random.Generator,
    hotel_type: HotelTypeRecord,
    catalogue: CatalogueSettings,
) -> str:
    """Composes a hotel name from a prefix, a noun and a type suffix."""
    prefix = random_generator.choice(catalogue.name_prefixes)
    noun = random_generator.choice(catalogue.name_nouns)
    core = random_generator.choice(hotel_type.name_cores)
    return f"{prefix} {noun} {core}"


def generate_hotel_name(
    random_generator: np.random.Generator,
    hotel_type: HotelTypeRecord,
    used_names: set[str],
    catalogue: CatalogueSettings,
) -> str:
    """Generates a hotel name that is unique across the catalogue."""
    name = compose_name(random_generator, hotel_type, catalogue)
    for _ in range(catalogue.name_maximum_tries):
        if name not in used_names:
            break
        name = compose_name(random_generator, hotel_type, catalogue)
    if name in used_names:
        name = f"{name} {len(used_names)}"
    used_names.add(name)
    return name


def add_level_and_board_columns(
    row: HotelRow,
    levels: FloatArray,
    boards: list[str],
    settings: Settings,
) -> None:
    """Adds the aspect-level and board-availability columns to a hotel row."""
    for name, level in zip(settings.catalogue.aspects.names, levels):
        row[f"{LEVEL_PREFIX}{name}"] = float(level)
    for board in settings.pricing.board_names:
        row[f"{BOARD_PREFIX}{board}"] = board in boards


def draw_hotel_traits(
    random_generator: np.random.Generator,
    destination: pd.Series,
    hotel_type: HotelTypeRecord,
    used_names: set[str],
    catalogue: CatalogueSettings,
) -> HotelRow:
    """Draws the scalar traits of a hotel, including its hidden effect."""
    stars = sample_stars(random_generator, hotel_type, catalogue)
    spread = catalogue.hotel_effect_standard_deviation
    effect = float(random_generator.normal(0.0, spread))
    noise = random_generator.normal(0.0, catalogue.price_noise_standard_deviation)
    row: HotelRow = {
        "destination_index": int(destination["destination_index"]),
        "city": str(destination["city"]),
        "country": str(destination["country"]),
        "hotel_type": hotel_type.name,
        "hotel_name": generate_hotel_name(
            random_generator,
            hotel_type,
            used_names,
            catalogue,
        ),
        "stars": stars,
        "rating": compute_rating(random_generator, stars, effect, catalogue),
        "hotel_effect": effect,
        "price_noise": float(np.exp(noise)),
    }
    return row


def build_hotel_row(
    random_generator: np.random.Generator,
    destination: pd.Series,
    hotel_type: HotelTypeRecord,
    used_names: set[str],
    settings: Settings,
) -> HotelRow:
    """Draws one hotel with its traits, aspect levels and boards."""
    catalogue = settings.catalogue
    names = catalogue.aspects.names
    row = draw_hotel_traits(
        random_generator,
        destination,
        hotel_type,
        used_names,
        catalogue,
    )
    levels = compute_hotel_levels(
        random_generator,
        read_destination_levels(destination, names),
        build_aspect_vector(hotel_type.shift, names),
        int(row["stars"]),
        catalogue,
    )
    keep = catalogue.board_keep_probability
    boards = sample_boards(random_generator, hotel_type, keep)
    add_level_and_board_columns(row, levels, boards, settings)
    return row


def count_destination_hotels(
    destination: pd.Series,
    catalogue: CatalogueSettings,
) -> int:
    """Decides how many hotels a destination has from its market size."""
    competition = float(destination["competition"])
    size = catalogue.size_base + catalogue.size_competition_weight * competition
    return int(round(catalogue.hotels_base + catalogue.hotels_scale * size))


def build_destination_hotels(
    random_generator: np.random.Generator,
    destination: pd.Series,
    used_names: set[str],
    settings: Settings,
) -> list[HotelRow]:
    """Generates all hotels of one destination."""
    names = settings.catalogue.aspects.names
    levels = read_destination_levels(destination, names)
    is_ski = bool(destination["ski"])
    probabilities = compute_type_probabilities(levels, is_ski, settings)
    rows: list[HotelRow] = []
    for _ in range(count_destination_hotels(destination, settings.catalogue)):
        hotel_type = settings.hotel_types[sample_index(random_generator, probabilities)]
        rows.append(
            build_hotel_row(
                random_generator,
                destination,
                hotel_type,
                used_names,
                settings,
            )
        )
    return rows


def build_hotel_table(
    random_generator: np.random.Generator,
    destinations: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Generates the hotels of every destination as one table."""
    used_names: set[str] = set()
    rows: list[HotelRow] = []
    for _, destination in destinations.iterrows():
        rows.extend(
            build_destination_hotels(
                random_generator,
                destination,
                used_names,
                settings,
            )
        )
    hotels = pd.DataFrame(rows)
    hotels.insert(0, "hotel_id", np.arange(len(hotels)))
    return hotels
