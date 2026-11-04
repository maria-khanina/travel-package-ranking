"""Assembles the typed project configuration from the configs package."""

import copy
from pathlib import Path

from pydantic import BaseModel

from configs.amenities import AMENITIES
from configs.archetypes import ARCHETYPES
from configs.catalogue import CATALOGUE, POOL, SEASONS
from configs.choice import CHOICE
from configs.comparison import COMPARISON
from configs.destinations import DESTINATIONS
from configs.features import FEATURES
from configs.hotel_types import HOTEL_TYPES
from configs.matching import MATCHING
from configs.modeling import MODELING
from configs.origins import ORIGINS
from configs.phrases import PHRASES
from configs.practical import PRACTICAL
from configs.pricing import PRICING
from configs.prompts import PROMPTS
from configs.room_types import ROOMS
from configs.run import RUN, TESTING
from configs.screening import SCREENING
from configs.sessions import SESSIONS
from configs.text_generation import TEXT
from configs.text_styles import TEXT_STYLES
from src.settings.amenity_models import AmenitySettings
from src.settings.comparison_models import ComparisonSettings
from src.settings.feature_models import FeatureSettings
from src.settings.matching_models import MatchSettings
from src.settings.modeling_models import ModelingSettings
from src.settings.screening_models import ScreeningSettings
from src.settings.models import (
    CatalogueSettings,
    ChoiceSettings,
    PoolSettings,
    PricingSettings,
    RunSettings,
    SeasonSettings,
    SessionSettings,
)
from src.settings.records import (
    ArchetypeRecord,
    DestinationRecord,
    HotelTypeRecord,
    OriginRecord,
    PhraseSettings,
    PracticalSettings,
    PromptSettings,
    RoomSettings,
    TestingSettings,
    TextGenerationSettings,
    TextStyleSettings,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    run: RunSettings
    features: FeatureSettings
    amenities: AmenitySettings
    modeling: ModelingSettings
    matching: MatchSettings
    comparison: ComparisonSettings
    screening: ScreeningSettings
    sessions: SessionSettings
    catalogue: CatalogueSettings
    seasons: SeasonSettings
    pool: PoolSettings
    pricing: PricingSettings
    choice: ChoiceSettings
    rooms: RoomSettings
    text: TextGenerationSettings
    text_styles: TextStyleSettings
    testing: TestingSettings
    phrases: PhraseSettings
    practical: PracticalSettings
    prompts: PromptSettings
    destinations: list[DestinationRecord]
    origins: dict[str, OriginRecord]
    archetypes: dict[str, ArchetypeRecord]
    hotel_types: list[HotelTypeRecord]


def load_settings() -> Settings:
    """Builds an independent copy of the full project configuration."""
    return Settings(
        run=copy.deepcopy(RUN),
        features=copy.deepcopy(FEATURES),
        amenities=copy.deepcopy(AMENITIES),
        modeling=copy.deepcopy(MODELING),
        matching=copy.deepcopy(MATCHING),
        comparison=copy.deepcopy(COMPARISON),
        screening=copy.deepcopy(SCREENING),
        sessions=copy.deepcopy(SESSIONS),
        catalogue=copy.deepcopy(CATALOGUE),
        seasons=copy.deepcopy(SEASONS),
        pool=copy.deepcopy(POOL),
        pricing=copy.deepcopy(PRICING),
        choice=copy.deepcopy(CHOICE),
        rooms=copy.deepcopy(ROOMS),
        text=copy.deepcopy(TEXT),
        text_styles=copy.deepcopy(TEXT_STYLES),
        testing=copy.deepcopy(TESTING),
        phrases=copy.deepcopy(PHRASES),
        practical=copy.deepcopy(PRACTICAL),
        prompts=copy.deepcopy(PROMPTS),
        destinations=copy.deepcopy(DESTINATIONS),
        origins=copy.deepcopy(ORIGINS),
        archetypes=copy.deepcopy(ARCHETYPES),
        hotel_types=copy.deepcopy(HOTEL_TYPES),
    )
