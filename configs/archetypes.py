"""Traveler archetypes and their wishes."""

from src.settings.records import ArchetypeRecord


ARCHETYPES: dict[str, ArchetypeRecord] = {
    "family_beach": ArchetypeRecord(
        budget_shift=0.0,
        wishes={
            "beach_sea": 0.9,
            "family_friendly": 0.9,
            "quiet_relax": 0.2,
            "nightlife": -0.3,
            "active_adventure": 0.2,
            "wellness_spa": 0.1,
        },
        multipliers={"kids": 4.0, "no_kids": 0.25},
    ),
    "family_adventure": ArchetypeRecord(
        budget_shift=0.0,
        wishes={
            "family_friendly": 0.8,
            "active_adventure": 0.6,
            "mountains_ski": 0.4,
            "beach_sea": 0.2,
            "nightlife": -0.3,
        },
        multipliers={"kids": 3.0, "no_kids": 0.1},
    ),
    "romantic_getaway": ArchetypeRecord(
        budget_shift=0.15,
        wishes={
            "romantic": 0.9,
            "quiet_relax": 0.6,
            "beach_sea": 0.4,
            "luxury": 0.4,
            "nightlife": -0.2,
            "family_friendly": -0.3,
        },
        multipliers={"kids": 0.2, "no_kids": 0.8, "couple": 3.125},
    ),
    "party_break": ArchetypeRecord(
        budget_shift=0.0,
        wishes={
            "nightlife": 0.9,
            "beach_sea": 0.5,
            "city_culture": 0.3,
            "quiet_relax": -0.7,
            "family_friendly": -0.5,
            "wellness_spa": -0.2,
        },
        multipliers={"kids": 0.1},
    ),
    "wellness_retreat": ArchetypeRecord(
        budget_shift=0.15,
        wishes={
            "wellness_spa": 0.9,
            "quiet_relax": 0.8,
            "nightlife": -0.6,
            "beach_sea": 0.2,
            "luxury": 0.3,
            "family_friendly": -0.2,
        },
        multipliers={"kids": 0.3},
    ),
    "ski_trip": ArchetypeRecord(
        budget_shift=0.0,
        wishes={
            "mountains_ski": 0.95,
            "active_adventure": 0.7,
            "nightlife": 0.2,
            "family_friendly": 0.1,
        },
        multipliers={"winter": 3.0, "off_winter": 0.25, "kids": 0.5},
    ),
    "culture_city": ArchetypeRecord(
        budget_shift=0.0,
        wishes={
            "city_culture": 0.95,
            "nightlife": 0.2,
            "romantic": 0.2,
            "beach_sea": -0.2,
            "quiet_relax": -0.2,
        },
        multipliers={"kids": 0.5},
    ),
    "adventure_outdoors": ArchetypeRecord(
        budget_shift=0.0,
        wishes={
            "active_adventure": 0.9,
            "mountains_ski": 0.6,
            "beach_sea": 0.2,
            "luxury": -0.2,
            "nightlife": -0.1,
        },
        multipliers={"kids": 0.7},
    ),
    "luxury_escape": ArchetypeRecord(
        budget_shift=0.5,
        wishes={
            "luxury": 0.9,
            "romantic": 0.4,
            "beach_sea": 0.3,
            "wellness_spa": 0.4,
            "quiet_relax": 0.3,
            "family_friendly": -0.1,
        },
        multipliers={"kids": 0.5},
    ),
    "budget_sun": ArchetypeRecord(
        budget_shift=-0.45,
        wishes={
            "beach_sea": 0.8,
            "family_friendly": 0.2,
            "nightlife": 0.3,
            "luxury": -0.4,
        },
        multipliers={"summer": 1.5},
    ),
    "quiet_nature": ArchetypeRecord(
        budget_shift=0.0,
        wishes={
            "quiet_relax": 0.9,
            "mountains_ski": 0.6,
            "nightlife": -0.7,
            "beach_sea": 0.2,
            "active_adventure": 0.3,
        },
        multipliers={"kids": 0.6},
    ),
    "friends_getaway": ArchetypeRecord(
        budget_shift=0.0,
        wishes={
            "nightlife": 0.5,
            "beach_sea": 0.5,
            "active_adventure": 0.4,
            "city_culture": 0.4,
            "family_friendly": -0.4,
        },
        multipliers={"kids": 0.0, "no_kids": 0.8, "group": 1.875},
    ),
}
