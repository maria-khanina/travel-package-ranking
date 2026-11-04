"""Checks of the simulated traveler behavior and of the text rules."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.behavior.utility import compute_utilities
from src.settings.loader import Settings
from src.texts.diversity import DiversityFilter, split_words
from src.texts.models import TextJobs, TextRecord
from src.texts.runner import collect_place_words
from src.texts.selection import select_practical_sessions
from src.texts.store import append_record, compact_records, read_records
from src.texts.validation import TextValidator
from src.world.helpers import read_floats
from src.world.models import FloatArray, WorldTables

EXTRA_TEXTS = 100


def build_probe_rows(world: WorldTables, count: int) -> pd.DataFrame:
    """Takes the first rows of the world as a probe sample."""
    return world.rows.head(count).copy()


def compute_probe_ratio(rows: pd.DataFrame, margin: float) -> FloatArray:
    """Computes the price ratio of the probe rows at one margin."""
    selling = read_floats(rows, "net_cost") * (1 + margin)
    return selling / read_floats(rows, "competitor_retail_price")


def measure_utility_gap(world: WorldTables, settings: Settings) -> FloatArray:
    """Computes the utility drop when moving from a low to a high margin."""
    testing = settings.testing
    rows = build_probe_rows(world, testing.probe_row_count)
    low = np.full(len(rows), testing.margin_low)
    high = np.full(len(rows), testing.margin_high)
    ratio = compute_probe_ratio(rows, testing.margin_low)
    kept = ratio > settings.choice.trust_threshold
    utility_low = compute_utilities(rows, settings, low, False)
    utility_high = compute_utilities(rows, settings, high, False)
    return (utility_low - utility_high)[kept]


def test_higher_margin_never_raises_utility_above_trust_zone(
    world: WorldTables,
    settings: Settings,
) -> None:
    assert (measure_utility_gap(world, settings) >= 0).all()


def utility_at_ratio(
    rows: pd.DataFrame,
    settings: Settings,
    ratio: float,
) -> FloatArray:
    """Computes utilities when the competitor price implies a given ratio."""
    margin = settings.testing.margin_low
    selling = read_floats(rows, "net_cost") * (1 + margin)
    rows["competitor_retail_price"] = selling / ratio
    return compute_utilities(rows, settings, np.full(len(rows), margin), False)


def share_where_cheap_wins(
    world: WorldTables,
    settings: Settings,
    trust_coefficient: float,
) -> float:
    """Shares the rows where a suspiciously cheap price beats a fair price."""
    testing = settings.testing
    adjusted = settings.model_copy(deep=True)
    adjusted.choice.trust_coefficient = trust_coefficient
    rows = build_probe_rows(world, testing.cheap_row_count)
    cheap = utility_at_ratio(rows, adjusted, testing.cheap_ratio)
    fair = utility_at_ratio(rows, adjusted, testing.fair_ratio)
    return float((cheap > fair).mean())


def test_without_trust_penalty_cheap_wins(
    world: WorldTables,
    settings: Settings,
) -> None:
    share = share_where_cheap_wins(world, settings, 0.0)
    assert share > settings.testing.cheap_win_without_penalty


def test_trust_penalty_hurts_suspiciously_cheap_offers(
    world: WorldTables,
    settings: Settings,
) -> None:
    share = share_where_cheap_wins(world, settings, settings.choice.trust_coefficient)
    assert share < settings.testing.cheap_win_with_penalty


def test_text_styles_vary_between_briefs(jobs: TextJobs) -> None:
    for briefs in (jobs.intents, jobs.hotels, jobs.rooms):
        labels: list[dict[str, str]] = []
        for spec in briefs["spec"]:
            labels.append(json.loads(spec)["labels"])
        for dimension in labels[0]:
            values = {label.get(dimension) for label in labels}
            assert len(values) > 1


def test_validator_accepts_vibe_and_rejects_hard_filters(
    settings: Settings,
) -> None:
    validator = TextValidator(settings.text, collect_place_words(settings))
    limit = settings.text.intent_words.maximum
    for accepted in settings.testing.accepted_intents:
        assert validator.validate_intent(accepted, limit)
    for rejected in settings.testing.rejected_intents:
        assert not validator.validate_intent(rejected, limit)


def test_diversity_filter_blocks_overused_phrases(settings: Settings) -> None:
    diversity = DiversityFilter(settings.text, [])
    repeated = "a calm beach where the air is warm"
    fresh = "snowy peaks and a cozy fireplace"
    registrations = settings.text.repeat_warmup_count
    registrations += settings.text.repeat_minimum_count
    for _ in range(registrations):
        diversity.register(repeated)
    assert not diversity.allows(repeated)
    assert diversity.allows(fresh)


def contains_good(record: TextRecord) -> bool:
    """Accepts records whose text mentions the word good."""
    return record.text is not None and "good" in record.text


def test_compaction_keeps_one_acceptable_record_per_item(
    storage_directory: Path,
) -> None:
    path = storage_directory / "texts.jsonl"
    records = [
        TextRecord(id=2, text=None, model="m", attempts=1),
        TextRecord(id=1, text="good", model="m", attempts=1),
        TextRecord(id=2, text="good again", model="m", attempts=2),
        TextRecord(id=1, text="bad", model="m", attempts=2),
        TextRecord(id=0, text="bad", model="m", attempts=1),
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            append_record(handle, record)
    compact_records(path, contains_good)
    kept = read_records(path)
    assert [record.id for record in kept] == [0, 1, 2]
    assert [record.text for record in kept] == [None, "good", "good again"]


def test_practical_validator_rejects_poetic_wording(settings: Settings) -> None:
    validator = TextValidator(settings.text, collect_place_words(settings))
    limit = settings.text.intent_words.maximum
    for accepted in settings.testing.accepted_practical:
        assert validator.validate_practical(accepted, limit)
    for rejected in settings.testing.rejected_practical:
        assert not validator.validate_practical(rejected, limit)


def test_practical_share_selection_is_stable_and_nested(settings: Settings) -> None:
    count = settings.testing.session_count
    first = select_practical_sessions(count, settings)
    again = select_practical_sessions(count, settings)
    wider = settings.model_copy(deep=True)
    wider.run.practical_intent_share = 1.0
    everyone = select_practical_sessions(count, wider)
    assert (first == again).all()
    assert everyone.all()
    assert (first <= everyone).all()


def test_diversity_filter_ignores_required_phrases(settings: Settings) -> None:
    exempt = [split_words("kids club")]
    diversity = DiversityFilter(settings.text, exempt)
    text = "enjoy the kids club every morning with family"
    for _ in range(settings.text.repeat_warmup_count + EXTRA_TEXTS):
        diversity.register(text)
    assert diversity.allows("a kids club for all")
    assert not diversity.allows(text)
