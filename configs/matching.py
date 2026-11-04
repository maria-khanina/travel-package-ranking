"""Settings of the open match and stated match scores."""

from src.settings.matching_models import MatchSettings

MATCHING = MatchSettings(
    strong_visibility=1.0,
    medium_visibility=0.5,
    chunk_rows=500000,
    reference_sessions=20000,
    reference_seed=3,
)
