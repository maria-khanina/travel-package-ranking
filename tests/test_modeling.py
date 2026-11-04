"""Checks of the split, the ranking metrics, the bootstrap and pool features."""

import numpy as np
import pandas as pd

from src.dataset import DatasetBundle
from src.features.extras import fit_projection
from src.features.relative import build_relative_features
from src.modeling.bootstrap import bootstrap_table, paired_table, session_values
from src.modeling.evaluation import safe_auc
from src.modeling.experiments import ModelInputs, fold_masks
from src.modeling.ranking import (
    locate_booked,
    random_ranking_metrics,
    rank_of_booked,
    ranking_metrics,
)
from src.modeling.split import NO_FOLD, SplitMasks, build_masks
from src.settings.loader import Settings
from src.world.helpers import read_integers

TOLERANCE = 1e-9
POOLS_IN_SAMPLE = 4
BOOKED_POSITIONS = [3, 5, 7]
SESSION_RANKS = [1.0, 2.0, 30.0]
SMALL_SAMPLE = 200
LARGE_SAMPLE = 20000
PAIRED_SAMPLE = 500
RELATIVE_POOLS = 5
PROJECTION_ROWS = 500
PROJECTION_DIMENSION = 6
PROJECTION_SCALE = 5.0
DOMINANT_WEIGHT = 0.9
TOP_THREE = 3
COMPONENTS = 2
TOP_TEN = 10
LOW_SCORES = np.array([0.1, 0.2, 0.8, 0.9])
HIGH_SCORES = np.array([0.9, 0.8, 0.2, 0.1])
AUC_LABELS = np.array([False, False, True, True])


def build_test_masks(bundle: DatasetBundle, settings: Settings) -> SplitMasks:
    modeling = settings.modeling
    return build_masks(
        bundle.public,
        bundle.truth[modeling.register_column],
        modeling.split,
        modeling.label_column,
        modeling.hotel_column,
    )


def build_inputs(bundle: DatasetBundle, settings: Settings) -> ModelInputs:
    labels = bundle.public[settings.modeling.label_column].to_numpy(dtype=np.bool_)
    return ModelInputs(
        table=pd.DataFrame(),
        labels=labels,
        sessions=read_integers(bundle.public, "session_id"),
        hotels=bundle.public[settings.modeling.hotel_column],
        hidden_fit=np.zeros(len(labels)),
        masks=build_test_masks(bundle, settings),
    )


def test_development_and_test_sessions_do_not_overlap(
    bundle: DatasetBundle,
    settings: Settings,
) -> None:
    masks = build_test_masks(bundle, settings)
    sessions = read_integers(bundle.public, "session_id")
    assert not set(sessions[masks.development]) & set(sessions[masks.test])
    assert (masks.development | masks.test).all()


def test_every_development_row_has_a_fold(
    bundle: DatasetBundle,
    settings: Settings,
) -> None:
    masks = build_test_masks(bundle, settings)
    folds = masks.fold[masks.development]
    assert (folds != NO_FOLD).all()
    assert (masks.fold[masks.test] == NO_FOLD).all()
    assert set(folds) == set(range(settings.modeling.split.fold_count))


def test_sessions_stay_inside_one_fold(
    bundle: DatasetBundle,
    settings: Settings,
) -> None:
    masks = build_test_masks(bundle, settings)
    frame = pd.DataFrame(
        {
            "session": read_integers(bundle.public, "session_id"),
            "fold": masks.fold,
        }
    )
    assert (frame.groupby("session")["fold"].nunique() == 1).all()


def test_training_rows_never_contain_cold_hotels(
    bundle: DatasetBundle,
    settings: Settings,
) -> None:
    inputs = build_inputs(bundle, settings)
    train, stopping, evaluation = fold_masks(0, inputs, settings, "Logloss")
    assert not (train & inputs.masks.cold).any()
    assert not (stopping & inputs.masks.cold).any()
    assert not (train & evaluation).any()
    assert not (stopping & evaluation).any()


def test_perfect_scores_rank_first(settings: Settings) -> None:
    pool = settings.modeling.split.pool_size
    labels = np.zeros(pool * POOLS_IN_SAMPLE, dtype=np.bool_)
    positions = [pool * index + at for index, at in enumerate(BOOKED_POSITIONS)]
    labels[positions] = True
    sessions, found = locate_booked(labels, pool)
    ranks = rank_of_booked(labels.astype(np.float64), sessions, found, pool)
    metrics = ranking_metrics(ranks, settings.modeling.top_k_values)
    assert (ranks == 1.0).all()
    assert metrics["hit_rate_at_1"] == 1.0


def test_random_ranking_hits_at_chance(settings: Settings) -> None:
    pool = settings.modeling.split.pool_size
    metrics = random_ranking_metrics(pool, settings.modeling.top_k_values)
    assert abs(metrics["hit_rate_at_1"] - 1.0 / pool) < TOLERANCE
    assert abs(metrics["hit_rate_at_10"] - TOP_TEN / pool) < TOLERANCE


def test_auc_of_perfect_and_reversed_scores() -> None:
    assert safe_auc(AUC_LABELS, LOW_SCORES) == 1.0
    assert safe_auc(AUC_LABELS, HIGH_SCORES) == 0.0


def test_session_values_cover_every_metric(settings: Settings) -> None:
    pool = settings.modeling.split.pool_size
    values = session_values(
        np.array(SESSION_RANKS), pool, settings.modeling.top_k_values
    )
    assert values["hit_rate_at_1"].tolist() == [1.0, 0.0, 0.0]
    assert values["session_auc"][0] == 1.0
    assert values["session_auc"][-1] == 0.0


def test_interval_contains_the_estimate(settings: Settings) -> None:
    pool = settings.modeling.split.pool_size
    ranks = np.random.default_rng(0).integers(1, pool + 1, SMALL_SAMPLE)
    bootstrap = settings.modeling.bootstrap
    samples = {"model": ranks.astype(np.float64)}
    table = bootstrap_table(samples, pool, [TOP_THREE], bootstrap)
    row = table[table["metric"] == "hit_rate_at_3"].iloc[0]
    assert float(row["lower"]) <= float(row["estimate"]) <= float(row["upper"])


def test_interval_shrinks_with_more_sessions(settings: Settings) -> None:
    pool = settings.modeling.split.pool_size
    generator = np.random.default_rng(0)
    samples = {
        "small": generator.integers(1, pool + 1, SMALL_SAMPLE).astype(np.float64),
        "large": generator.integers(1, pool + 1, LARGE_SAMPLE).astype(np.float64),
    }
    table = bootstrap_table(samples, pool, [TOP_THREE], settings.modeling.bootstrap)
    hits = table[table["metric"] == "hit_rate_at_3"].set_index("model")
    widths = hits["upper"] - hits["lower"]
    assert float(widths["large"]) < float(widths["small"])


def test_identical_models_have_zero_paired_difference(settings: Settings) -> None:
    pool = settings.modeling.split.pool_size
    bootstrap = settings.modeling.bootstrap.model_copy(update={"baseline": "a"})
    generator = np.random.default_rng(1)
    ranks = generator.integers(1, pool + 1, PAIRED_SAMPLE).astype(np.float64)
    paired = paired_table({"a": ranks, "b": ranks.copy()}, pool, [TOP_THREE], bootstrap)
    assert (paired["difference"] == 0.0).all()
    assert (paired["lower"] == 0.0).all()
    assert (paired["upper"] == 0.0).all()


def test_pool_ranks_are_shares_of_the_pool(settings: Settings) -> None:
    pool = settings.modeling.split.pool_size
    values = np.random.default_rng(2).normal(size=pool * RELATIVE_POOLS)
    relative = build_relative_features(pd.DataFrame({"price": values}), ["price"], pool)
    ranks = relative["price_pool_rank"].to_numpy().reshape(-1, pool)
    assert ranks.max() == 1.0
    assert np.isclose(ranks.min(), 1.0 / pool)
    assert np.allclose(ranks.sum(axis=1), (pool + 1) / 2.0)


def test_projection_directions_are_ordered_by_variance() -> None:
    generator = np.random.default_rng(3)
    shape = (PROJECTION_ROWS, PROJECTION_DIMENSION)
    vectors = generator.normal(size=shape).astype(np.float32)
    vectors[:, 0] *= PROJECTION_SCALE
    mean, directions = fit_projection(vectors, COMPONENTS, PROJECTION_ROWS)
    assert directions.shape == (COMPONENTS, PROJECTION_DIMENSION)
    assert abs(directions[0, 0]) > DOMINANT_WEIGHT
    assert mean.shape == (PROJECTION_DIMENSION,)
