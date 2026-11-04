"""Shared fixtures that build one small simulated world for all tests."""

from pathlib import Path

import pytest

from src.dataset import DatasetBundle, build_dataset, build_placeholder_texts
from src.settings.loader import Settings, load_settings
from src.texts.jobs import build_text_jobs
from src.texts.models import TextJobs
from src.world.builder import build_world
from src.world.models import WorldTables


@pytest.fixture(scope="module")
def settings() -> Settings:
    """Settings with the small session count used by the tests."""
    loaded = load_settings()
    loaded.run.session_count = loaded.testing.session_count
    return loaded


@pytest.fixture(scope="module")
def world(settings: Settings) -> WorldTables:
    """A small simulated world."""
    return build_world(settings, settings.testing.seed)


@pytest.fixture(scope="module")
def jobs(settings: Settings, world: WorldTables) -> TextJobs:
    """The text briefs of the small world."""
    return build_text_jobs(settings.testing.seed, world, settings)


@pytest.fixture(scope="module")
def bundle(settings: Settings, world: WorldTables, jobs: TextJobs) -> DatasetBundle:
    """The dataset simulated with placeholder texts."""
    texts = build_placeholder_texts(jobs)
    return build_dataset(world, texts, settings, None)


@pytest.fixture
def storage_directory(tmp_path: Path) -> Path:
    """An empty directory for tests that write files."""
    return tmp_path
