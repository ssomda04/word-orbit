"""Shared pytest fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import answer_selector, game_repository
from app.core.config import get_settings
from app.main import create_app
from app.services.embedding import reset_embedding_service
from app.services.game import InMemoryGameRepository
from app.services.ranking import reset_rank_provider
from app.services.scoring import reset_artifact_store

# Pinned answer word: makes guessing deterministic and lets tests assert that
# this exact string never appears in a response or log while a game is playing.
TEST_ANSWER = "사과"

# Words that are never the answer, for scoring-only assertions.
WRONG_WORD = "학생"
OTHER_WRONG_WORD = "선생"


def _reset_process_wide_state() -> None:
    """Drop every cached, configuration-dependent singleton."""
    get_settings.cache_clear()
    reset_embedding_service()
    reset_rank_provider()
    reset_artifact_store()


@pytest.fixture(autouse=True)
def _isolated_providers() -> Iterator[None]:
    """Keep the process-wide services from leaking between tests.

    The settings, the embedding service, the rank provider and the artifact
    store are all cached for the lifetime of the process. A test that points
    `EMBEDDING_PROVIDER` at FastText, `VOCABULARY_PATH` at a temporary word
    list, or `SCORING_PROVIDER` at a temporary artifact root would otherwise
    hand its state to every test that ran afterwards.
    """
    _reset_process_wide_state()
    yield
    _reset_process_wide_state()


@pytest.fixture
def app() -> FastAPI:
    """App with a per-test game store and a fixed answer word.

    The real providers are process-cached (`@lru_cache`), so without these
    overrides game state would leak between tests.
    """
    application = create_app()
    repository = InMemoryGameRepository()
    application.dependency_overrides[game_repository] = lambda: repository
    application.dependency_overrides[answer_selector] = lambda: (lambda: TEST_ANSWER)
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """A TestClient bound to the isolated app (default: mock embeddings)."""
    return TestClient(app)


# --- Artifact mode -----------------------------------------------------------


def configure_environment(monkeypatch: pytest.MonkeyPatch, **env: str) -> None:
    """Set environment variables and drop every cached singleton that reads them.

    `Settings`, the embedding service, the rank provider and the artifact store
    are all process-cached, so changing the environment without this leaves the
    previous configuration in place.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _reset_process_wide_state()


def artifact_app(
    monkeypatch: pytest.MonkeyPatch, root: Path, **env: str
) -> tuple[FastAPI, InMemoryGameRepository]:
    """An app configured to score from `root`, plus its isolated game store.

    Deliberately does *not* override `answer_selector`: which answers a game can
    have is part of what artifact mode changes, so pinning one would test around
    the behaviour instead of testing it. The returned repository is how a test
    reads back an answer the API is not allowed to reveal.
    """
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root), **env
    )
    application = create_app()
    repository = InMemoryGameRepository()
    application.dependency_overrides[game_repository] = lambda: repository
    return application, repository
