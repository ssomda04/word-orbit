"""Shared pytest fixtures."""

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import answer_selector, game_repository
from app.core.config import get_settings
from app.main import create_app
from app.services.embedding import reset_embedding_service
from app.services.game import InMemoryGameRepository
from app.services.ranking import reset_rank_provider

# Pinned answer word: makes guessing deterministic and lets tests assert that
# this exact string never appears in a response or log while a game is playing.
TEST_ANSWER = "사과"

# Words that are never the answer, for scoring-only assertions.
WRONG_WORD = "학생"
OTHER_WRONG_WORD = "선생"


@pytest.fixture(autouse=True)
def _isolated_embedding_provider() -> Iterator[None]:
    """Keep the process-wide services from leaking between tests.

    The settings, the embedding service and the rank provider are all cached for
    the lifetime of the process. A test that points `EMBEDDING_PROVIDER` at
    FastText, or `VOCABULARY_PATH` at a temporary word list, would otherwise hand
    its service to every test that ran afterwards.
    """
    get_settings.cache_clear()
    reset_embedding_service()
    reset_rank_provider()
    yield
    get_settings.cache_clear()
    reset_embedding_service()
    reset_rank_provider()


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
