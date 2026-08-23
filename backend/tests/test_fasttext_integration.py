"""Optional tests against a real FastText model.

Skipped unless BOTH are true:
  1. the `fasttext` extra is installed  (`uv sync --extra fasttext`)
  2. FASTTEXT_MODEL_PATH points at an existing file

So `pytest` stays green with no model anywhere, and CI — which installs no
extras and sets no path — never touches this file.

Run it explicitly:

    uv sync --extra fasttext
    $env:FASTTEXT_MODEL_PATH = "C:/models/cc.ko.300.bin"   # PowerShell
    uv run --extra fasttext pytest -m fasttext -v

These are smoke tests, not an evaluation. Quality metrics live in `ml/`
(docs/MODEL_EVALUATION.md); assertions here stay loose on purpose — notably,
FastText separates `veryClose` from `related` at only ~51% accuracy, so no test
may depend on that ordering.
"""

import importlib.util
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import answer_selector, game_repository
from app.core.config import get_settings
from app.main import create_app
from app.services.embedding import (
    FastTextEmbeddingService,
    get_embedding_service,
    reset_embedding_service,
)
from app.services.game import InMemoryGameRepository

_MODEL_PATH = os.environ.get("FASTTEXT_MODEL_PATH", "").strip()

pytestmark = [
    pytest.mark.fasttext,
    pytest.mark.skipif(
        importlib.util.find_spec("fasttext") is None,
        reason="the `fasttext` extra is not installed (uv sync --extra fasttext)",
    ),
    pytest.mark.skipif(
        not _MODEL_PATH or not Path(_MODEL_PATH).is_file(),
        reason="FASTTEXT_MODEL_PATH is unset or does not point at a file",
    ),
]

# Pinned answer, so the reveal rule can be asserted without winning by accident.
ANSWER = "바다"
RELATED = "해양"
UNRELATED = "컴퓨터"

EXPECTED_DIMENSION = 300


@pytest.fixture(autouse=True)
def _isolated_embedding_provider() -> Iterator[None]:
    """Shadow the conftest isolation fixture for this module only.

    conftest resets the process-wide service around *every* test. That is right
    for the mock, but here it would reload several gigabytes of weights per test:
    a real run took 132 s for 11 tests, and a module-scoped model plus a
    per-test one meant two copies resident at once. This module loads the model
    exactly once instead — see `_fasttext_environment`.
    """
    yield


@pytest.fixture(scope="module", autouse=True)
def _fasttext_environment() -> Iterator[None]:
    """Point the whole module at the real provider, and clean up afterwards."""
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setenv("EMBEDDING_PROVIDER", "fasttext")
        patcher.setenv("FASTTEXT_MODEL_PATH", _MODEL_PATH)
        get_settings.cache_clear()
        reset_embedding_service()
        yield
    # Drop the multi-gigabyte model as soon as this module is done.
    get_settings.cache_clear()
    reset_embedding_service()


@pytest.fixture(scope="module")
def service() -> FastTextEmbeddingService:
    """The one real model for this module — the single expensive load."""
    loaded = get_embedding_service()
    assert isinstance(loaded, FastTextEmbeddingService)
    return loaded


@pytest.fixture
def fasttext_client(service: FastTextEmbeddingService) -> TestClient:
    """An app on the real provider, with a per-test game store.

    Deliberately *not* used as a context manager by the tests below: entering it
    would run the lifespan, and the point here is to reuse the already-loaded
    model. Startup warm-up has its own test.
    """
    application = create_app()
    repository = InMemoryGameRepository()
    application.dependency_overrides[game_repository] = lambda: repository
    application.dependency_overrides[answer_selector] = lambda: (lambda: ANSWER)
    return TestClient(application)


# --- The model itself -------------------------------------------------------


def test_vector_has_the_documented_dimension(service: FastTextEmbeddingService) -> None:
    """300 per docs/MODEL_EVALUATION.md."""
    assert len(service.encode("학생")) == EXPECTED_DIMENSION


def test_encode_many_returns_uniform_vectors(service: FastTextEmbeddingService) -> None:
    vectors = service.encode_many(["학생", "선생", "학교"])

    assert len(vectors) == 3
    assert {len(vector) for vector in vectors} == {EXPECTED_DIMENSION}


def test_self_similarity_is_one(service: FastTextEmbeddingService) -> None:
    assert service.similarity(ANSWER, ANSWER) == pytest.approx(1.0)


def test_related_scores_above_unrelated(service: FastTextEmbeddingService) -> None:
    """The one quality claim the baseline supports: related > unrelated (~98%)."""
    assert service.similarity(ANSWER, RELATED) > service.similarity(ANSWER, UNRELATED)


def test_similarity_stays_in_range(service: FastTextEmbeddingService) -> None:
    for candidate in (RELATED, UNRELATED, "행복", "자동차"):
        assert -1.0 <= service.similarity(ANSWER, candidate) <= 1.0


def test_subword_oov_word_still_gets_a_vector(service: FastTextEmbeddingService) -> None:
    """No vocabulary gate: FastText composes character n-grams."""
    vector = service.encode("바다바다바다스러운")

    assert len(vector) == EXPECTED_DIMENSION


def test_provider_is_built_once(service: FastTextEmbeddingService) -> None:
    assert get_embedding_service() is service
    assert get_embedding_service() is get_embedding_service()


def test_startup_reuses_the_already_loaded_model(service: FastTextEmbeddingService) -> None:
    """Running the lifespan must not load a second copy of the weights."""
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert get_embedding_service() is service


# --- Through the API --------------------------------------------------------


def test_dev_similarity_endpoint_uses_the_real_provider(fasttext_client: TestClient) -> None:
    response = fasttext_client.post(
        "/api/dev/similarity", json={"first": ANSWER, "second": RELATED}
    )

    body = response.json()
    assert response.status_code == 200
    assert body["provider"] == "fasttext"
    assert -1.0 <= body["similarity"] <= 1.0


def test_game_flow_scores_guesses_with_the_real_model(fasttext_client: TestClient) -> None:
    game_id = fasttext_client.post("/api/games").json()["gameId"]

    related = fasttext_client.post(f"/api/games/{game_id}/guesses", json={"word": RELATED})
    unrelated = fasttext_client.post(f"/api/games/{game_id}/guesses", json={"word": UNRELATED})
    state = fasttext_client.get(f"/api/games/{game_id}")

    assert related.status_code == 200
    assert unrelated.status_code == 200
    assert related.json()["similarity"] > unrelated.json()["similarity"]
    # Contract unchanged: still null in this phase.
    assert related.json()["rank"] is None
    assert related.json()["coordinate"] is None
    assert state.json()["guessCount"] == 2


def test_answer_is_not_exposed_while_playing(fasttext_client: TestClient) -> None:
    """The secrecy rule holds with a real model too (AGENTS.md)."""
    create = fasttext_client.post("/api/games")
    game_id = create.json()["gameId"]
    guess = fasttext_client.post(f"/api/games/{game_id}/guesses", json={"word": RELATED})
    state = fasttext_client.get(f"/api/games/{game_id}")

    escaped = ANSWER.encode("unicode_escape").decode()
    for response in (create, guess, state):
        assert ANSWER not in response.text
        assert escaped not in response.text
    assert state.json()["answer"] is None
    assert state.json()["status"] == "playing"


def test_winning_guess_scores_one(fasttext_client: TestClient) -> None:
    game_id = fasttext_client.post("/api/games").json()["gameId"]
    won = fasttext_client.post(f"/api/games/{game_id}/guesses", json={"word": ANSWER})
    state = fasttext_client.get(f"/api/games/{game_id}")

    assert won.json()["isAnswer"] is True
    assert won.json()["similarity"] == pytest.approx(1.0)
    assert state.json()["status"] == "won"
    assert state.json()["answer"] == ANSWER
