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


@pytest.fixture(scope="module")
def service() -> FastTextEmbeddingService:
    """Load the real model once for the whole module."""
    loaded = FastTextEmbeddingService.load(Path(_MODEL_PATH))
    assert isinstance(loaded, FastTextEmbeddingService)
    return loaded


@pytest.fixture
def fasttext_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """An app wired to the real FastText provider, with an isolated game store."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fasttext")
    monkeypatch.setenv("FASTTEXT_MODEL_PATH", _MODEL_PATH)
    get_settings.cache_clear()
    reset_embedding_service()

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


def test_provider_is_built_once(fasttext_client: TestClient) -> None:
    assert get_embedding_service() is get_embedding_service()


# --- Through the API --------------------------------------------------------


def test_dev_similarity_endpoint_uses_the_real_provider(fasttext_client: TestClient) -> None:
    with fasttext_client as client:
        response = client.post(
            "/api/dev/similarity", json={"first": ANSWER, "second": RELATED}
        )

    body = response.json()
    assert response.status_code == 200
    assert body["provider"] == "fasttext"
    assert -1.0 <= body["similarity"] <= 1.0


def test_game_flow_scores_guesses_with_the_real_model(fasttext_client: TestClient) -> None:
    with fasttext_client as client:
        game_id = client.post("/api/games").json()["gameId"]

        related = client.post(f"/api/games/{game_id}/guesses", json={"word": RELATED})
        unrelated = client.post(f"/api/games/{game_id}/guesses", json={"word": UNRELATED})
        state = client.get(f"/api/games/{game_id}")

    assert related.status_code == 200
    assert unrelated.status_code == 200
    assert related.json()["similarity"] > unrelated.json()["similarity"]
    # Contract unchanged: still null in this phase.
    assert related.json()["rank"] is None
    assert related.json()["coordinate"] is None
    assert state.json()["guessCount"] == 2


def test_answer_is_not_exposed_while_playing(fasttext_client: TestClient) -> None:
    """The secrecy rule holds with a real model too (AGENTS.md)."""
    with fasttext_client as client:
        create = client.post("/api/games")
        game_id = create.json()["gameId"]
        guess = client.post(f"/api/games/{game_id}/guesses", json={"word": RELATED})
        state = client.get(f"/api/games/{game_id}")

    escaped = ANSWER.encode("unicode_escape").decode()
    for response in (create, guess, state):
        assert ANSWER not in response.text
        assert escaped not in response.text
    assert state.json()["answer"] is None
    assert state.json()["status"] == "playing"


def test_winning_guess_scores_one(fasttext_client: TestClient) -> None:
    with fasttext_client as client:
        game_id = client.post("/api/games").json()["gameId"]
        won = client.post(f"/api/games/{game_id}/guesses", json={"word": ANSWER})
        state = client.get(f"/api/games/{game_id}")

    assert won.json()["isAnswer"] is True
    assert won.json()["similarity"] == pytest.approx(1.0)
    assert state.json()["status"] == "won"
    assert state.json()["answer"] == ANSWER
