"""The `GuessScorer` seam, and proof it changed nothing.

This module exists because the seam was introduced by a refactor: every scored
value must still be produced by the same call, with the same arguments, in the
same order, as before `GuessScorer` existed. So the tests below are mostly about
what `EmbeddingGuessScorer` does *not* do — it applies no policy of its own, it
rewrites no error, and it rounds nothing.

The last section is the one worth keeping an eye on. `GameService` now depends on
`guess_scorer` rather than on `embedding_service` and `rank_provider` directly,
and `guess_scorer` is composed from those two dependencies precisely so that an
override of either still reaches a guess. If that ever stops being true,
`tests/test_ranking.py::test_a_replayed_guess_keeps_its_original_rank` would go
green while testing nothing.
"""

from collections.abc import Sequence
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import embedding_service, rank_provider
from app.services.embedding import DeterministicEmbeddingService
from app.services.ranking import NullRankProvider
from app.services.scoring import EmbeddingGuessScorer, GuessScore, GuessScorer
from tests.conftest import TEST_ANSWER

WORD = "학생"


class RecordingEmbeddingService:
    """Records the exact arguments `similarity` was called with."""

    def __init__(self, score: float = 0.25) -> None:
        self.score = score
        self.calls: list[tuple[str, str]] = []

    def encode(self, text: str) -> list[float]:
        raise NotImplementedError

    def encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError

    def similarity(self, first: str, second: str) -> float:
        self.calls.append((first, second))
        return self.score

    def project_3d(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class RecordingRankProvider:
    """Records the exact arguments `rank_of` was called with."""

    def __init__(self, rank: int | None = 42) -> None:
        self.rank = rank
        self.calls: list[tuple[str, str]] = []

    def rank_of(self, answer: str, word: str) -> int | None:
        self.calls.append((answer, word))
        return self.rank


# --- GuessScore --------------------------------------------------------------


def test_guess_score_is_immutable() -> None:
    """A recorded score never changes, like the `Guess` it ends up on."""
    score = GuessScore(similarity=0.5, rank=3)

    with pytest.raises(AttributeError):
        score.similarity = 0.9  # type: ignore[misc]


def test_rank_defaults_to_none() -> None:
    """Null rank is the contract's default, not a special case."""
    assert GuessScore(similarity=0.5).rank is None


# --- The adapter applies no policy of its own --------------------------------


def test_satisfies_the_guess_scorer_protocol() -> None:
    scorer = EmbeddingGuessScorer(RecordingEmbeddingService(), NullRankProvider())

    assert isinstance(scorer, GuessScorer)


def test_similarity_is_asked_for_as_guess_then_answer() -> None:
    """`similarity(word, answer)` — the order `GameService` always used."""
    embedder = RecordingEmbeddingService()

    EmbeddingGuessScorer(embedder, NullRankProvider()).score(TEST_ANSWER, WORD)

    assert embedder.calls == [(WORD, TEST_ANSWER)]


def test_rank_is_asked_for_as_answer_then_guess() -> None:
    """`rank_of(answer, word)` — the opposite order, also unchanged."""
    ranker = RecordingRankProvider()

    EmbeddingGuessScorer(RecordingEmbeddingService(), ranker).score(TEST_ANSWER, WORD)

    assert ranker.calls == [(TEST_ANSWER, WORD)]


def test_both_values_are_returned_untouched() -> None:
    scorer = EmbeddingGuessScorer(RecordingEmbeddingService(-0.0657), RecordingRankProvider(7))

    score = scorer.score(TEST_ANSWER, WORD)

    assert score == GuessScore(similarity=-0.0657, rank=7)


def test_a_missing_rank_passes_through_as_none() -> None:
    scorer = EmbeddingGuessScorer(RecordingEmbeddingService(), NullRankProvider())

    assert scorer.score(TEST_ANSWER, WORD).rank is None


def test_each_seam_is_consulted_exactly_once() -> None:
    embedder = RecordingEmbeddingService()
    ranker = RecordingRankProvider()

    EmbeddingGuessScorer(embedder, ranker).score(TEST_ANSWER, WORD)

    assert len(embedder.calls) == 1
    assert len(ranker.calls) == 1


# --- Failures propagate as they did ------------------------------------------


def test_a_model_failure_propagates_and_skips_the_rank_lookup() -> None:
    """Order is load-bearing: the model error is the one a caller sees."""

    class ExplodingEmbeddingService(RecordingEmbeddingService):
        def similarity(self, first: str, second: str) -> float:
            raise ValueError("model failed")

    ranker = RecordingRankProvider()

    with pytest.raises(ValueError, match="model failed"):
        EmbeddingGuessScorer(ExplodingEmbeddingService(), ranker).score(TEST_ANSWER, WORD)

    assert ranker.calls == [], "rank must not be consulted after the model failed"


def test_a_rank_failure_propagates_unwrapped() -> None:
    """The adapter adds no error type of its own; the catch-all handles it."""

    class ExplodingRankProvider:
        def rank_of(self, answer: str, word: str) -> int | None:
            raise RuntimeError("vocabulary failed")

    scorer = EmbeddingGuessScorer(RecordingEmbeddingService(), ExplodingRankProvider())

    with pytest.raises(RuntimeError, match="vocabulary failed"):
        scorer.score(TEST_ANSWER, WORD)


# --- Wiring: the seams behind the scorer are still injectable -----------------


def test_overriding_the_rank_provider_still_reaches_a_guess(app: FastAPI) -> None:
    """The regression this refactor could plausibly have introduced."""

    class FixedRankProvider:
        def rank_of(self, answer: str, word: str) -> int | None:
            return 11

    app.dependency_overrides[rank_provider] = FixedRankProvider
    client = TestClient(app)

    game_id = client.post("/api/games").json()["gameId"]
    guess = client.post(f"/api/games/{game_id}/guesses", json={"word": WORD})

    assert guess.json()["rank"] == 11


def test_overriding_the_embedding_service_still_reaches_a_guess(app: FastAPI) -> None:
    embedder = RecordingEmbeddingService(0.5)
    app.dependency_overrides[embedding_service] = lambda: embedder
    client = TestClient(app)

    game_id = client.post("/api/games").json()["gameId"]
    guess = client.post(f"/api/games/{game_id}/guesses", json={"word": WORD})

    assert guess.json()["similarity"] == 0.5
    assert embedder.calls == [(WORD, TEST_ANSWER)]


def test_the_default_wiring_uses_the_embedding_scorer() -> None:
    """No settings involved yet: there is exactly one scorer to build."""
    from app.api.deps import guess_scorer

    scorer = guess_scorer(DeterministicEmbeddingService(), NullRankProvider())

    assert isinstance(scorer, EmbeddingGuessScorer)


def test_the_scored_guess_still_matches_the_documented_shape(client: TestClient) -> None:
    """End to end through the refactored path, straight from docs/API_SPEC.md."""
    game_id = client.post("/api/games").json()["gameId"]

    body: Any = client.post(f"/api/games/{game_id}/guesses", json={"word": WORD}).json()

    assert set(body) == {"guessId", "word", "similarity", "rank", "isAnswer", "coordinate"}
    assert -1.0 <= body["similarity"] <= 1.0
    assert body["rank"] is None
    assert body["coordinate"] is None
