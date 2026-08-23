"""Tests for `GameService` — dependencies are injected, so behaviour is exact.

A fake embedder records its calls, which is how we assert that scoring happens
in the service (never in the router) and exactly once per new guess.
"""

import uuid
from collections.abc import Sequence

import pytest

from app.core.errors import GameAlreadyFinishedError, GameNotFoundError, InvalidWordError
from app.domain.game import MAX_WORD_LENGTH, GameStatus
from app.services.embedding import DeterministicEmbeddingService
from app.services.game import GameService, InMemoryGameRepository
from app.services.ranking import NullRankProvider

ANSWER = "사과"


class FakeEmbeddingService:
    """Minimal `EmbeddingService` stand-in that records `similarity` calls."""

    def __init__(self, score: float = 0.5) -> None:
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


@pytest.fixture
def embedder() -> FakeEmbeddingService:
    return FakeEmbeddingService()


@pytest.fixture
def service(embedder: FakeEmbeddingService) -> GameService:
    return GameService(
        repository=InMemoryGameRepository(),
        embedder=embedder,
        answer_selector=lambda: ANSWER,
        # No vocabulary in these tests, so every rank is None — the default
        # wiring. Rank policy itself is covered in tests/test_ranking.py.
        rank_provider=NullRankProvider(),
    )


# --- create_game -----------------------------------------------------------


def test_create_game_starts_in_playing_state(service: GameService) -> None:
    game = service.create_game()

    assert game.status is GameStatus.PLAYING
    assert game.guess_count == 0
    assert game.answer == ANSWER


def test_create_game_is_retrievable(service: GameService) -> None:
    created = service.create_game()

    assert service.get_game(created.id) is created


def test_game_id_is_a_uuid(service: GameService) -> None:
    """Opaque, non-enumerable ids: a client cannot walk to another game."""
    game = service.create_game()

    assert uuid.UUID(game.id)


def test_game_ids_are_unique(service: GameService) -> None:
    ids = {service.create_game().id for _ in range(5)}

    assert len(ids) == 5


def test_get_unknown_game_raises(service: GameService) -> None:
    with pytest.raises(GameNotFoundError):
        service.get_game("does-not-exist")


# --- submit_guess ----------------------------------------------------------


def test_submit_guess_scores_against_the_answer(
    service: GameService, embedder: FakeEmbeddingService
) -> None:
    game = service.create_game()

    guess = service.submit_guess(game.id, "학생")

    assert embedder.calls == [("학생", ANSWER)]
    assert guess.similarity == embedder.score
    assert guess.is_answer is False


def test_submit_guess_normalizes_the_word(service: GameService) -> None:
    game = service.create_game()

    guess = service.submit_guess(game.id, "  학생  ")

    assert guess.word == "학생"


def test_guess_ids_increment_within_a_game(service: GameService) -> None:
    game = service.create_game()

    first = service.submit_guess(game.id, "학생")
    second = service.submit_guess(game.id, "선생")

    assert (first.id, second.id) == ("guess-001", "guess-002")


def test_correct_guess_wins(service: GameService) -> None:
    game = service.create_game()

    guess = service.submit_guess(game.id, ANSWER)

    assert guess.is_answer is True
    assert service.get_game(game.id).status is GameStatus.WON


def test_duplicate_guess_is_idempotent(
    service: GameService, embedder: FakeEmbeddingService
) -> None:
    """Re-guessing a word returns the stored result and creates nothing new."""
    game = service.create_game()

    first = service.submit_guess(game.id, "학생")
    repeated = service.submit_guess(game.id, "학생")

    assert repeated is first
    assert service.get_game(game.id).guess_count == 1
    assert len(embedder.calls) == 1  # no second scoring call


def test_duplicate_detection_uses_the_normalized_word(service: GameService) -> None:
    game = service.create_game()

    first = service.submit_guess(game.id, "학생")
    repeated = service.submit_guess(game.id, "  학생 ")

    assert repeated is first
    assert service.get_game(game.id).guess_count == 1


def test_guess_on_unknown_game_raises(service: GameService) -> None:
    with pytest.raises(GameNotFoundError):
        service.submit_guess("does-not-exist", "학생")


def test_guess_on_finished_game_raises(service: GameService) -> None:
    game = service.create_game()
    service.submit_guess(game.id, ANSWER)

    with pytest.raises(GameAlreadyFinishedError):
        service.submit_guess(game.id, "학생")


def test_replaying_the_winning_guess_on_a_finished_game_raises(service: GameService) -> None:
    """Finished beats idempotent: a closed game accepts nothing at all."""
    game = service.create_game()
    service.submit_guess(game.id, ANSWER)

    with pytest.raises(GameAlreadyFinishedError):
        service.submit_guess(game.id, ANSWER)


def test_invalid_word_raises(service: GameService) -> None:
    game = service.create_game()

    with pytest.raises(InvalidWordError):
        service.submit_guess(game.id, "가" * (MAX_WORD_LENGTH + 1))


def test_guesses_keep_submission_order(service: GameService) -> None:
    game = service.create_game()
    for word in ("학생", "선생", "학교"):
        service.submit_guess(game.id, word)

    stored = service.get_game(game.id)

    assert [guess.word for guess in stored.guesses] == ["학생", "선생", "학교"]
    assert stored.guess_count == 3


# --- with the real deterministic embedder ----------------------------------


@pytest.fixture
def real_service() -> GameService:
    return GameService(
        repository=InMemoryGameRepository(),
        embedder=DeterministicEmbeddingService(),
        answer_selector=lambda: ANSWER,
        rank_provider=NullRankProvider(),
    )


def test_similarity_stays_in_contract_range(real_service: GameService) -> None:
    game = real_service.create_game()

    for word in ("학생", "선생", "컴퓨터"):
        guess = real_service.submit_guess(game.id, word)
        assert -1.0 <= guess.similarity <= 1.0


def test_answer_guess_scores_one(real_service: GameService) -> None:
    """Self-similarity is 1.0, so the winning guess reports a perfect score."""
    game = real_service.create_game()

    guess = real_service.submit_guess(game.id, ANSWER)

    assert guess.similarity == pytest.approx(1.0)
    assert guess.is_answer is True
