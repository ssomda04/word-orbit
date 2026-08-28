"""Tests for the pure game rules in `app.domain` — no FastAPI, no embeddings."""

import dataclasses
import random
from datetime import UTC, datetime

import pytest

from app.core.errors import GameAlreadyFinishedError, InvalidWordError
from app.domain.game import (
    MAX_WORD_LENGTH,
    FinishReason,
    Game,
    GameStatus,
    normalize_word,
)
from app.domain.words import ANSWER_WORDS, RandomAnswerSelector

ANSWER = "사과"


def _now() -> datetime:
    return datetime(2026, 7, 24, 11, 22, 57, tzinfo=UTC)


@pytest.fixture
def game() -> Game:
    return Game(id="game-1", answer=ANSWER, created_at=_now())


def _record(game: Game, word: str, similarity: float = 0.5):
    return game.record_guess(
        guess_id=f"guess-{game.guess_count + 1:03d}",
        word=word,
        similarity=similarity,
        created_at=_now(),
    )


# --- normalize_word ---------------------------------------------------------


def test_normalize_strips_surrounding_whitespace() -> None:
    assert normalize_word("  학생  ") == "학생"


def test_normalize_applies_nfkc() -> None:
    """Full-width input must collapse to its canonical form."""
    assert normalize_word("Ａ") == "A"  # FULLWIDTH LATIN CAPITAL A


def test_normalize_rejects_blank() -> None:
    with pytest.raises(InvalidWordError):
        normalize_word("   ")


def test_normalize_rejects_too_long() -> None:
    with pytest.raises(InvalidWordError):
        normalize_word("가" * (MAX_WORD_LENGTH + 1))


def test_normalize_accepts_max_length() -> None:
    word = "가" * MAX_WORD_LENGTH
    assert normalize_word(word) == word


def test_normalize_rejects_internal_whitespace() -> None:
    with pytest.raises(InvalidWordError):
        normalize_word("학생 선생")


# --- Game rules ------------------------------------------------------------


def test_new_game_is_playing_and_empty(game: Game) -> None:
    assert game.status is GameStatus.PLAYING
    assert game.guess_count == 0
    assert game.is_finished is False


def test_wrong_guess_keeps_game_playing(game: Game) -> None:
    guess = _record(game, "학생")

    assert guess.is_answer is False
    assert game.status is GameStatus.PLAYING
    assert game.guess_count == 1


def test_correct_guess_wins_the_game(game: Game) -> None:
    guess = _record(game, ANSWER, similarity=1.0)

    assert guess.is_answer is True
    assert game.status is GameStatus.WON
    assert game.is_finished is True


def test_finished_game_rejects_further_guesses(game: Game) -> None:
    _record(game, ANSWER, similarity=1.0)

    with pytest.raises(GameAlreadyFinishedError):
        _record(game, "학생")


def test_guesses_keep_submission_order(game: Game) -> None:
    _record(game, "학생")
    _record(game, "선생")
    _record(game, "학교")

    assert [guess.word for guess in game.guesses] == ["학생", "선생", "학교"]


def test_find_guess_returns_existing_or_none(game: Game) -> None:
    recorded = _record(game, "학생")

    assert game.find_guess("학생") is recorded
    assert game.find_guess("선생") is None


def test_rank_and_coordinate_default_to_none(game: Game) -> None:
    """Phase 1 contract: both fields exist but stay unpopulated."""
    guess = _record(game, "학생")

    assert guess.rank is None
    assert guess.coordinate is None


def test_guess_is_immutable(game: Game) -> None:
    guess = _record(game, "학생")

    with pytest.raises(dataclasses.FrozenInstanceError):
        guess.similarity = 0.99  # type: ignore[misc]


# --- Giving up -------------------------------------------------------------


def test_give_up_finishes_the_game(game: Game) -> None:
    game.give_up()

    assert game.status is GameStatus.ABANDONED
    assert game.is_finished is True
    assert game.finish_reason is FinishReason.GAVE_UP


def test_give_up_records_no_guess(game: Game) -> None:
    """Giving up is not an attempt, so the history is untouched."""
    _record(game, "학생")

    game.give_up()

    assert game.guess_count == 1
    assert [guess.word for guess in game.guesses] == ["학생"]


def test_give_up_keeps_the_answer_on_the_game(game: Game) -> None:
    """The reveal is the schema layer's decision; the domain only ends the round."""
    game.give_up()

    assert game.answer == ANSWER


def test_guess_after_give_up_is_rejected(game: Game) -> None:
    game.give_up()

    with pytest.raises(GameAlreadyFinishedError):
        _record(game, "학생")


def test_giving_up_twice_is_rejected(game: Game) -> None:
    game.give_up()

    with pytest.raises(GameAlreadyFinishedError):
        game.give_up()


def test_give_up_after_winning_is_rejected(game: Game) -> None:
    _record(game, ANSWER, similarity=1.0)

    with pytest.raises(GameAlreadyFinishedError):
        game.give_up()

    assert game.status is GameStatus.WON  # the rejected call changed nothing


# --- finish_reason ---------------------------------------------------------


def test_a_playing_game_has_no_finish_reason(game: Game) -> None:
    assert game.finish_reason is None


def test_a_won_game_finished_because_it_was_correct(game: Game) -> None:
    _record(game, ANSWER, similarity=1.0)

    assert game.finish_reason is FinishReason.CORRECT


# --- Answer selection ------------------------------------------------------


def test_random_answer_selector_is_seedable() -> None:
    """A pinned RNG makes answer selection reproducible in tests."""
    first = RandomAnswerSelector(rng=random.Random(42))()
    second = RandomAnswerSelector(rng=random.Random(42))()

    assert first == second
    assert first in ANSWER_WORDS


def test_random_answer_selector_rejects_empty_word_list() -> None:
    with pytest.raises(ValueError):
        RandomAnswerSelector(words=[])


def test_answer_words_are_normalized() -> None:
    """Every candidate must survive normalization, or a game could be unwinnable."""
    for word in ANSWER_WORDS:
        assert normalize_word(word) == word
