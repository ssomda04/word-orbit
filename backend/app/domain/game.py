"""Pure single-player game rules — no FastAPI, no embedding model.

``Game`` owns the invariants of one round: which guesses exist, whether a guess
wins, and the resulting status transition. Similarity scores are computed
elsewhere (``app.services.game.service``) and passed in, which keeps this module
trivially unit-testable.

The answer word lives here and never reaches a response schema on its own; see
``app.schemas.game.to_game_state_response`` and ``to_give_up_response`` for the
only two reveal points, both of which require the game to have left ``PLAYING``.
"""

import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from app.core.errors import GameAlreadyFinishedError, InvalidWordError

# Upper bound on a guess. The deterministic mock would happily hash any length,
# so the limit is a product rule rather than a model constraint.
MAX_WORD_LENGTH = 50


class GameStatus(StrEnum):
    """Game lifecycle (docs/API_SPEC.md).

    ``ABANDONED`` is the state a game reaches when the player gives up
    (``Game.give_up``). It was reserved by the contract before any endpoint
    produced it; there is still no attempt limit, so those two endings —
    ``WON`` and ``ABANDONED`` — remain the only ways a game finishes.
    """

    PLAYING = "playing"
    WON = "won"
    ABANDONED = "abandoned"


class FinishReason(StrEnum):
    """Why a finished game ended (docs/API_SPEC.md).

    Derived from ``GameStatus`` rather than stored alongside it: the status
    already records which of the two endings happened, so a second stored field
    would only add a way for the two to disagree.
    """

    CORRECT = "correct"
    GAVE_UP = "gave_up"


# The whole mapping, in one place. A finishing status that is missing here has
# no reason to report, which `Game.finish_reason` surfaces as `None`.
_FINISH_REASONS: dict[GameStatus, FinishReason] = {
    GameStatus.WON: FinishReason.CORRECT,
    GameStatus.ABANDONED: FinishReason.GAVE_UP,
}


def normalize_word(raw: str) -> str:
    """Return the canonical form used for *both* comparison and embedding.

    Using one function for both is deliberate: if the answer comparison and the
    vector lookup normalized differently, a guess could score 1.0 without being
    accepted as the answer.

    Blank input is normally rejected earlier by ``GuessRequest`` (422
    ``INVALID_INPUT``); the check here is a safety net for non-HTTP callers.

    Raises:
        InvalidWordError: the word is blank, too long, or contains whitespace.
    """
    normalized = unicodedata.normalize("NFKC", raw).strip()
    if not normalized:
        raise InvalidWordError("단어가 비어 있습니다.")
    if len(normalized) > MAX_WORD_LENGTH:
        raise InvalidWordError(f"단어는 {MAX_WORD_LENGTH}자를 넘을 수 없습니다.")
    # Phase 1 is word-level; sentence mode (Phase 4) will need to relax this.
    if any(char.isspace() for char in normalized):
        raise InvalidWordError("단어에 공백을 포함할 수 없습니다.")
    return normalized


@dataclass(frozen=True, slots=True)
class Guess:
    """One scored guess. Immutable — a recorded result never changes."""

    id: str
    word: str
    similarity: float
    is_answer: bool
    created_at: datetime
    # Both nullable by contract (docs/API_SPEC.md). `rank` is null when no
    # vocabulary is configured or the word falls outside it; `coordinate` waits
    # on the Phase 2 projection.
    rank: int | None = None
    coordinate: tuple[float, float, float] | None = None


@dataclass(slots=True)
class Game:
    """One single-player round, including the hidden answer."""

    id: str
    answer: str  # server-only; never serialized while the game is playing
    created_at: datetime
    status: GameStatus = GameStatus.PLAYING
    # Submission order is preserved: it is the raw history the API returns, and
    # the Phase 2 exploration path depends on it.
    guesses: list[Guess] = field(default_factory=list)

    @property
    def guess_count(self) -> int:
        return len(self.guesses)

    @property
    def is_finished(self) -> bool:
        return self.status is not GameStatus.PLAYING

    @property
    def finish_reason(self) -> FinishReason | None:
        """Why this game ended, or ``None`` while it is still playing."""
        return _FINISH_REASONS.get(self.status)

    def give_up(self) -> None:
        """End the round at the player's request.

        The same transition ``record_guess`` performs on a winning guess, with
        the other ending: the game leaves ``PLAYING``, which is what stops
        further guesses and what makes the answer revealable. Nothing is
        recorded in ``guesses`` — giving up is not an attempt, so ``guessCount``
        does not grow.

        Raises:
            GameAlreadyFinishedError: the game has already ended, whether by a
                correct guess or by an earlier give-up.
        """
        if self.is_finished:
            raise GameAlreadyFinishedError()

        self.status = GameStatus.ABANDONED

    def find_guess(self, word: str) -> Guess | None:
        """Return the stored guess for ``word``, or ``None`` if it is new."""
        return next((guess for guess in self.guesses if guess.word == word), None)

    def record_guess(
        self,
        *,
        guess_id: str,
        word: str,
        similarity: float,
        created_at: datetime,
        rank: int | None = None,
    ) -> Guess:
        """Append a scored guess and apply the win transition.

        ``rank`` is supplied by the caller for the same reason ``similarity`` is:
        it comes from a vocabulary and an embedding model, neither of which
        belongs in the domain. ``None`` means no rank is available, which is a
        normal outcome (no vocabulary configured, or the word is outside it).

        Raises:
            GameAlreadyFinishedError: the game no longer accepts guesses.
        """
        if self.is_finished:
            raise GameAlreadyFinishedError()

        guess = Guess(
            id=guess_id,
            word=word,
            similarity=similarity,
            is_answer=word == self.answer,
            created_at=created_at,
            rank=rank,
        )
        self.guesses.append(guess)
        if guess.is_answer:
            self.status = GameStatus.WON
        return guess
