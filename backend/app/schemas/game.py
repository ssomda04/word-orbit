"""Wire schemas for the single-player game API (docs/API_SPEC.md).

Note what is *absent*: only two response models here carry the answer word —
``GameStateResponse`` and ``GiveUpResponse`` — and each one gets its value from
exactly one mapper, ``to_game_state_response()`` and ``to_give_up_response()``.
Both mappers refuse to reveal anything for a game still ``playing``, which keeps
"never reveal the answer early" to two reviewable lines.
"""

from pydantic import Field, field_validator

from app.domain.game import FinishReason, Game, GameStatus, Guess
from app.schemas.base import CamelModel, UtcTimestamp


class Coordinate(CamelModel):
    """A position on the 3D semantic map. Unused until Phase 2."""

    x: float
    y: float
    z: float


class CreateGameResponse(CamelModel):
    """Response of `POST /api/games`. Has no answer field by design."""

    game_id: str
    status: GameStatus
    created_at: UtcTimestamp


class GuessRequest(CamelModel):
    word: str = Field(..., min_length=1, description="Word to guess.")

    @field_validator("word")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty or whitespace-only")
        return stripped


class GuessResponse(CamelModel):
    guess_id: str
    word: str
    similarity: float
    # Both stay null in Phase 1 by contract: `rank` needs precomputed nearest
    # neighbours, `coordinate` needs the Phase 2 projection.
    rank: int | None = None
    is_answer: bool
    coordinate: Coordinate | None = None


class GiveUpResponse(CamelModel):
    """Response of `POST /api/games/{gameId}/give-up`.

    `answer` is non-nullable here, unlike on `GameStateResponse`: this response
    exists only for a game that has just ended, so there is no in-progress case
    for it to represent.
    """

    game_id: str
    status: GameStatus
    finish_reason: FinishReason
    answer: str


class GameStateResponse(CamelModel):
    game_id: str
    status: GameStatus
    created_at: UtcTimestamp
    guess_count: int
    # Submission order, oldest first; sorting by similarity is the client's job.
    guesses: list[GuessResponse]
    answer: str | None = None


def to_guess_response(guess: Guess) -> GuessResponse:
    coordinate = None
    if guess.coordinate is not None:
        x, y, z = guess.coordinate
        coordinate = Coordinate(x=x, y=y, z=z)

    return GuessResponse(
        guess_id=guess.id,
        word=guess.word,
        similarity=guess.similarity,
        rank=guess.rank,
        is_answer=guess.is_answer,
        coordinate=coordinate,
    )


def to_game_state_response(game: Game) -> GameStateResponse:
    """Map a game to its wire form.

    The only path by which the answer word can reach a client: it is included
    solely once the game has left `playing` (the reveal phase in
    `docs/API_SPEC.md`), and is `None` for every in-progress game.
    """
    is_revealed = game.status is not GameStatus.PLAYING

    return GameStateResponse(
        game_id=game.id,
        status=game.status,
        created_at=game.created_at,
        guess_count=game.guess_count,
        guesses=[to_guess_response(guess) for guess in game.guesses],
        answer=game.answer if is_revealed else None,
    )


def to_give_up_response(game: Game) -> GiveUpResponse:
    """Map a game the player has just given up to its wire form.

    The second and last path by which the answer word can reach a client. The
    reveal is guarded the same way as in `to_game_state_response`, by the game's
    own status rather than by who called: `finish_reason` is `None` for exactly
    the games that are still `playing`, so an in-progress game fails here
    instead of being serialized with its answer.
    """
    reason = game.finish_reason
    if reason is None:
        # Unreachable through the router — `GameService.give_up` has already
        # ended the game. Stated as a hard failure rather than a default,
        # because the only alternative would be sending the answer word for a
        # game still in progress. The resulting 500 says nothing (see
        # `app.main.INTERNAL_ERROR_MESSAGE`), and this message never names the
        # answer.
        raise ValueError("cannot build a give-up response for a game still playing")

    return GiveUpResponse(
        game_id=game.id,
        status=game.status,
        finish_reason=reason,
        answer=game.answer,
    )
