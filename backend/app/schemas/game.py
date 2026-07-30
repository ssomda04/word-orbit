"""Wire schemas for the single-player game API (docs/API_SPEC.md).

Note what is *absent*: no response model here carries the answer word except
``GameStateResponse``, and its value is set in exactly one place —
``to_game_state_response()``. That keeps "never reveal the answer early" to a
single reviewable line.
"""

from pydantic import Field, field_validator

from app.domain.game import Game, GameStatus, Guess
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
