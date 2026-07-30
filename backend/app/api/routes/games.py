"""Single-player game endpoints (docs/API_SPEC.md).

Thin by design: validate input, call `GameService`, shape the response. No game
rules and no embedding calls here.
"""

from fastapi import APIRouter

from app.api.deps import GameServiceDep
from app.schemas.game import (
    CreateGameResponse,
    GameStateResponse,
    GuessRequest,
    GuessResponse,
    to_game_state_response,
    to_guess_response,
)

router = APIRouter(prefix="/api/games", tags=["games"])


@router.post("", response_model=CreateGameResponse)
def create_game(service: GameServiceDep) -> CreateGameResponse:
    """Start a game. The answer is chosen server-side and never returned."""
    game = service.create_game()
    return CreateGameResponse(
        game_id=game.id,
        status=game.status,
        created_at=game.created_at,
    )


@router.post("/{game_id}/guesses", response_model=GuessResponse)
def submit_guess(
    game_id: str,
    payload: GuessRequest,
    service: GameServiceDep,
) -> GuessResponse:
    """Score a guess.

    Re-submitting a word already guessed returns the stored result unchanged.
    A finished game rejects further guesses with `409 GAME_ALREADY_FINISHED`.
    """
    guess = service.submit_guess(game_id, payload.word)
    return to_guess_response(guess)


@router.get("/{game_id}", response_model=GameStateResponse)
def get_game(game_id: str, service: GameServiceDep) -> GameStateResponse:
    """Return the game state and full guess history (oldest guess first)."""
    return to_game_state_response(service.get_game(game_id))
