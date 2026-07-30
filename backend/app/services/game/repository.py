"""Game storage. In-memory today; the Protocol is the seam for Redis/Postgres.

Concurrency: route handlers are sync ``def``, so FastAPI runs them in a thread
pool and two requests can touch the store at the same time. One lock guards the
dict — cheap, and sufficient for a process-local store.

Nothing here survives a restart; persistence arrives with Phase 3
(docs/ROADMAP.md).
"""

import threading
import uuid
from typing import Protocol, runtime_checkable

from app.domain.game import Game


@runtime_checkable
class GameRepository(Protocol):
    """Storage and identifier allocation for games. Holds no game rules."""

    def add(self, game: Game) -> None:
        """Store a newly created game."""
        ...

    def get(self, game_id: str) -> Game | None:
        """Return the game, or ``None`` if it does not exist (never raises)."""
        ...

    def save(self, game: Game) -> None:
        """Persist changes to an existing game."""
        ...

    def next_game_id(self) -> str:
        """Allocate an opaque game identifier."""
        ...

    def next_guess_id(self, game: Game) -> str:
        """Allocate an identifier for the next guess of ``game``."""
        ...


class InMemoryGameRepository:
    """Process-local, dict-backed ``GameRepository``."""

    def __init__(self) -> None:
        self._games: dict[str, Game] = {}
        self._lock = threading.Lock()

    def add(self, game: Game) -> None:
        with self._lock:
            self._games[game.id] = game

    def get(self, game_id: str) -> Game | None:
        with self._lock:
            return self._games.get(game_id)

    def save(self, game: Game) -> None:
        # A no-op in practice (callers mutate the stored object), but keeping it
        # explicit means a database implementation can slot in unchanged.
        with self._lock:
            self._games[game.id] = game

    def next_game_id(self) -> str:
        # Full UUID4: opaque and not enumerable, so a client cannot walk other
        # players' games. Clients must not parse IDs (docs/API_SPEC.md).
        return str(uuid.uuid4())

    def next_guess_id(self, game: Game) -> str:
        # Sequential within one game: a guess id is only meaningful next to its
        # game, and it keeps the documented `guess-001` shape.
        return f"guess-{game.guess_count + 1:03d}"
