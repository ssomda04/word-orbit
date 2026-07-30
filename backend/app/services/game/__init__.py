"""Game service package: the storage seam plus single-player use cases.

Callers depend on ``GameRepository`` (a Protocol), so the in-memory store can be
replaced by Redis/Postgres in Phase 3 without touching ``GameService``.
"""

from app.services.game.repository import GameRepository, InMemoryGameRepository
from app.services.game.service import GameService

__all__ = [
    "GameRepository",
    "GameService",
    "InMemoryGameRepository",
]
