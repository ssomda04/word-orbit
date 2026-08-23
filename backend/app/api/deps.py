"""FastAPI dependency providers.

Routers depend on these instead of constructing services directly, which keeps
handlers testable: a test overrides `game_repository` or `answer_selector` via
`app.dependency_overrides` to get an isolated store and a pinned answer word.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.domain.words import AnswerSelector, RandomAnswerSelector
from app.services.embedding import EmbeddingService, get_embedding_service
from app.services.game import GameRepository, GameService, InMemoryGameRepository
from app.services.ranking import RankProvider, get_rank_provider


def embedding_service() -> EmbeddingService:
    return get_embedding_service()


def rank_provider() -> RankProvider:
    return get_rank_provider()


@lru_cache
def _cached_game_repository() -> GameRepository:
    """Build the process-wide store once (same pattern as the embedding factory).

    Game state lives here, so it must outlive a single request. Tests override
    the `game_repository` dependency instead of clearing this cache.
    """
    return InMemoryGameRepository()


def game_repository() -> GameRepository:
    return _cached_game_repository()


@lru_cache
def _cached_answer_selector() -> AnswerSelector:
    return RandomAnswerSelector()


def answer_selector() -> AnswerSelector:
    return _cached_answer_selector()


# Reusable annotated dependencies for route signatures.
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(embedding_service)]
GameRepositoryDep = Annotated[GameRepository, Depends(game_repository)]
AnswerSelectorDep = Annotated[AnswerSelector, Depends(answer_selector)]
RankProviderDep = Annotated[RankProvider, Depends(rank_provider)]


def game_service(
    repository: GameRepositoryDep,
    embedder: EmbeddingServiceDep,
    selector: AnswerSelectorDep,
    ranker: RankProviderDep,
) -> GameService:
    """Construct the service per request — cheap, since all state is in the store."""
    return GameService(
        repository=repository,
        embedder=embedder,
        answer_selector=selector,
        rank_provider=ranker,
    )


GameServiceDep = Annotated[GameService, Depends(game_service)]
