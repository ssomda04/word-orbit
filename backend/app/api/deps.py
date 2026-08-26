"""FastAPI dependency providers.

Routers depend on these instead of constructing services directly, which keeps
handlers testable: a test overrides `game_repository` or `answer_selector` via
`app.dependency_overrides` to get an isolated store and a pinned answer word.

Why there are two guess-scorer dependencies
-------------------------------------------
`guess_scorer` and `artifact_guess_scorer` do the same job for the two scoring
providers, and `app.main.create_app` picks one, once. That is not indecision —
it is the only shape that satisfies both of the requirements below at the same
time, because FastAPI resolves every declared sub-dependency before it calls the
function that declared them:

1. Overriding `embedding_service` or `rank_provider` must still change what a
   guess scores, which means the embedding scorer has to *declare* both.
2. In artifact mode the embedding stack must not be built at all — no model
   load, no vocabulary matrix — which means the artifact scorer must declare
   neither.

One function cannot do both: declaring the two seams would resolve them in
artifact mode too, and with `EMBEDDING_PROVIDER=fasttext` that is a multi-
gigabyte model loaded to serve a scorer that never calls it. So the choice moves
one level up, to where the app is built, and each dependency stays honest about
what it needs.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.domain.words import AnswerSelector, RandomAnswerSelector
from app.services.embedding import EmbeddingService, get_embedding_service
from app.services.game import GameRepository, GameService, InMemoryGameRepository
from app.services.ranking import RankProvider, get_rank_provider
from app.services.scoring import (
    ArtifactGuessScorer,
    EmbeddingGuessScorer,
    GuessScorer,
    ScoringProvider,
    get_artifact_store,
    resolve_scoring_provider,
)


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
    """Choose the word list a new game's answer is drawn from.

    One selector class, two sources. In artifact mode the answers are exactly
    the answers the loaded root can serve, because an answer with no artifact is
    a game nobody can score; `ANSWER_WORDS` is a placeholder list that has no
    relationship to any root and must not be used there. In embedding mode a
    live model can score any answer, so the placeholder list stays.

    Cached because it is process-wide, like the store it reads. It is *provider*
    dependent, so `reset_answer_selector` exists alongside the other factory
    resets — a test that switches providers has to drop this too.
    """
    if resolve_scoring_provider() is ScoringProvider.ARTIFACT:
        return RandomAnswerSelector(get_artifact_store().answers)
    return RandomAnswerSelector()


def answer_selector() -> AnswerSelector:
    return _cached_answer_selector()


def reset_answer_selector() -> None:
    """Drop the cached selector so the next call rebuilds it.

    For tests only: production wiring builds it once and never resets it.
    """
    _cached_answer_selector.cache_clear()


# Reusable annotated dependencies for route signatures.
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(embedding_service)]
GameRepositoryDep = Annotated[GameRepository, Depends(game_repository)]
AnswerSelectorDep = Annotated[AnswerSelector, Depends(answer_selector)]
RankProviderDep = Annotated[RankProvider, Depends(rank_provider)]


def guess_scorer(embedder: EmbeddingServiceDep, ranker: RankProviderDep) -> GuessScorer:
    """Compose the embedding-mode scorer from the two seams it reads through.

    Built per request rather than cached, for the same reason ``game_service``
    is: it owns no state, so construction is a couple of attribute assignments.
    Composing it from the *dependencies* — not from the module factories
    directly — is what keeps `app.dependency_overrides[rank_provider]` (and the
    embedding equivalent) reaching the guess path, as they always have.
    """
    return EmbeddingGuessScorer(embedder, ranker)


def artifact_guess_scorer() -> GuessScorer:
    """Compose the artifact-mode scorer from the process-wide store.

    Note the empty parameter list. It is the production isolation, stated in the
    only way FastAPI can enforce: a dependency that declares no embedding seam
    cannot cause one to be built. The store is expensive and cached; the scorer
    around it is a single attribute assignment, so it is built per request like
    its embedding counterpart.
    """
    return ArtifactGuessScorer(get_artifact_store())


GuessScorerDep = Annotated[GuessScorer, Depends(guess_scorer)]


def game_service(
    repository: GameRepositoryDep,
    scorer: GuessScorerDep,
    selector: AnswerSelectorDep,
) -> GameService:
    """Construct the service per request — cheap, since all state is in the store."""
    return GameService(
        repository=repository,
        scorer=scorer,
        answer_selector=selector,
    )


GameServiceDep = Annotated[GameService, Depends(game_service)]
