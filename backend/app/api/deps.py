"""FastAPI dependency providers.

Routers depend on these instead of constructing services directly, which keeps
handlers testable: a test overrides `game_repository` or `answer_selector` via
`app.dependency_overrides` to get an isolated store and a pinned answer word.

Why the artifact dependencies are built by a function
-----------------------------------------------------
`guess_scorer` is a dependency; `artifact_guess_scorer_for` and
`artifact_answer_selector_for` are functions that *return* one, bound to a
particular `Settings`. `app.main.create_app` calls them once, with the settings
that app was created from. Two separate problems make that the shape.

**The embedding stack must not be built in artifact mode.** FastAPI resolves
every declared sub-dependency before calling the function that declared them, so
one dependency cannot both

1. declare `embedding_service` and `rank_provider` — which is what keeps
   overriding either of them changing what a guess scores — and
2. avoid building them when nothing will call them, which with
   `EMBEDDING_PROVIDER=fasttext` is a multi-gigabyte model loaded to serve a
   scorer that never touches it.

So the choice of *which* dependency moves up to where the app is built, and each
one stays honest about what it needs.

**An app must serve the configuration it was created from.** `create_app` takes
an explicit `Settings`, so a dependency reaching for `get_settings()` could
answer from a different one — a different artifact root than the app selected
artifact mode from. Binding the settings into the dependency at wiring time
removes the question: there is no global to consult, and so no chance of
consulting the wrong one.

Note what is *not* here as a result. The answer selector has no provider branch:
in artifact mode `create_app` overrides it with a bound one, so the cached
default stays the placeholder-word selector it has always been, belonging to the
one provider that uses it.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from app.core.config import Settings
from app.domain.words import AnswerSelector, RandomAnswerSelector
from app.services.embedding import EmbeddingService, get_embedding_service
from app.services.game import GameRepository, GameService, InMemoryGameRepository
from app.services.ranking import RankProvider, get_rank_provider
from app.services.scoring import (
    ArtifactGuessScorer,
    EmbeddingGuessScorer,
    GuessScorer,
    get_artifact_store,
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
    return RandomAnswerSelector()


def answer_selector() -> AnswerSelector:
    """Draw a new game's answer from the placeholder word list.

    Artifact mode replaces this wholesale (`artifact_answer_selector_for`),
    because there the answers a game may have are exactly the answers the loaded
    root can serve. Keeping that out of here is what stops one provider caching
    a selector the other would inherit.
    """
    return _cached_answer_selector()


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


def artifact_guess_scorer_for(settings: Settings) -> Callable[[], GuessScorer]:
    """Return the artifact-mode scorer dependency, bound to ``settings``.

    Note the returned dependency's empty parameter list. It is the production
    isolation, stated in the only way FastAPI can enforce: a dependency that
    declares no embedding seam cannot cause one to be built. The store behind it
    is expensive and cached; the scorer is a single attribute assignment, so it
    is built per request like its embedding counterpart.
    """

    def _artifact_guess_scorer() -> GuessScorer:
        return ArtifactGuessScorer(get_artifact_store(settings))

    return _artifact_guess_scorer


def artifact_answer_selector_for(settings: Settings) -> Callable[[], AnswerSelector]:
    """Return the artifact-mode answer selector dependency, bound to ``settings``.

    The same store the scorer reads, because both ask for it with the same
    settings and the store is cached under exactly that configuration. An app
    therefore cannot draw an answer from one root and score guesses against
    another.

    Built per request rather than cached: ``store.answers`` is already a tuple so
    ``RandomAnswerSelector`` re-wraps nothing, and a fresh ``random.Random``
    costs a seed. Both are trivial next to creating a game, and neither is worth
    a second cache whose invalidation would have to track this one's settings.
    """

    def _artifact_answer_selector() -> AnswerSelector:
        return RandomAnswerSelector(get_artifact_store(settings).answers)

    return _artifact_answer_selector


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
