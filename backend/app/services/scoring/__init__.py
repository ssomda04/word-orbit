"""Guess scoring: what one guess is worth, relative to one answer.

``GameService`` depends only on the ``GuessScorer`` Protocol, so where the
similarity and rank come from — a live model, or data precomputed offline — is
a configuration choice, resolved once in ``factory`` and wired in
``app.api.deps``.

The two scorers are cached differently because they own different things.
``EmbeddingGuessScorer`` is stateless and costs nothing to build, so it follows
the same pattern as ``GameService``: composed per request from the seams that
*are* process-wide (``app.services.embedding``, ``app.services.ranking``).
``ArtifactGuessScorer`` reads through an ``ArtifactStore``, which owns a
validated manifest and a bounded array cache, so the store gets the
cached-factory-plus-lock treatment those two seams use — which is what ``factory``
is for.
"""

from app.services.scoring.artifact import ArtifactGuessScorer, ArtifactStore
from app.services.scoring.base import GuessScore, GuessScorer
from app.services.scoring.embedding import EmbeddingGuessScorer
from app.services.scoring.factory import (
    ScoringProvider,
    get_artifact_store,
    reset_artifact_store,
    resolve_scoring_provider,
)

__all__ = [
    "ArtifactGuessScorer",
    "ArtifactStore",
    "EmbeddingGuessScorer",
    "GuessScore",
    "GuessScorer",
    "ScoringProvider",
    "get_artifact_store",
    "reset_artifact_store",
    "resolve_scoring_provider",
]
