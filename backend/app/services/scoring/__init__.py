"""Guess scoring: what one guess is worth, relative to one answer.

``GameService`` depends only on the ``GuessScorer`` Protocol, so where the
similarity and rank come from — a live model, or data precomputed offline — is
a wiring decision made in ``app.api.deps``.

There is no factory module here, deliberately. ``EmbeddingGuessScorer`` is
stateless and costs nothing to build, so it follows the same pattern as
``GameService``: composed per request from the seams that *are* process-wide
(``app.services.embedding``, ``app.services.ranking``). A scorer that owns
expensive state would need the cached-factory-plus-lock treatment those two use,
and should bring its own factory when it arrives.
"""

from app.services.scoring.base import GuessScore, GuessScorer
from app.services.scoring.embedding import EmbeddingGuessScorer

__all__ = [
    "EmbeddingGuessScorer",
    "GuessScore",
    "GuessScorer",
]
