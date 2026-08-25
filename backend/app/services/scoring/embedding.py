"""The scorer that computes both values at request time, from a live model.

This is the ``GuessScorer`` the application has effectively always used: it is a
literal composition of the two calls ``GameService`` used to make itself. Keeping
it that thin is the point — it exists so the scoring seam can be introduced
without changing a single scored value, and it stays afterwards as the scorer
that a mock or a local model runs behind (tests, CI, local development).

It never returns "cannot score": an ``EmbeddingService`` produces a vector for
any non-blank string, so every accepted guess has a similarity. A scorer reading
precomputed data has no such guarantee, which is a difference in what the two
providers *can* answer, not a difference in policy — the same kind of gap as
``project_3d`` being implemented for the mock but not for FastText.
"""

from app.services.embedding import EmbeddingService
from app.services.ranking import RankProvider
from app.services.scoring.base import GuessScore


class EmbeddingGuessScorer:
    """Scores a guess with a live ``EmbeddingService`` and ``RankProvider``."""

    def __init__(self, embedder: EmbeddingService, rank_provider: RankProvider) -> None:
        self._embedder = embedder
        self._rank_provider = rank_provider

    def score(self, answer: str, word: str) -> GuessScore:
        """Compute the similarity, then the rank — in that order, as before.

        The two argument orders differ and that is not a slip: ``similarity`` is
        a symmetric comparison of two texts, while ``rank_of`` reads "where does
        this word sit, for this answer" and so leads with the answer. Both are
        passed exactly as ``GameService`` passed them before this class existed.

        The evaluation order is load-bearing too: if a model failure and a rank
        failure could both occur, the model's error is the one that propagates,
        which is what the previous code did.
        """
        similarity = self._embedder.similarity(word, answer)
        rank = self._rank_provider.rank_of(answer, word)
        return GuessScore(similarity=similarity, rank=rank)
