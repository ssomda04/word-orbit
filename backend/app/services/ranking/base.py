"""The seam between the game and whatever knows a guess's rank.

``rank_of`` takes the answer as a parameter rather than being bound to one, so a
provider stays stateless across games and holds no per-game memory. That is what
lets a future implementation — a precomputed artifact, a memory-mapped table, a
separate service — replace ``VectorRankProvider`` without any change to
``GameService``, the routers, or the wire schemas.

Returning ``None`` is a normal outcome, not an error: it means "this provider has
no rank for that word", which the API contract already models as ``rank: null``
(docs/API_SPEC.md).
"""

from typing import Protocol, runtime_checkable


class RankingError(ValueError):
    """Rank cannot be computed for the given inputs.

    Raised for unusable *inputs* (a blank answer, a vector the model cannot
    produce), never for a word that simply has no rank — that is ``None``.

    Mirrors ``RankTableError`` in ``ml/src/contextle_eval/rank_table.py`` so the
    two implementations fail on the same inputs, not just agree on the good ones.
    """


class NonFiniteEmbeddingError(RankingError):
    """A vector, norm, or similarity contained NaN or infinity.

    Mirrors the ML harness's ``NonFiniteEmbeddingError``. Split out from
    ``RankingError`` because a non-finite vector points at a broken model or a
    corrupt file, while the other cases point at bad input.
    """


@runtime_checkable
class RankProvider(Protocol):
    """Answers "where does this word sit among all words, for this answer?"."""

    def rank_of(self, answer: str, word: str) -> int | None:
        """Return the 1-based rank of ``word`` for ``answer``, or ``None``.

        The answer itself is always rank 1. ``None`` means the provider does not
        rank this word (typically: it is outside the vocabulary).

        Raises:
            RankingError: the answer is blank or a vector cannot be produced.
        """
        ...
