"""The seam between a game and whatever knows what a guess is worth.

Why one seam for two values
---------------------------
``similarity`` and ``rank`` answer the same question — how close is this guess to
the answer? — and a client reads them side by side on a single guess. Today they
are produced by two independent seams (``EmbeddingService``, ``RankProvider``)
and that is harmless, because both are derived from the same vectors, in the
same process, at the same moment.

That stops being true the moment either value is precomputed. A stored artifact
holds a word's similarity and rank as two cells of one row, so fetching them
through separate seams would mean locating that row twice, and would leave room
for a similarity from one source to be reported next to a rank from another.
Scoring is therefore modelled as one lookup that returns both.

``score`` takes the answer as a parameter rather than being bound to one, for
exactly the reason ``RankProvider.rank_of`` does: a scorer stays stateless across
games and holds no per-game memory.

Why "cannot score" is ``None`` and not an exception
---------------------------------------------------
A live model produces a vector for any non-blank string, so every accepted guess
has a similarity. A scorer reading precomputed data does not: an artifact holds
a row for exactly the words in its canonical vocabulary, and there is nothing to
read for a word outside it.

That gap is reported by returning ``None``. It is deliberately *not* reported by
raising ``InvalidWordError``, even though that is what the client eventually
sees. ``INVALID_WORD`` is a game rule carrying an HTTP status; a scorer only
knows whether it holds a number. Keeping the two apart means the scoring layer
stays free of application semantics, and ``GameService`` — which already turns a
finished game into a conflict — owns the one place a missing score becomes an
error.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class GuessScore:
    """What one guess is worth, relative to one answer.

    Mirrors the two scored fields of ``GuessResponse`` (docs/API_SPEC.md) and
    nothing else: ``isAnswer`` is a domain rule, ``coordinate`` is Phase 2.

    ``rank`` is ``None`` when the scorer has no rank for the word — no
    vocabulary configured, or the word falls outside it. That is a normal
    outcome, not an error, and it surfaces as ``rank: null``.
    """

    similarity: float
    rank: int | None = None


@runtime_checkable
class GuessScorer(Protocol):
    """Scores one guess against one answer in a single lookup."""

    def score(self, answer: str, word: str) -> GuessScore | None:
        """Return the similarity and rank of ``word`` with respect to ``answer``.

        ``None`` means this scorer cannot score ``word`` at all — typically that
        the word is outside the data it reads. That is a normal outcome, not an
        error; the caller decides what it means for the game. An implementation
        that can score anything simply never returns it.

        Both arguments are already normalized by ``app.domain.game.normalize_word``.
        Implementations MUST NOT log or return the answer word.
        """
        ...
