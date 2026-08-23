"""The provider used when no vocabulary is configured.

Ranking needs a word set to rank *against*. Without ``VOCABULARY_PATH`` there is
none, so every rank is ``None`` and `GuessResponse.rank` stays ``null`` — which
is exactly what the contract has always said it may be (docs/API_SPEC.md). This
keeps the default install, the test suite, and CI working with no vocabulary
file and no behaviour change.
"""


class NullRankProvider:
    """A ``RankProvider`` that ranks nothing."""

    def rank_of(self, answer: str, word: str) -> int | None:
        """Always ``None``. Never raises — there is nothing to validate."""
        return None
