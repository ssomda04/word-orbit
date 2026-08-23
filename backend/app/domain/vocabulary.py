"""The word set that ranks are computed over.

Why this is not ``app.domain.game.normalize_word``
--------------------------------------------------
That function normalizes a *guess*, so it also enforces guess rules: a length
cap and no internal whitespace. Applying it here would silently drop vocabulary
entries — a dictionary headword like ``"고유 명사"`` contains a space — and every
word ranked below a dropped entry would shift up by one. The vocabulary must be
normalized by the vocabulary policy only.

The policy is deliberately identical to ``ml/src/contextle_eval/rank_table.py``
(``normalize_word`` / ``normalize_vocabulary`` / ``load_vocabulary``): NFKC,
strip, drop blanks, deduplicate keeping the first occurrence, and read files as
UTF-8 with an optional BOM. The backend and the ML harness produce the same
ranks only if they first agree on the same word set, so any change here is a
change to a shared contract — see ``tests/test_ranking.py`` for the parity
tests that pin it.

Both normalizers share NFKC+strip as their core, so a word the game accepts as a
guess always normalizes to the same string under either function. The two only
diverge on inputs the game rejects outright.
"""

import unicodedata
from collections.abc import Iterable
from pathlib import Path


class VocabularyError(ValueError):
    """The vocabulary cannot be built or read."""


def normalize_vocabulary_word(word: str) -> str:
    """Apply NFKC and trim. Returns ``""`` for a blank entry (callers drop it).

    Raises:
        VocabularyError: ``word`` is not a string.
    """
    if not isinstance(word, str):
        raise VocabularyError("Vocabulary entries and the answer must be strings.")
    return unicodedata.normalize("NFKC", word).strip()


def normalize_vocabulary(words: Iterable[str]) -> tuple[str, ...]:
    """Normalize, drop blanks, and keep the first occurrence of each word.

    Insertion order is preserved so the result is stable, but rank never depends
    on it: ranks are derived from a total order over (similarity, word).
    """
    unique: dict[str, None] = {}
    for word in words:
        normalized = normalize_vocabulary_word(word)
        if normalized:
            unique.setdefault(normalized, None)
    return tuple(unique)


def load_vocabulary(path: Path) -> tuple[str, ...]:
    """Load one word per line from a UTF-8 file (BOM tolerated).

    Raises:
        VocabularyError: the file is missing, unreadable, or not valid UTF-8.
    """
    try:
        with path.open(encoding="utf-8-sig") as vocabulary_file:
            return normalize_vocabulary(vocabulary_file)
    except FileNotFoundError as exc:
        raise VocabularyError(f"Vocabulary file not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise VocabularyError(f"Vocabulary file is not valid UTF-8: {path}") from exc
    except OSError as exc:
        raise VocabularyError(f"Could not read vocabulary file {path}: {exc}") from exc
