"""Reading the canonical vocabulary that an artifact root indexes by.

Why this does not reuse ``app.domain.vocabulary.load_vocabulary``
-----------------------------------------------------------------
That loader is deliberately forgiving: it reads ``utf-8-sig``, so a byte-order
mark is swallowed, and it accepts blank lines and duplicates because it
normalizes them away. Those are the right manners for a file an operator hand-
maintains and hands to ``VOCABULARY_PATH``.

They are the wrong manners here. ``vocabulary.txt`` is not input, it is *part of
an artifact* — the exact byte sequence whose sha256 the manifest records and
whose line order every stored array is indexed by. Anything the reader silently
repairs is a difference between what the arrays were built against and what the
server believes they were built against, and a repaired vocabulary still hashes
to whatever it happens to hash to. So this reader repairs nothing: it either
reads back byte-for-byte what a canonical writer would have produced, or it
fails.

A BOM is the sharpest example. ``utf-8-sig`` would strip it; strict UTF-8 decodes
it to U+FEFF, which is not whitespace, so it survives ``strip()`` and merges into
the first word (``'\\ufeff가'``) — an entry no guess can ever match, and one that
shifts nothing else, so the corruption is invisible. It is rejected explicitly.

The *normalization policy* is shared, not re-implemented:
``app.domain.vocabulary.normalize_vocabulary`` is the single definition of NFKC +
strip + drop-blank + first-occurrence dedup, and it is already the function whose
output reproduces the artifact hash.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from app.domain.vocabulary import normalize_vocabulary
from app.services.scoring.artifact.errors import ArtifactError

# UTF-8 encoding of U+FEFF. Present at the start of a file it is a byte-order
# mark; strict UTF-8 keeps it as a character, so it must be refused up front.
_UTF8_BOM = b"\xef\xbb\xbf"


@dataclass(frozen=True, slots=True)
class CanonicalVocabulary:
    """The word list an artifact root's arrays are indexed by.

    ``words[i]`` is the word that ``similarity[i]`` and ``rank[i]`` describe, so
    line order is part of the contract, not a presentation detail.
    """

    words: tuple[str, ...]
    sha256: str
    _index: Mapping[str, int] = field(repr=False, compare=False)

    @property
    def size(self) -> int:
        return len(self.words)

    def index_of(self, word: str) -> int | None:
        """Return the canonical index of an already-normalized ``word``, or None."""
        return self._index.get(word)

    def word_at(self, index: int) -> str | None:
        """Return the word at ``index``, or None when it is out of range."""
        if 0 <= index < len(self.words):
            return self.words[index]
        return None


def canonical_payload(words: tuple[str, ...]) -> bytes:
    """Return the exact bytes a canonical ``vocabulary.txt`` holds.

    One normalized word per line, ``\\n`` separated, trailing newline included.
    This is also the input the recorded sha256 is taken over, so the two can
    never disagree about what was hashed.
    """
    return "".join(f"{word}\n" for word in words).encode("utf-8")


def read_canonical_vocabulary(path: Path) -> CanonicalVocabulary:
    """Read ``vocabulary.txt`` and require it to already be canonical.

    Raises:
        ArtifactError: the file is missing or unreadable, starts with a BOM, is
            not valid UTF-8, or is not the exact canonical representation of its
            own normalized contents (blank line, duplicate, un-normalized word,
            CRLF, or a missing trailing newline).
    """
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ArtifactError(f"Could not read the artifact vocabulary: {exc}") from exc

    if payload.startswith(_UTF8_BOM):
        raise ArtifactError(
            "Artifact vocabulary must not start with a byte-order mark: a BOM "
            "survives normalization and would merge into the first word."
        )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactError(f"Artifact vocabulary is not valid UTF-8: {exc}") from exc

    words = normalize_vocabulary(text.splitlines())
    if not words:
        raise ArtifactError("Artifact vocabulary contains no usable words.")
    if payload != canonical_payload(words):
        raise ArtifactError(
            "Artifact vocabulary is not in canonical form. It must be one NFKC "
            "normalized, stripped word per line, LF separated, with a trailing "
            "newline, no blank lines, and no duplicates."
        )

    return CanonicalVocabulary(
        words=words,
        sha256=hashlib.sha256(payload).hexdigest(),
        _index={word: index for index, word in enumerate(words)},
    )
