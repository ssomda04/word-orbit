"""Build a reusable full-vocabulary rank table for one answer word."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol


class RankTableError(ValueError):
    """Raised when rank-table inputs or embedding results are unusable."""


class VocabularyLoadError(RankTableError):
    """Raised when a vocabulary file cannot be read as UTF-8 text."""


class NonFiniteEmbeddingError(RankTableError):
    """Raised when a vector or calculated similarity contains NaN or infinity."""


class WordVectorProvider(Protocol):
    """Minimal injection seam implemented by ``FastTextVectorProvider`` and fakes."""

    def vector(self, word: str) -> Sequence[float]:
        """Return a FastText vector, including a subword-composed OOV vector."""
        ...


@dataclass(frozen=True, slots=True)
class RankEntry:
    """One immutable lookup result in an answer-specific rank table."""

    word: str
    rank: int
    similarity: float


@dataclass(frozen=True, slots=True)
class RankTable:
    """A table built once at game creation and reused for every guess lookup."""

    answer: str
    ranked_entries: tuple[RankEntry, ...]
    _entries: Mapping[str, RankEntry] = field(repr=False, compare=False)

    @classmethod
    def create(cls, answer: str, ranked_entries: tuple[RankEntry, ...]) -> RankTable:
        by_word = {entry.word: entry for entry in ranked_entries}
        return cls(
            answer=answer,
            ranked_entries=ranked_entries,
            _entries=MappingProxyType(by_word),
        )

    @property
    def entries(self) -> Mapping[str, RankEntry]:
        """Expose a read-only ``word -> RankEntry`` mapping."""
        return self._entries

    def lookup(self, word: str) -> RankEntry | None:
        """Return the cached entry for a normalized guess, without model work."""
        normalized = normalize_word(word)
        return self._entries.get(normalized) if normalized else None

    def rank_of(self, word: str) -> int | None:
        """Return only the cached rank expected by the eventual game API."""
        entry = self.lookup(word)
        return entry.rank if entry is not None else None

    def similarity_of(self, word: str) -> float | None:
        """Return the cached answer similarity without another vector lookup."""
        entry = self.lookup(word)
        return entry.similarity if entry is not None else None

    def __len__(self) -> int:
        return len(self.ranked_entries)


def normalize_word(word: str) -> str:
    """Apply the vocabulary policy's Unicode NFKC normalization and trim."""
    if not isinstance(word, str):
        raise RankTableError("Vocabulary entries and the answer must be strings.")
    return unicodedata.normalize("NFKC", word).strip()


def normalize_vocabulary(words: Iterable[str]) -> tuple[str, ...]:
    """Normalize words, drop blanks, and preserve the first unique occurrence."""
    unique: dict[str, None] = {}
    for word in words:
        normalized = normalize_word(word)
        if normalized:
            unique.setdefault(normalized, None)
    return tuple(unique)


def load_vocabulary(path: Path) -> tuple[str, ...]:
    """Load and normalize one UTF-8 vocabulary word per line."""
    try:
        with path.open(encoding="utf-8-sig") as vocabulary_file:
            return normalize_vocabulary(vocabulary_file)
    except FileNotFoundError as exc:
        raise VocabularyLoadError(f"Vocabulary file not found: {path}") from exc
    except UnicodeDecodeError as exc:
        raise VocabularyLoadError(f"Vocabulary file is not valid UTF-8: {path}") from exc
    except OSError as exc:
        raise VocabularyLoadError(f"Could not read vocabulary file {path}: {exc}") from exc


def _load_vector(model: WordVectorProvider, word: str) -> tuple[float, ...]:
    try:
        vector = tuple(float(value) for value in model.vector(word))
    except Exception as exc:
        raise RankTableError(f"Embedding model could not produce a vector: {exc}") from exc
    if not vector:
        raise RankTableError("Embedding model returned an empty vector.")
    if any(not math.isfinite(value) for value in vector):
        raise NonFiniteEmbeddingError("Embedding vector contains NaN or infinity.")
    return vector


def _zero_safe_cosine(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise RankTableError(
            f"Embedding vector dimensions differ: first={len(first)}, second={len(second)}."
        )

    first_norm = math.hypot(*first)
    second_norm = math.hypot(*second)
    if not math.isfinite(first_norm) or not math.isfinite(second_norm):
        raise NonFiniteEmbeddingError("Embedding norm is NaN or infinity.")
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0

    similarity = sum(
        (left / first_norm) * (right / second_norm)
        for left, right in zip(first, second, strict=True)
    )
    if not math.isfinite(similarity):
        raise NonFiniteEmbeddingError("Cosine similarity is NaN or infinity.")
    return max(-1.0, min(1.0, similarity))


def build_rank_table(
    answer: str,
    vocabulary: Iterable[str],
    model: WordVectorProvider,
) -> RankTable:
    """Calculate one answer's full-vocabulary ranks exactly once.

    The answer is forced to rank 1. Remaining words are ordered by descending
    cosine similarity and then ascending normalized word for exact ties.
    """
    normalized_answer = normalize_word(answer)
    if not normalized_answer:
        raise RankTableError("Answer must not be empty after normalization.")

    words = list(normalize_vocabulary(vocabulary))
    if normalized_answer not in words:
        words.append(normalized_answer)

    answer_vector = _load_vector(model, normalized_answer)
    similarities: dict[str, float] = {}
    for word in words:
        vector = answer_vector if word == normalized_answer else _load_vector(model, word)
        similarities[word] = _zero_safe_cosine(answer_vector, vector)

    ranked_words = [normalized_answer]
    ranked_words.extend(
        sorted(
            (word for word in words if word != normalized_answer),
            key=lambda word: (-similarities[word], word),
        )
    )
    entries = tuple(
        RankEntry(word=word, rank=rank, similarity=similarities[word])
        for rank, word in enumerate(ranked_words, start=1)
    )
    return RankTable.create(normalized_answer, entries)


def build_rank_table_from_file(
    answer: str,
    vocabulary_path: Path,
    model: WordVectorProvider,
) -> RankTable:
    """Load a vocabulary file and build the reusable answer-specific table."""
    return build_rank_table(answer, load_vocabulary(vocabulary_path), model)
