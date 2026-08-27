"""Build a development-only provisional answer pool from calibrated candidates.

Development-only provisional answer pool derived from FrequencyWords calibration.
Do not treat as the final answer pool.
"""

from __future__ import annotations

import csv
import math
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from contextle_eval.frequency_calibration import CONTENT_POS, canonical_pos_parts
from contextle_eval.lemma_frequency import normalize_word

DEFAULT_PERCENTILE_CUTOFF = 70.0
EXCLUDED_CALIBRATION_STATUSES = frozenset({"mixed_pos", "cross_pos"})
SPECIAL_RISK_WORDS = frozenset(
    {"하다", "되다", "있다", "보다", "않다", "감사하다", "진정하다", "달다"}
)
REQUIRED_FIELDS = frozenset(
    {
        "word",
        "pos",
        "frequency_found",
        "frequency_percentile",
        "frequency_calibration_status",
        "manual_review_required",
        "high_count_risk",
    }
)


class ProvisionalAnswerPoolError(RuntimeError):
    """Raised when calibrated candidates cannot produce a valid pool."""


@dataclass(frozen=True)
class ProvisionalPoolEntry:
    """A retained word and the calibrated metadata needed for validation."""

    word: str
    pos: str
    frequency_percentile: float


def load_calibrated_candidates(path: Path) -> list[dict[str, str]]:
    """Load an existing calibrated CSV without recalculating its fields."""
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ProvisionalAnswerPoolError(f"Could not read calibrated CSV {path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not REQUIRED_FIELDS <= set(reader.fieldnames):
            raise ProvisionalAnswerPoolError(
                f"Calibrated CSV requires fields {sorted(REQUIRED_FIELDS)}."
            )
        return [dict(row) for row in reader]


def is_content_word_pos(raw_pos: str) -> bool:
    """Match the calibration report's content-word-only scenario."""
    parts = canonical_pos_parts(raw_pos)
    return bool(parts) and parts <= CONTENT_POS


def select_provisional_entries(
    rows: Sequence[dict[str, str]],
    *,
    percentile_cutoff: float = DEFAULT_PERCENTILE_CUTOFF,
) -> list[ProvisionalPoolEntry]:
    """Select safe P70 content words using only persisted calibration fields."""
    retained: list[ProvisionalPoolEntry] = []
    for row_number, row in enumerate(rows, start=2):
        if row.get("frequency_found", "").lower() != "true":
            continue
        try:
            percentile = float(row.get("frequency_percentile", ""))
        except ValueError as exc:
            raise ProvisionalAnswerPoolError(
                f"Calibrated row {row_number} has an invalid frequency percentile."
            ) from exc
        if not math.isfinite(percentile):
            raise ProvisionalAnswerPoolError(
                f"Calibrated row {row_number} has a non-finite frequency percentile."
            )
        word = normalize_word(row.get("word", ""))
        if (
            percentile < percentile_cutoff
            or not is_content_word_pos(row.get("pos", ""))
            or row.get("frequency_calibration_status") in EXCLUDED_CALIBRATION_STATUSES
            or row.get("manual_review_required", "").lower() == "true"
            or row.get("high_count_risk", "").lower() == "true"
            or word in SPECIAL_RISK_WORDS
        ):
            continue
        if not word:
            raise ProvisionalAnswerPoolError(f"Calibrated row {row_number} has a blank word.")
        retained.append(
            ProvisionalPoolEntry(
                word=word,
                pos=row.get("pos", ""),
                frequency_percentile=percentile,
            )
        )

    words = [entry.word for entry in retained]
    duplicates = sorted(word for word, count in Counter(words).items() if count > 1)
    if duplicates:
        raise ProvisionalAnswerPoolError(
            f"Provisional selection contains duplicate words: {duplicates[:10]}"
        )
    return sorted(retained, key=lambda entry: entry.word)


def validate_reference_membership(
    entries: Sequence[ProvisionalPoolEntry],
    *,
    game_words: Iterable[str],
    answer_candidate_words: Iterable[str],
) -> None:
    """Ensure every retained word exists in both upstream vocabularies."""
    game_set = {normalize_word(word) for word in game_words if normalize_word(word)}
    candidate_set = {
        normalize_word(word) for word in answer_candidate_words if normalize_word(word)
    }
    words = {entry.word for entry in entries}
    missing_game = sorted(words - game_set)
    missing_candidates = sorted(words - candidate_set)
    if missing_game or missing_candidates:
        raise ProvisionalAnswerPoolError(
            "Reference membership failed: "
            f"missing from game_words={missing_game[:10]}, "
            f"missing from answer_candidates={missing_candidates[:10]}"
        )


def write_provisional_pool(path: Path, entries: Sequence[ProvisionalPoolEntry]) -> None:
    """Write normalized words, one per line, in deterministic lexical order."""
    for entry in entries:
        if entry.word != unicodedata.normalize("NFKC", entry.word):
            raise ProvisionalAnswerPoolError(f"Word is not Unicode NFKC: {entry.word!r}")
        if "\n" in entry.word or "\r" in entry.word:
            raise ProvisionalAnswerPoolError(f"Word contains a line break: {entry.word!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{entry.word}\n" for entry in entries).encode("utf-8")
    path.write_bytes(payload)
