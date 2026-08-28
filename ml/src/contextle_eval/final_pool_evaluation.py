"""Prepare auditable final-pool candidate evaluations without selecting a pool."""

from __future__ import annotations

import csv
import math
import os
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CandidateStatus = Literal["eligible", "review_required", "excluded"]
GenreMatchType = Literal["exact", "aggregated", "none"]
GenreName = Literal["newspaper", "dialogue", "online"]
GENRES: tuple[GenreName, ...] = ("newspaper", "dialogue", "online")

CANDIDATE_REQUIRED_FIELDS = frozenset(
    {
        "word",
        "pos",
        "is_proper_noun",
        "is_general_lexical_pos",
        "is_archaic",
        "is_dialect",
        "is_historical",
        "is_technical",
        "wiktionary_labels",
        "domain_labels",
        "review_required",
        "review_reason",
        "selected_frequency",
        "selected_frequency_source",
        "frequency_found",
        "frequency_percentile",
        "frequency_calibration_status",
        "manual_review_required",
        "high_count_risk",
        "hybrid_risk_flags",
    }
)
AUDIT_FIELDS = (
    "canonical_word",
    "pos",
    "candidate_status",
    "reasons",
    "source_review_reasons",
    "available_evidence",
    "genre_coverage",
    "observed_genres",
    "genre_evidence_pos",
    "genre_match_type",
    "manual_review_required",
    "in_provisional_pool_baseline",
)


class FinalPoolEvaluationError(ValueError):
    """Raised when candidate or optional genre evidence is invalid."""


@dataclass(frozen=True, slots=True)
class AnswerCandidateMetadata:
    """Existing answer-candidate metadata retained for audit."""

    is_proper_noun: bool
    is_general_lexical_pos: bool
    is_archaic: bool
    is_dialect: bool
    is_historical: bool
    is_technical: bool
    wiktionary_labels: tuple[str, ...]
    domain_labels: tuple[str, ...]
    review_required: bool
    review_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrequencyEvidence:
    """Persisted pre-genre frequency evidence; no cutoff is applied here."""

    found: bool
    selected_frequency: int | None
    source: str
    percentile: float | None
    calibration_status: str
    high_count_risk: bool
    manual_review_required: bool
    risk_flags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenreComparisonEvidence:
    """Optional, module-independent view of a future genre comparison row."""

    pos: str
    genre_coverage: int
    observed_genres: tuple[GenreName, ...]
    mean_percentile: float | None = None
    median_percentile: float | None = None
    max_percentile: float | None = None


@dataclass(frozen=True, slots=True)
class FinalPoolCandidate:
    """Common candidate record joining existing and optional future evidence."""

    canonical_word: str
    pos: str
    metadata: AnswerCandidateMetadata
    frequency: FrequencyEvidence
    genre: GenreComparisonEvidence | None = None
    in_provisional_pool_baseline: bool = False
    genre_match_type: GenreMatchType = "none"


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Audit routing result, not a final answer-pool selection."""

    candidate: FinalPoolCandidate
    status: CandidateStatus
    reasons: tuple[str, ...]
    available_evidence: tuple[str, ...]
    manual_review_required: bool

    def as_audit_row(self) -> dict[str, str | int]:
        genre = self.candidate.genre
        return {
            "canonical_word": self.candidate.canonical_word,
            "pos": self.candidate.pos,
            "candidate_status": self.status,
            "reasons": "|".join(self.reasons),
            "source_review_reasons": "|".join(
                self.candidate.metadata.review_reasons
            ),
            "available_evidence": "|".join(self.available_evidence),
            "genre_coverage": "" if genre is None else genre.genre_coverage,
            "observed_genres": (
                "" if genre is None else "|".join(genre.observed_genres)
            ),
            "genre_evidence_pos": "" if genre is None else genre.pos,
            "genre_match_type": self.candidate.genre_match_type,
            "manual_review_required": _format_bool(self.manual_review_required),
            "in_provisional_pool_baseline": _format_bool(
                self.candidate.in_provisional_pool_baseline
            ),
        }


@dataclass(frozen=True, slots=True)
class GenreComparisonCsvAdapter:
    """Map a future comparison CSV without importing its implementation module."""

    word_field: str = "canonical_word"
    pos_field: str = "pos"
    coverage_field: str = "genre_coverage"
    mean_percentile_field: str = "mean_percentile"
    median_percentile_field: str = "median_percentile"
    max_percentile_field: str = "max_percentile"


def _normalize_word(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def _split_pipe(value: str) -> tuple[str, ...]:
    return tuple(sorted({part.strip() for part in value.split("|") if part.strip()}))


def _parse_bool(value: str, *, field: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise FinalPoolEvaluationError(
            f"Candidate row {row_number} has invalid boolean {field}."
        )
    return normalized == "true"


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


def _parse_optional_int(value: str, *, field: str, row_number: int) -> int | None:
    if not value.strip():
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise FinalPoolEvaluationError(
            f"Candidate row {row_number} has invalid {field}."
        ) from exc
    if parsed < 0:
        raise FinalPoolEvaluationError(
            f"Candidate row {row_number} has negative {field}."
        )
    return parsed


def _parse_optional_float(value: str, *, field: str, row_number: int) -> float | None:
    if not value.strip():
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise FinalPoolEvaluationError(
            f"Candidate row {row_number} has invalid {field}."
        ) from exc
    if not math.isfinite(parsed):
        raise FinalPoolEvaluationError(
            f"Candidate row {row_number} has non-finite {field}."
        )
    return parsed


def _require_range(
    value: float | None,
    *,
    field: str,
    row_number: int,
    minimum: float,
    maximum: float,
) -> float | None:
    if value is not None and not minimum <= value <= maximum:
        raise FinalPoolEvaluationError(
            f"Row {row_number} has {field} outside {minimum}..{maximum}."
        )
    return value


def evaluate_candidate(candidate: FinalPoolCandidate) -> CandidateEvaluation:
    """Route established risks for audit without applying a genre threshold."""
    reasons: set[str] = set()
    metadata = candidate.metadata
    frequency = candidate.frequency
    if metadata.is_proper_noun:
        reasons.add("proper_noun")
    if frequency.calibration_status == "mixed_pos":
        reasons.add("mixed_pos")
    if frequency.calibration_status == "cross_pos":
        reasons.add("cross_pos")
    if frequency.high_count_risk or frequency.calibration_status == "high_count_risk":
        reasons.add("high_count_risk")
    if "derivational_review" in frequency.risk_flags:
        reasons.add("derivational_review")
    if not frequency.found or (
        candidate.genre is not None and candidate.genre.genre_coverage == 0
    ):
        reasons.add("insufficient_frequency_evidence")
    if metadata.review_required or frequency.manual_review_required:
        reasons.add("manual_review")

    ordered_reasons = tuple(sorted(reasons))
    if "proper_noun" in reasons:
        status: CandidateStatus = "excluded"
    elif reasons:
        status = "review_required"
    else:
        status = "eligible"
    available = ["answer_candidate_metadata"]
    if frequency.found:
        available.append("frequency_evidence")
    if candidate.genre is not None:
        available.append("genre_comparison_evidence")
    if candidate.in_provisional_pool_baseline:
        available.append("provisional_pool_baseline")
    return CandidateEvaluation(
        candidate=candidate,
        status=status,
        reasons=ordered_reasons,
        available_evidence=tuple(available),
        manual_review_required=status == "review_required",
    )


def evaluate_candidates(
    candidates: Iterable[FinalPoolCandidate],
) -> tuple[CandidateEvaluation, ...]:
    """Evaluate candidates in deterministic word/POS order."""
    values = tuple(candidates)
    keys = [(candidate.canonical_word, candidate.pos) for candidate in values]
    if len(keys) != len(set(keys)):
        raise FinalPoolEvaluationError("Candidate input contains duplicate word/POS keys.")
    return tuple(
        evaluate_candidate(candidate)
        for candidate in sorted(values, key=lambda item: (item.canonical_word, item.pos))
    )


def load_genre_comparison_evidence(
    path: Path, adapter: GenreComparisonCsvAdapter | None = None
) -> Mapping[tuple[str, str], GenreComparisonEvidence]:
    """Load only the stable comparison concepts needed by candidate evaluation."""
    adapter = adapter or GenreComparisonCsvAdapter()
    summary_fields = (
        adapter.mean_percentile_field,
        adapter.median_percentile_field,
        adapter.max_percentile_field,
    )
    required = {
        adapter.word_field,
        adapter.pos_field,
        adapter.coverage_field,
        *summary_fields,
        *(f"{genre}_raw" for genre in GENRES),
    }
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise FinalPoolEvaluationError(
            f"Could not read genre comparison CSV {path}: {exc}"
        ) from exc
    evidence: dict[tuple[str, str], GenreComparisonEvidence] = {}
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise FinalPoolEvaluationError(
                f"Genre comparison CSV requires fields {sorted(required)}."
            )
        genre_percentile_fields = tuple(
            f"{genre}_percentile"
            for genre in GENRES
            if f"{genre}_percentile" in reader.fieldnames
        )
        for row_number, row in enumerate(reader, start=2):
            word = _normalize_word(row[adapter.word_field])
            pos = row[adapter.pos_field].strip()
            key = (word, pos)
            if not word or not pos or key in evidence:
                raise FinalPoolEvaluationError(
                    f"Genre comparison row {row_number} has an invalid key."
                )
            coverage = _parse_optional_int(
                row[adapter.coverage_field], field="genre coverage", row_number=row_number
            )
            if coverage is None or coverage > len(GENRES):
                raise FinalPoolEvaluationError(
                    f"Genre comparison row {row_number} has invalid coverage."
                )
            observed = tuple(
                genre for genre in GENRES if row[f"{genre}_raw"].strip()
            )
            if coverage != len(observed):
                raise FinalPoolEvaluationError(
                    f"Genre comparison row {row_number} has inconsistent coverage."
                )
            for field in genre_percentile_fields:
                _require_range(
                    _parse_optional_float(row[field], field=field, row_number=row_number),
                    field=field,
                    row_number=row_number,
                    minimum=0.0,
                    maximum=1.0,
                )
            evidence[key] = GenreComparisonEvidence(
                pos=pos,
                genre_coverage=coverage,
                observed_genres=observed,
                mean_percentile=_require_range(
                    _parse_optional_float(
                        row[summary_fields[0]],
                        field=summary_fields[0],
                        row_number=row_number,
                    ),
                    field=summary_fields[0],
                    row_number=row_number,
                    minimum=0.0,
                    maximum=1.0,
                ),
                median_percentile=_require_range(
                    _parse_optional_float(
                        row[summary_fields[1]],
                        field=summary_fields[1],
                        row_number=row_number,
                    ),
                    field=summary_fields[1],
                    row_number=row_number,
                    minimum=0.0,
                    maximum=1.0,
                ),
                max_percentile=_require_range(
                    _parse_optional_float(
                        row[summary_fields[2]],
                        field=summary_fields[2],
                        row_number=row_number,
                    ),
                    field=summary_fields[2],
                    row_number=row_number,
                    minimum=0.0,
                    maximum=1.0,
                ),
            )
    return evidence


def load_final_pool_candidates(
    path: Path,
    *,
    genre_evidence: Mapping[tuple[str, str], GenreComparisonEvidence] | None = None,
    provisional_words: Iterable[str] = (),
    aggregated_genre_pos: str = "AGGREGATED",
) -> tuple[FinalPoolCandidate, ...]:
    """Adapt the enriched candidate CSV plus optional, separately loaded evidence."""
    genre_evidence = genre_evidence or {}
    provisional = {_normalize_word(word) for word in provisional_words}
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise FinalPoolEvaluationError(f"Could not read candidate CSV {path}: {exc}") from exc
    candidates: list[FinalPoolCandidate] = []
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not CANDIDATE_REQUIRED_FIELDS <= set(
            reader.fieldnames
        ):
            raise FinalPoolEvaluationError(
                f"Candidate CSV requires fields {sorted(CANDIDATE_REQUIRED_FIELDS)}."
            )
        for row_number, row in enumerate(reader, start=2):
            word = _normalize_word(row["word"])
            pos = row["pos"].strip()
            if not word or not pos:
                raise FinalPoolEvaluationError(
                    f"Candidate row {row_number} has an empty word/POS key."
                )
            found = _parse_bool(
                row["frequency_found"], field="frequency_found", row_number=row_number
            )
            selected = _parse_optional_int(
                row["selected_frequency"],
                field="selected_frequency",
                row_number=row_number,
            )
            percentile = _require_range(
                _parse_optional_float(
                    row["frequency_percentile"],
                    field="frequency_percentile",
                    row_number=row_number,
                ),
                field="frequency_percentile",
                row_number=row_number,
                minimum=0.0,
                maximum=100.0,
            )
            if found != (selected is not None) or found != (percentile is not None):
                raise FinalPoolEvaluationError(
                    f"Candidate row {row_number} has inconsistent frequency evidence."
                )
            metadata = AnswerCandidateMetadata(
                is_proper_noun=_parse_bool(
                    row["is_proper_noun"], field="is_proper_noun", row_number=row_number
                ),
                is_general_lexical_pos=_parse_bool(
                    row["is_general_lexical_pos"],
                    field="is_general_lexical_pos",
                    row_number=row_number,
                ),
                is_archaic=_parse_bool(
                    row["is_archaic"], field="is_archaic", row_number=row_number
                ),
                is_dialect=_parse_bool(
                    row["is_dialect"], field="is_dialect", row_number=row_number
                ),
                is_historical=_parse_bool(
                    row["is_historical"], field="is_historical", row_number=row_number
                ),
                is_technical=_parse_bool(
                    row["is_technical"], field="is_technical", row_number=row_number
                ),
                wiktionary_labels=_split_pipe(row["wiktionary_labels"]),
                domain_labels=_split_pipe(row["domain_labels"]),
                review_required=_parse_bool(
                    row["review_required"], field="review_required", row_number=row_number
                ),
                review_reasons=_split_pipe(row["review_reason"]),
            )
            frequency = FrequencyEvidence(
                found=found,
                selected_frequency=selected,
                source=row["selected_frequency_source"].strip(),
                percentile=percentile,
                calibration_status=row["frequency_calibration_status"].strip(),
                high_count_risk=_parse_bool(
                    row["high_count_risk"], field="high_count_risk", row_number=row_number
                ),
                manual_review_required=_parse_bool(
                    row["manual_review_required"],
                    field="manual_review_required",
                    row_number=row_number,
                ),
                risk_flags=_split_pipe(row["hybrid_risk_flags"]),
            )
            joined_genre = genre_evidence.get((word, pos))
            genre_match_type: GenreMatchType = "exact"
            if joined_genre is None:
                joined_genre = genre_evidence.get((word, aggregated_genre_pos))
                genre_match_type = "aggregated" if joined_genre is not None else "none"
            candidates.append(
                FinalPoolCandidate(
                    canonical_word=word,
                    pos=pos,
                    metadata=metadata,
                    frequency=frequency,
                    genre=joined_genre,
                    in_provisional_pool_baseline=word in provisional,
                    genre_match_type=genre_match_type,
                )
            )
    return tuple(candidates)


def write_candidate_audit(
    path: Path, evaluations: Sequence[CandidateEvaluation]
) -> None:
    """Write metadata-only audit rows, never a final word-list artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(evaluation.as_audit_row() for evaluation in evaluations)
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise FinalPoolEvaluationError(f"Could not write candidate audit {path}: {exc}") from exc
