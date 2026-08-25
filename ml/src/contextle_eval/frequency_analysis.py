"""Join real corpus frequency data to answer candidates and analyze coverage."""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import random
import statistics
import unicodedata
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

DEFAULT_SAMPLE_SEED = 20260823
DEFAULT_SAMPLE_SIZE = 100
VALUE_COLUMNS = frozenset({"count", "frequency"})
DUPLICATE_POLICIES = frozenset({"sum", "error"})
OUTPUT_FREQUENCY_FIELDS = (
    "frequency",
    "frequency_rank",
    "frequency_found",
    "document_frequency",
    "source",
    "source_frequency_rank",
)
REVIEW_REASONS = (
    "technical_term",
    "historical_term",
    "archaic",
    "long_word",
    "dialect",
    "explicit_rare_label",
)
TOP_PERCENTILES = (1, 5, 10, 25, 50)


class FrequencyAnalysisError(RuntimeError):
    """Raised when frequency or candidate input is invalid or cannot be joined."""


@dataclass
class FrequencyRecord:
    """One normalized, optionally aggregated corpus-frequency record."""

    frequency: Decimal
    document_frequency: Decimal | None = None
    source_frequency_rank: int | None = None
    sources: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class FrequencyInputStats:
    """Validation and aggregation counts for one frequency input file."""

    rows_seen: int
    usable_rows: int
    normalized_words: int
    duplicate_rows: int
    skipped_blank_words: int
    value_column: str

    def as_dict(self) -> dict[str, int | str]:
        return {
            "rows_seen": self.rows_seen,
            "usable_rows": self.usable_rows,
            "normalized_words": self.normalized_words,
            "duplicate_rows": self.duplicate_rows,
            "skipped_blank_words": self.skipped_blank_words,
            "value_column": self.value_column,
        }


@dataclass(frozen=True, slots=True)
class FrequencyAnalysis:
    """Serializable coverage, distribution, group, and sample analysis."""

    candidate_count: int
    matched_count: int
    unmatched_count: int
    coverage_ratio: float
    frequency_min: Decimal | None
    frequency_max: Decimal | None
    frequency_median: Decimal | None
    percentile_cutoffs: dict[str, dict[str, int | str]]
    review_relationship: dict[str, dict[str, int | float | str | None]]
    review_reason_stats: dict[str, dict[str, int | float | str | None]]
    top_sample: tuple[dict[str, str], ...]
    middle_sample: tuple[dict[str, str], ...]
    low_sample: tuple[dict[str, str], ...]
    unmatched_sample: tuple[dict[str, str], ...]
    sample_seed: int
    input_stats: FrequencyInputStats

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "frequency_matched": self.matched_count,
            "frequency_unmatched": self.unmatched_count,
            "coverage_ratio": self.coverage_ratio,
            "frequency_min": _optional_decimal(self.frequency_min),
            "frequency_max": _optional_decimal(self.frequency_max),
            "frequency_median": _optional_decimal(self.frequency_median),
            "frequency_rank_percentiles": self.percentile_cutoffs,
            "top_sample": list(self.top_sample),
            "middle_sample": list(self.middle_sample),
            "low_sample": list(self.low_sample),
            "unmatched_sample": list(self.unmatched_sample),
            "review_required_relationship": self.review_relationship,
            "review_reason_frequency": self.review_reason_stats,
            "sample_seed": self.sample_seed,
            "frequency_input": self.input_stats.as_dict(),
        }


def normalize_frequency_word(word: str) -> str:
    """Apply the shared NFKC-and-strip join policy."""
    return unicodedata.normalize("NFKC", word).strip()


def _parse_decimal(raw: str, *, field_name: str, row_number: int) -> Decimal:
    try:
        value = Decimal(raw.strip())
    except (InvalidOperation, AttributeError) as exc:
        raise FrequencyAnalysisError(
            f"Frequency row {row_number} has invalid {field_name}: {raw!r}."
        ) from exc
    if not value.is_finite() or value < 0:
        raise FrequencyAnalysisError(
            f"Frequency row {row_number} requires finite non-negative {field_name}; got {raw!r}."
        )
    return value


def _parse_optional_document_frequency(raw: str, row_number: int) -> Decimal | None:
    if not raw.strip():
        return None
    return _parse_decimal(raw, field_name="document_frequency", row_number=row_number)


def _parse_optional_rank(raw: str, row_number: int) -> int | None:
    if not raw.strip():
        return None
    value = _parse_decimal(raw, field_name="frequency_rank", row_number=row_number)
    if value < 1 or value != value.to_integral_value():
        raise FrequencyAnalysisError(
            f"Frequency row {row_number} requires a positive integer frequency_rank; got {raw!r}."
        )
    return int(value)


def _select_value_column(fieldnames: list[str], requested: str | None) -> str:
    if requested is not None:
        if requested not in VALUE_COLUMNS:
            raise FrequencyAnalysisError(
                f"Unknown value column {requested!r}; expected one of {sorted(VALUE_COLUMNS)}."
            )
        if requested not in fieldnames:
            raise FrequencyAnalysisError(f"Frequency CSV has no {requested!r} column.")
        return requested

    available = sorted(VALUE_COLUMNS & set(fieldnames))
    if len(available) == 1:
        return available[0]
    if not available:
        raise FrequencyAnalysisError(
            "Frequency CSV requires a 'count' or 'frequency' column."
        )
    raise FrequencyAnalysisError(
        "Frequency CSV contains both 'count' and 'frequency'; select one with --value-column."
    )


def load_frequency_csv(
    path: Path,
    *,
    value_column: str | None = None,
    duplicate_policy: str = "sum",
) -> tuple[dict[str, FrequencyRecord], FrequencyInputStats]:
    """Validate and aggregate a generic UTF-8 corpus-frequency CSV."""
    if duplicate_policy not in DUPLICATE_POLICIES:
        raise FrequencyAnalysisError(
            f"Unknown duplicate policy {duplicate_policy!r}; expected one of {sorted(DUPLICATE_POLICIES)}."
        )
    try:
        frequency_file = path.open(encoding="utf-8-sig", newline="")
    except FileNotFoundError as exc:
        raise FrequencyAnalysisError(f"Frequency file not found: {path}") from exc
    except OSError as exc:
        raise FrequencyAnalysisError(
            f"Could not read frequency file {path}: {exc}"
        ) from exc

    records: dict[str, FrequencyRecord] = {}
    rows_seen = 0
    usable_rows = 0
    duplicate_rows = 0
    skipped_blank_words = 0
    try:
        with frequency_file:
            reader = csv.DictReader(frequency_file)
            if reader.fieldnames is None:
                raise FrequencyAnalysisError("Frequency CSV is empty or has no header.")
            fieldnames = [field.strip() for field in reader.fieldnames]
            if "word" not in fieldnames:
                raise FrequencyAnalysisError("Frequency CSV requires a 'word' column.")
            selected_value_column = _select_value_column(fieldnames, value_column)

            for row_number, row in enumerate(reader, start=2):
                rows_seen += 1
                word = normalize_frequency_word(row.get("word") or "")
                if not word:
                    skipped_blank_words += 1
                    continue
                raw_value = row.get(selected_value_column) or ""
                if not raw_value.strip():
                    raise FrequencyAnalysisError(
                        f"Frequency row {row_number} has an empty {selected_value_column}."
                    )
                frequency = _parse_decimal(
                    raw_value, field_name=selected_value_column, row_number=row_number
                )
                document_frequency = _parse_optional_document_frequency(
                    row.get("document_frequency") or "", row_number
                )
                source_rank = _parse_optional_rank(
                    row.get("frequency_rank") or "", row_number
                )
                source = (row.get("source") or "").strip()
                usable_rows += 1

                existing = records.get(word)
                if existing is None:
                    records[word] = FrequencyRecord(
                        frequency=frequency,
                        document_frequency=document_frequency,
                        source_frequency_rank=source_rank,
                        sources={source} if source else set(),
                    )
                    continue
                duplicate_rows += 1
                if duplicate_policy == "error":
                    raise FrequencyAnalysisError(
                        f"Duplicate normalized frequency word {word!r} at row {row_number}."
                    )
                existing.frequency += frequency
                if document_frequency is not None:
                    existing.document_frequency = (
                        document_frequency
                        if existing.document_frequency is None
                        else existing.document_frequency + document_frequency
                    )
                if source_rank is not None:
                    existing.source_frequency_rank = (
                        source_rank
                        if existing.source_frequency_rank is None
                        else min(existing.source_frequency_rank, source_rank)
                    )
                if source:
                    existing.sources.add(source)
    except csv.Error as exc:
        raise FrequencyAnalysisError(
            f"Could not parse frequency CSV {path}: {exc}"
        ) from exc

    if not records:
        raise FrequencyAnalysisError(
            "Frequency CSV contains no usable non-empty word rows."
        )
    return records, FrequencyInputStats(
        rows_seen=rows_seen,
        usable_rows=usable_rows,
        normalized_words=len(records),
        duplicate_rows=duplicate_rows,
        skipped_blank_words=skipped_blank_words,
        value_column=selected_value_column,
    )


def _load_candidates(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        candidate_file = path.open(encoding="utf-8-sig", newline="")
    except FileNotFoundError as exc:
        raise FrequencyAnalysisError(f"Candidate file not found: {path}") from exc
    except OSError as exc:
        raise FrequencyAnalysisError(
            f"Could not read candidate file {path}: {exc}"
        ) from exc

    try:
        with candidate_file:
            reader = csv.DictReader(candidate_file)
            if reader.fieldnames is None or "word" not in reader.fieldnames:
                raise FrequencyAnalysisError("Candidate CSV requires a 'word' column.")
            fieldnames = list(reader.fieldnames)
            collisions = set(fieldnames) & set(OUTPUT_FREQUENCY_FIELDS)
            if collisions:
                raise FrequencyAnalysisError(
                    f"Candidate CSV already contains output frequency columns: {sorted(collisions)}."
                )
            rows: list[dict[str, str]] = []
            seen: set[str] = set()
            for row_number, row in enumerate(reader, start=2):
                word = normalize_frequency_word(row.get("word") or "")
                if not word:
                    continue
                if word in seen:
                    raise FrequencyAnalysisError(
                        f"Candidate CSV has duplicate normalized word {word!r} at row {row_number}."
                    )
                seen.add(word)
                normalized_row = dict(row)
                normalized_row["word"] = word
                rows.append(normalized_row)
    except csv.Error as exc:
        raise FrequencyAnalysisError(
            f"Could not parse candidate CSV {path}: {exc}"
        ) from exc
    if not rows:
        raise FrequencyAnalysisError("Candidate CSV contains no usable word rows.")
    return fieldnames, rows


def _format_decimal(value: Decimal) -> str:
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


def _optional_decimal(value: Decimal | None) -> str | None:
    return _format_decimal(value) if value is not None else None


def _competition_ranks(
    candidate_rows: list[dict[str, str]], records: dict[str, FrequencyRecord]
) -> tuple[list[str], dict[str, int]]:
    matched_words = [row["word"] for row in candidate_rows if row["word"] in records]
    ranked_words = sorted(
        matched_words, key=lambda word: (-records[word].frequency, word)
    )
    ranks: dict[str, int] = {}
    previous_frequency: Decimal | None = None
    previous_rank = 0
    for position, word in enumerate(ranked_words, start=1):
        frequency = records[word].frequency
        if previous_frequency is None or frequency != previous_frequency:
            previous_rank = position
            previous_frequency = frequency
        ranks[word] = previous_rank
    return ranked_words, ranks


def _sample_view(row: dict[str, str]) -> dict[str, str]:
    return {
        key: row.get(key, "")
        for key in (
            "word",
            "frequency",
            "frequency_rank",
            "review_required",
            "review_reason",
        )
    }


def _sample_rows(
    rows: list[dict[str, str]], *, seed: int, sample_size: int
) -> tuple[dict[str, str], ...]:
    if not rows or sample_size <= 0:
        return ()
    return tuple(
        _sample_view(row)
        for row in random.Random(seed).sample(rows, min(sample_size, len(rows)))
    )


def _group_stats(
    rows: list[dict[str, str]], matched_count: int
) -> dict[str, int | float | str | None]:
    matched = [row for row in rows if row["frequency_found"] == "true"]
    frequencies = [Decimal(row["frequency"]) for row in matched]
    ranks = [int(row["frequency_rank"]) for row in matched]
    return {
        "candidate_count": len(rows),
        "matched_count": len(matched),
        "unmatched_count": len(rows) - len(matched),
        "coverage_ratio": len(matched) / len(rows) if rows else 0.0,
        "frequency_min": _optional_decimal(min(frequencies) if frequencies else None),
        "frequency_max": _optional_decimal(max(frequencies) if frequencies else None),
        "frequency_median": _optional_decimal(
            statistics.median(frequencies) if frequencies else None
        ),
        "median_frequency_rank": statistics.median(ranks) if ranks else None,
        "median_rank_percentile": (
            statistics.median(ranks) * 100 / matched_count
            if ranks and matched_count
            else None
        ),
    }


def _build_analysis(
    joined_rows: list[dict[str, str]],
    ranked_rows: list[dict[str, str]],
    input_stats: FrequencyInputStats,
    *,
    seed: int,
    sample_size: int,
) -> FrequencyAnalysis:
    matched_count = len(ranked_rows)
    unmatched_rows = [row for row in joined_rows if row["frequency_found"] == "false"]
    frequencies = [Decimal(row["frequency"]) for row in ranked_rows]
    percentile_cutoffs: dict[str, dict[str, int | str]] = {}
    for percentile in TOP_PERCENTILES:
        if not ranked_rows:
            continue
        top_count = max(1, math.ceil(matched_count * percentile / 100))
        boundary = ranked_rows[top_count - 1]
        percentile_cutoffs[f"top_{percentile}_percent"] = {
            "candidate_count": top_count,
            "boundary_rank": int(boundary["frequency_rank"]),
            "minimum_frequency": boundary["frequency"],
        }

    middle_start = math.floor(matched_count * 0.45)
    middle_end = math.ceil(matched_count * 0.55)
    low_start = math.floor(matched_count * 0.90)
    reviewed = [
        row for row in joined_rows if row.get("review_required", "").lower() == "true"
    ]
    not_reviewed = [
        row for row in joined_rows if row.get("review_required", "").lower() != "true"
    ]
    review_relationship = {
        "review_required": _group_stats(reviewed, matched_count),
        "not_review_required": _group_stats(not_reviewed, matched_count),
    }
    reason_stats = {
        reason: _group_stats(
            [
                row
                for row in joined_rows
                if reason in row.get("review_reason", "").split("|")
            ],
            matched_count,
        )
        for reason in REVIEW_REASONS
    }
    return FrequencyAnalysis(
        candidate_count=len(joined_rows),
        matched_count=matched_count,
        unmatched_count=len(unmatched_rows),
        coverage_ratio=matched_count / len(joined_rows),
        frequency_min=min(frequencies) if frequencies else None,
        frequency_max=max(frequencies) if frequencies else None,
        frequency_median=statistics.median(frequencies) if frequencies else None,
        percentile_cutoffs=percentile_cutoffs,
        review_relationship=review_relationship,
        review_reason_stats=reason_stats,
        top_sample=tuple(_sample_view(row) for row in ranked_rows[:sample_size]),
        middle_sample=_sample_rows(
            ranked_rows[middle_start:middle_end], seed=seed + 1, sample_size=sample_size
        ),
        low_sample=_sample_rows(
            ranked_rows[low_start:], seed=seed + 2, sample_size=sample_size
        ),
        unmatched_sample=_sample_rows(
            unmatched_rows, seed=seed + 3, sample_size=sample_size
        ),
        sample_seed=seed,
        input_stats=input_stats,
    )


def join_answer_candidate_frequency(
    candidate_path: Path,
    frequency_path: Path,
    output_path: Path,
    *,
    value_column: str | None = None,
    duplicate_policy: str = "sum",
    seed: int = DEFAULT_SAMPLE_SEED,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
) -> FrequencyAnalysis:
    """Join frequency data, write a review CSV, and return analysis statistics."""
    if sample_size < 0:
        raise FrequencyAnalysisError("Sample size must be non-negative.")
    fieldnames, candidates = _load_candidates(candidate_path)
    records, input_stats = load_frequency_csv(
        frequency_path,
        value_column=value_column,
        duplicate_policy=duplicate_policy,
    )
    ranked_words, ranks = _competition_ranks(candidates, records)
    joined_rows: list[dict[str, str]] = []
    rows_by_word: dict[str, dict[str, str]] = {}
    for candidate in candidates:
        word = candidate["word"]
        record = records.get(word)
        row = dict(candidate)
        if record is None:
            row.update({field: "" for field in OUTPUT_FREQUENCY_FIELDS})
            row["frequency_found"] = "false"
        else:
            row.update(
                {
                    "frequency": _format_decimal(record.frequency),
                    "frequency_rank": str(ranks[word]),
                    "frequency_found": "true",
                    "document_frequency": _optional_decimal(record.document_frequency)
                    or "",
                    "source": "|".join(sorted(record.sources)),
                    "source_frequency_rank": (
                        str(record.source_frequency_rank)
                        if record.source_frequency_rank is not None
                        else ""
                    ),
                }
            )
        joined_rows.append(row)
        rows_by_word[word] = row
    ranked_rows = [rows_by_word[word] for word in ranked_words]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=[*fieldnames, *OUTPUT_FREQUENCY_FIELDS],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(joined_rows)
        temporary_path.replace(output_path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise FrequencyAnalysisError(
            f"Could not write frequency join CSV {output_path}: {exc}"
        ) from exc

    return _build_analysis(
        joined_rows,
        ranked_rows,
        input_stats,
        seed=seed,
        sample_size=sample_size,
    )


def format_frequency_analysis(analysis: FrequencyAnalysis) -> str:
    """Render the complete analysis as deterministic UTF-8 JSON."""
    return json.dumps(analysis.as_dict(), ensure_ascii=False, indent=2)
