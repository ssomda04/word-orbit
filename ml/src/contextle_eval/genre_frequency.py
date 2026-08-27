"""Load and compare genre frequencies without choosing a final score policy."""

from __future__ import annotations

import csv
import json
import math
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

Genre = Literal["newspaper", "dialogue", "online"]
GENRES: tuple[Genre, ...] = ("newspaper", "dialogue", "online")
PRODUCTION_FREQUENCY_FIELDS = ("canonical_form", "count")
PRODUCTION_AGGREGATED_POS = "AGGREGATED"
PRODUCTION_REPORT_SCHEMA_VERSION = "1.0"


class GenreFrequencyError(ValueError):
    """Raised when genre frequency inputs violate the common contract."""


@dataclass(frozen=True, slots=True)
class GenreFrequencyRecord:
    """Schema-independent frequency observation for one lexical key and genre."""

    genre: Genre
    canonical_word: str
    pos: str
    raw_count: int
    genre_total_lexical_count: int


@dataclass(frozen=True, slots=True)
class CsvRecordAdapter:
    """Map a not-yet-final production CSV schema to the common record fields."""

    canonical_word_field: str = "canonical_word"
    pos_field: str = "pos"
    raw_count_field: str = "raw_count"
    genre_total_lexical_count_field: str = "genre_total_lexical_count"
    genre_field: str = "genre"
    fixed_genre: Genre | None = None


@dataclass(frozen=True, slots=True)
class NormalizedGenreFrequency:
    """Four independent normalization candidates for an observed record."""

    raw_count: int
    relative_frequency_per_million: float
    log1p_relative_frequency: float
    empirical_percentile: float


@dataclass(frozen=True, slots=True)
class GenreComparison:
    """Per-word/POS genre observations and unweighted percentile summaries."""

    canonical_word: str
    pos: str
    newspaper: NormalizedGenreFrequency | None
    dialogue: NormalizedGenreFrequency | None
    online: NormalizedGenreFrequency | None
    genre_coverage: int
    mean_percentile: float | None
    median_percentile: float | None
    max_percentile: float | None


def _canonicalize(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def _parse_nonnegative_int(value: str, field: str, line_number: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise GenreFrequencyError(
            f"Genre frequency row {line_number} has an invalid {field}."
        ) from exc
    if parsed < 0:
        raise GenreFrequencyError(
            f"Genre frequency row {line_number} has a negative {field}."
        )
    return parsed


def load_genre_frequency_csv(
    path: Path, adapter: CsvRecordAdapter | None = None
) -> tuple[GenreFrequencyRecord, ...]:
    """Adapt a CSV into validated common records; extra columns are allowed."""
    adapter = adapter or CsvRecordAdapter()
    required = {
        adapter.canonical_word_field,
        adapter.pos_field,
        adapter.raw_count_field,
        adapter.genre_total_lexical_count_field,
    }
    if adapter.fixed_genre is None:
        required.add(adapter.genre_field)
    try:
        handle = path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise GenreFrequencyError(f"Could not read genre frequency CSV: {exc}") from exc
    records: list[GenreFrequencyRecord] = []
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise GenreFrequencyError(
                f"Genre frequency CSV is missing fields: {sorted(required - set(reader.fieldnames or ()))!r}."
            )
        for line_number, row in enumerate(reader, start=2):
            genre_value = adapter.fixed_genre or row[adapter.genre_field].strip()
            if genre_value not in GENRES:
                raise GenreFrequencyError(
                    f"Genre frequency row {line_number} has an invalid genre."
                )
            word = _canonicalize(row[adapter.canonical_word_field])
            pos = row[adapter.pos_field].strip()
            if not word or not pos:
                raise GenreFrequencyError(
                    f"Genre frequency row {line_number} has an empty lexical key."
                )
            count = _parse_nonnegative_int(
                row[adapter.raw_count_field], "raw count", line_number
            )
            total = _parse_nonnegative_int(
                row[adapter.genre_total_lexical_count_field],
                "genre total lexical count",
                line_number,
            )
            if total == 0:
                raise GenreFrequencyError(
                    f"Genre frequency row {line_number} has a zero genre total."
                )
            if count > total:
                raise GenreFrequencyError(
                    f"Genre frequency row {line_number} has count greater than total."
                )
            records.append(
                GenreFrequencyRecord(
                    cast(Genre, genre_value), word, pos, count, total
                )
            )
    return validate_records(records)


def load_production_genre_frequency(
    frequency_path: Path, report_path: Path, genre: Genre
) -> tuple[GenreFrequencyRecord, ...]:
    """Adapt one raw runner CSV plus its report-held assignment total."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GenreFrequencyError(
            f"Could not read genre frequency report JSON: {exc}"
        ) from exc
    if not isinstance(report, dict):
        raise GenreFrequencyError("Genre frequency report must be a JSON object.")
    if report.get("schema_version") != PRODUCTION_REPORT_SCHEMA_VERSION:
        raise GenreFrequencyError("Genre frequency report schema is unsupported.")
    if report.get("source") != genre:
        raise GenreFrequencyError("Genre frequency report source does not match.")
    frequency = report.get("frequency")
    if not isinstance(frequency, dict):
        raise GenreFrequencyError("Genre frequency report has no frequency object.")
    total = frequency.get("assignments")
    if not isinstance(total, int) or isinstance(total, bool) or total <= 0:
        raise GenreFrequencyError(
            "Genre frequency report has invalid frequency.assignments."
        )

    try:
        handle = frequency_path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise GenreFrequencyError(f"Could not read genre frequency CSV: {exc}") from exc
    records: list[GenreFrequencyRecord] = []
    seen: set[str] = set()
    reader: csv.DictReader[str] | None = None
    row_context = 2
    try:
        with handle:
            reader = csv.DictReader(handle, strict=True)
            if reader.fieldnames != list(PRODUCTION_FREQUENCY_FIELDS):
                raise GenreFrequencyError("Production frequency CSV header is invalid.")
            for line_number, row in enumerate(reader, start=2):
                row_context = line_number
                word = _canonicalize(row["canonical_form"])
                if not word or word != row["canonical_form"]:
                    raise GenreFrequencyError(
                        f"Genre frequency row {line_number} has a noncanonical form."
                    )
                if word in seen:
                    raise GenreFrequencyError(
                        f"Genre frequency CSV contains duplicate form {word!r}."
                    )
                seen.add(word)
                count = _parse_nonnegative_int(row["count"], "count", line_number)
                records.append(
                    GenreFrequencyRecord(
                        genre=genre,
                        canonical_word=word,
                        pos=PRODUCTION_AGGREGATED_POS,
                        raw_count=count,
                        genre_total_lexical_count=total,
                    )
                )
    except GenreFrequencyError:
        raise
    except (csv.Error, UnicodeError, TypeError, ValueError) as exc:
        if reader is not None:
            row_context = max(row_context, reader.line_num)
        raise GenreFrequencyError(
            f"Production frequency CSV {frequency_path} is malformed "
            f"at or near row {row_context}."
        ) from exc
    if not records:
        raise GenreFrequencyError("Production frequency CSV is empty.")
    if sum(record.raw_count for record in records) != total:
        raise GenreFrequencyError(
            "Production CSV counts do not equal report frequency.assignments."
        )
    return validate_records(records)


def validate_records(
    records: Iterable[GenreFrequencyRecord],
) -> tuple[GenreFrequencyRecord, ...]:
    """Validate uniqueness and one consistent lexical-token total per genre."""
    validated = tuple(records)
    seen: set[tuple[Genre, str, str]] = set()
    totals: dict[Genre, int] = {}
    for record in validated:
        if record.genre not in GENRES:
            raise GenreFrequencyError(f"Unsupported genre: {record.genre!r}.")
        if not record.canonical_word or not record.pos:
            raise GenreFrequencyError("Canonical word and POS must not be empty.")
        if record.raw_count < 0 or record.genre_total_lexical_count <= 0:
            raise GenreFrequencyError("Counts must be nonnegative and totals positive.")
        if record.raw_count > record.genre_total_lexical_count:
            raise GenreFrequencyError("Raw count must not exceed its genre total.")
        key = (record.genre, record.canonical_word, record.pos)
        if key in seen:
            raise GenreFrequencyError(f"Duplicate genre frequency key: {key!r}.")
        seen.add(key)
        prior_total = totals.setdefault(record.genre, record.genre_total_lexical_count)
        if prior_total != record.genre_total_lexical_count:
            raise GenreFrequencyError(
                f"Inconsistent lexical totals for genre {record.genre!r}."
            )
    return validated


def normalize_genre_frequencies(
    records: Iterable[GenreFrequencyRecord],
) -> Mapping[tuple[Genre, str, str], NormalizedGenreFrequency]:
    """Calculate candidates independently, using weak empirical CDF percentiles."""
    validated = validate_records(records)
    count_histograms: dict[Genre, Counter[int]] = defaultdict(Counter)
    for record in validated:
        count_histograms[record.genre][record.raw_count] += 1

    percentiles: dict[Genre, dict[int, float]] = {}
    for genre, histogram in count_histograms.items():
        cumulative = 0
        genre_size = histogram.total()
        percentiles[genre] = {}
        for count, frequency in sorted(histogram.items()):
            cumulative += frequency
            percentiles[genre][count] = cumulative / genre_size

    normalized: dict[tuple[Genre, str, str], NormalizedGenreFrequency] = {}
    for record in validated:
        relative = record.raw_count * 1_000_000 / record.genre_total_lexical_count
        normalized[(record.genre, record.canonical_word, record.pos)] = (
            NormalizedGenreFrequency(
                raw_count=record.raw_count,
                relative_frequency_per_million=relative,
                log1p_relative_frequency=math.log1p(relative),
                empirical_percentile=percentiles[record.genre][record.raw_count],
            )
        )
    return normalized


def compare_genre_frequencies(
    records: Iterable[GenreFrequencyRecord],
    comparison_keys: Iterable[tuple[str, str]] = (),
) -> tuple[GenreComparison, ...]:
    """Build deterministic comparisons, including requested fully absent keys."""
    normalized = normalize_genre_frequencies(records)
    lexical_keys = sorted(
        {(word, pos) for _, word, pos in normalized} | set(comparison_keys)
    )
    comparisons: list[GenreComparison] = []
    for word, pos in lexical_keys:
        if not word or not pos:
            raise GenreFrequencyError("Comparison word and POS must not be empty.")
        values = {
            genre: normalized.get((genre, word, pos)) for genre in GENRES
        }
        percentiles = [
            value.empirical_percentile for value in values.values() if value is not None
        ]
        comparisons.append(
            GenreComparison(
                canonical_word=word,
                pos=pos,
                newspaper=values["newspaper"],
                dialogue=values["dialogue"],
                online=values["online"],
                genre_coverage=len(percentiles),
                mean_percentile=(statistics.fmean(percentiles) if percentiles else None),
                median_percentile=(statistics.median(percentiles) if percentiles else None),
                max_percentile=(max(percentiles) if percentiles else None),
            )
        )
    return tuple(comparisons)
