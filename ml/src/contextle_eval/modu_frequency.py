"""Aggregate raw NIKL morpheme-form/POS frequencies by MP source subtype."""

from __future__ import annotations

import csv
import json
import os
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from contextle_eval.modu_corpus import (
    SOURCE_SUBTYPES,
    ParsedSentence,
    SourceSubtype,
    iter_mp_sentences,
)

REPORT_SCHEMA_VERSION = "1.0"
CSV_FIELDS = ("source_subtype", "morpheme", "pos", "count")
EXCLUDING_ISSUES = frozenset(
    {
        "empty_morpheme",
        "empty_pos",
        "missing_word_id",
        "orphan_word_id",
        "duplicate_mp_id",
    }
)
WARNING_ISSUES = frozenset({"position_order"})
SUPPORTED_ISSUES = EXCLUDING_ISSUES | WARNING_ISSUES
TOP_MORPHEME_POS = ("NNG", "VV", "VA", "MAG")


class ModuFrequencyError(ValueError):
    """Raised when parsed MP records cannot be aggregated without silent loss."""


@dataclass(frozen=True, slots=True)
class FrequencyRow:
    """One deterministic raw morpheme-form/POS frequency row."""

    source_subtype: SourceSubtype
    morpheme: str
    pos: str
    count: int


@dataclass(frozen=True, slots=True)
class SubtypeFrequencyStats:
    """Auditable aggregation totals for one MP source subtype."""

    total_sentences: int
    total_mp_records: int
    counted_mp: int
    excluded_mp: int
    unique_morpheme_pos: int
    pos_counts: Mapping[str, int]
    issue_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class ModuFrequencyResult:
    """Deterministically sorted rows and subtype-specific audit statistics."""

    rows: tuple[FrequencyRow, ...]
    subtypes: Mapping[SourceSubtype, SubtypeFrequencyStats]


@dataclass
class _MutableStats:
    total_sentences: int = 0
    total_mp_records: int = 0
    counted_mp: int = 0
    excluded_mp: int = 0
    issue_counts: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.issue_counts is None:
            self.issue_counts = Counter()


def canonicalize_morpheme(value: str) -> str:
    """Apply only the agreed NFKC+strip policy to a morpheme form."""
    return unicodedata.normalize("NFKC", value).strip()


def aggregate_morpheme_frequencies(
    sentences: Iterable[ParsedSentence],
) -> ModuFrequencyResult:
    """Aggregate streamed sentences while applying explicit issue policy."""
    frequencies: Counter[tuple[SourceSubtype, str, str]] = Counter()
    mutable = {subtype: _MutableStats() for subtype in SOURCE_SUBTYPES}

    for sentence in sentences:
        subtype = sentence.source_subtype
        if subtype not in SOURCE_SUBTYPES:
            raise ModuFrequencyError(f"Unsupported source subtype: {subtype!r}")
        stats = mutable[subtype]
        stats.total_sentences += 1
        stats.total_mp_records += len(sentence.records)
        issues_by_record: dict[int, set[str]] = defaultdict(set)
        for issue in sentence.issues:
            if issue.code not in SUPPORTED_ISSUES:
                raise ModuFrequencyError(
                    f"Unsupported validation issue {issue.code!r} in {sentence.sentence_id}."
                )
            if not 0 <= issue.record_index < len(sentence.records):
                raise ModuFrequencyError(
                    f"Validation issue has invalid record index {issue.record_index} "
                    f"in {sentence.sentence_id}."
                )
            assert stats.issue_counts is not None
            stats.issue_counts[issue.code] += 1
            issues_by_record[issue.record_index].add(issue.code)

        for index, record in enumerate(sentence.records):
            if record.source_subtype != subtype:
                raise ModuFrequencyError(
                    f"Sentence/record subtype mismatch in {sentence.sentence_id}."
                )
            record_issues = issues_by_record[index]
            exclude = bool(record_issues & EXCLUDING_ISSUES)
            morpheme = canonicalize_morpheme(record.morpheme)
            if not morpheme:
                exclude = True
                if "empty_morpheme" not in record_issues:
                    assert stats.issue_counts is not None
                    stats.issue_counts["empty_morpheme"] += 1
            if not record.pos.strip():
                exclude = True
                if "empty_pos" not in record_issues:
                    assert stats.issue_counts is not None
                    stats.issue_counts["empty_pos"] += 1
            if exclude:
                stats.excluded_mp += 1
                continue
            frequencies[(subtype, morpheme, record.pos)] += 1
            stats.counted_mp += 1

    rows = tuple(
        FrequencyRow(subtype, morpheme, pos, count)
        for (subtype, morpheme, pos), count in sorted(
            frequencies.items(),
            key=lambda item: (
                SOURCE_SUBTYPES.index(item[0][0]),
                -item[1],
                item[0][1],
                item[0][2],
            ),
        )
    )
    subtype_stats: dict[SourceSubtype, SubtypeFrequencyStats] = {}
    for subtype in SOURCE_SUBTYPES:
        stats = mutable[subtype]
        subtype_rows = [row for row in rows if row.source_subtype == subtype]
        pos_counts: Counter[str] = Counter()
        for row in subtype_rows:
            pos_counts[row.pos] += row.count
        subtype_stats[subtype] = SubtypeFrequencyStats(
            total_sentences=stats.total_sentences,
            total_mp_records=stats.total_mp_records,
            counted_mp=stats.counted_mp,
            excluded_mp=stats.excluded_mp,
            unique_morpheme_pos=len(subtype_rows),
            pos_counts=dict(sorted(pos_counts.items())),
            issue_counts=dict(sorted((stats.issue_counts or {}).items())),
        )
    return ModuFrequencyResult(rows=rows, subtypes=subtype_stats)


def aggregate_mp_zip(zip_path: Path) -> ModuFrequencyResult:
    """Stream the complete NXMP and SXMP entries and aggregate them separately."""
    sentences = (
        sentence
        for subtype in SOURCE_SUBTYPES
        for sentence in iter_mp_sentences(zip_path, source_subtype=subtype)
    )
    return aggregate_morpheme_frequencies(sentences)


def build_report(result: ModuFrequencyResult) -> dict[str, object]:
    """Build deterministic, JSON-serializable aggregation metadata."""
    report_subtypes: dict[str, object] = {}
    for subtype in SOURCE_SUBTYPES:
        stats = result.subtypes[subtype]
        subtype_rows = [row for row in result.rows if row.source_subtype == subtype]
        top_morphemes: dict[str, list[dict[str, object]]] = {}
        for pos in TOP_MORPHEME_POS:
            top_morphemes[pos] = [
                {"morpheme": row.morpheme, "count": row.count}
                for row in sorted(
                    (row for row in subtype_rows if row.pos == pos),
                    key=lambda row: (-row.count, row.morpheme),
                )[:20]
            ]
        report_subtypes[subtype] = {
            "total_sentences": stats.total_sentences,
            "total_mp_records": stats.total_mp_records,
            "counted_mp": stats.counted_mp,
            "excluded_mp": stats.excluded_mp,
            "unique_morpheme_pos": stats.unique_morpheme_pos,
            "pos_counts": dict(stats.pos_counts),
            "top_pos": [
                {"pos": pos, "count": count}
                for pos, count in sorted(
                    stats.pos_counts.items(), key=lambda item: (-item[1], item[0])
                )
            ],
            "validation_issue_counts": dict(stats.issue_counts),
            "top_morphemes": top_morphemes,
        }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "aggregation_key": ["source_subtype", "morpheme", "pos"],
        "morpheme_normalization": "NFKC+strip",
        "pos_policy": "original_label",
        "issue_policy": {
            "excluded": sorted(EXCLUDING_ISSUES),
            "counted_with_warning": sorted(WARNING_ISSUES),
        },
        "subtypes": report_subtypes,
    }


def write_outputs(
    result: ModuFrequencyResult, csv_path: Path, report_path: Path
) -> dict[str, object]:
    """Atomically write deterministic UTF-8 CSV and JSON outputs."""
    report = build_report(result)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    csv_temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    report_temporary = report_path.with_name(f".{report_path.name}.{os.getpid()}.tmp")
    try:
        with csv_temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(
                {
                    "source_subtype": row.source_subtype,
                    "morpheme": row.morpheme,
                    "pos": row.pos,
                    "count": row.count,
                }
                for row in result.rows
            )
        report_temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        csv_temporary.replace(csv_path)
        report_temporary.replace(report_path)
    except OSError:
        csv_temporary.unlink(missing_ok=True)
        report_temporary.unlink(missing_ok=True)
        raise
    return report
