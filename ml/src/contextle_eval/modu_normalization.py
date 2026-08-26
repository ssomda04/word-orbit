"""Apply the conservative Modu MP normalization contract with provenance."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from contextle_eval.modu_baseform_analysis import normalize_form

SourceSubtype = Literal["NXMP", "SXMP"]
NormalizationStatus = Literal["matched", "review", "unmatched"]
SOURCE_SUBTYPES: tuple[SourceSubtype, ...] = ("NXMP", "SXMP")
STATUSES: tuple[NormalizationStatus, ...] = ("matched", "review", "unmatched")
INPUT_FIELDS = ("source_subtype", "morpheme", "pos", "count")
NORMALIZED_FIELDS = (
    "source_subtype",
    "source_morpheme",
    "source_pos",
    "canonical_form",
    "status",
    "count",
    "in_game_vocabulary",
    "in_answer_candidates",
)
COLLISION_FIELDS = (
    "collision_type",
    "key",
    "source_subtypes",
    "source_morphemes",
    "source_pos",
    "statuses",
    "source_count",
    "total_count",
)
RAW_MATCH_POS = frozenset({"NNG", "MAG"})
BASEFORM_POS = frozenset({"VV", "VA"})
REVIEW_POS = frozenset({"NNP", "IC"})
REPORT_SCHEMA_VERSION = "1.0"
REQUIRED_AUDIT_CASES = frozenset(
    {"있다", "보다", "맞다", "크다", "쓰다", "사다", "이제", "안", "하다", "되다"}
)


class ModuNormalizationError(ValueError):
    """Raised when normalization inputs or invariants are invalid."""


@dataclass(frozen=True, slots=True)
class RawFrequencyRow:
    """One source frequency row before lexical normalization."""

    source_subtype: SourceSubtype
    morpheme: str
    pos: str
    count: int


@dataclass(frozen=True, slots=True)
class NormalizedFrequencyRow:
    """One normalized row retaining its full source provenance."""

    source_subtype: SourceSubtype
    source_morpheme: str
    source_pos: str
    canonical_form: str
    status: NormalizationStatus
    count: int
    in_game_vocabulary: bool
    in_answer_candidates: bool

    def as_csv_row(self) -> dict[str, str | int]:
        return {
            "source_subtype": self.source_subtype,
            "source_morpheme": self.source_morpheme,
            "source_pos": self.source_pos,
            "canonical_form": self.canonical_form,
            "status": self.status,
            "count": self.count,
            "in_game_vocabulary": str(self.in_game_vocabulary).lower(),
            "in_answer_candidates": str(self.in_answer_candidates).lower(),
        }


@dataclass(frozen=True, slots=True)
class CollisionRecord:
    """One collision group without merging any source frequency rows."""

    collision_type: str
    key: str
    source_subtypes: tuple[str, ...]
    source_morphemes: tuple[str, ...]
    source_pos: tuple[str, ...]
    statuses: tuple[str, ...]
    source_count: int
    total_count: int

    def as_csv_row(self) -> dict[str, str | int]:
        return {
            "collision_type": self.collision_type,
            "key": self.key,
            "source_subtypes": json.dumps(
                self.source_subtypes, ensure_ascii=False, separators=(",", ":")
            ),
            "source_morphemes": json.dumps(
                self.source_morphemes, ensure_ascii=False, separators=(",", ":")
            ),
            "source_pos": json.dumps(
                self.source_pos, ensure_ascii=False, separators=(",", ":")
            ),
            "statuses": json.dumps(
                self.statuses, ensure_ascii=False, separators=(",", ":")
            ),
            "source_count": self.source_count,
            "total_count": self.total_count,
        }


def load_raw_frequency(path: Path) -> tuple[RawFrequencyRow, ...]:
    """Load every canonical MP frequency row, including non-target POS."""
    try:
        handle = path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise ModuNormalizationError(f"Could not read MP frequency CSV: {exc}") from exc
    rows: list[RawFrequencyRow] = []
    seen: set[tuple[str, str, str]] = set()
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(INPUT_FIELDS):
            raise ModuNormalizationError("MP frequency CSV header is invalid.")
        for line_number, row in enumerate(reader, start=2):
            subtype = row["source_subtype"]
            morpheme = normalize_form(row["morpheme"])
            pos = row["pos"].strip()
            if subtype not in SOURCE_SUBTYPES:
                raise ModuNormalizationError(
                    f"MP frequency row {line_number} has an invalid subtype."
                )
            if not morpheme or not pos:
                raise ModuNormalizationError(
                    f"MP frequency row {line_number} has an empty normalized key."
                )
            try:
                count = int(row["count"])
            except ValueError as exc:
                raise ModuNormalizationError(
                    f"MP frequency row {line_number} has an invalid count."
                ) from exc
            if count < 1:
                raise ModuNormalizationError(
                    f"MP frequency row {line_number} has a non-positive count."
                )
            key = (subtype, morpheme, pos)
            if key in seen:
                raise ModuNormalizationError("MP frequency CSV contains a duplicate key.")
            seen.add(key)
            rows.append(RawFrequencyRow(subtype, morpheme, pos, count))
    if not rows:
        raise ModuNormalizationError("MP frequency CSV is empty.")
    return tuple(rows)


def normalize_row(
    row: RawFrequencyRow,
    game_words: frozenset[str],
    answer_words: frozenset[str],
) -> NormalizedFrequencyRow:
    """Apply the POS gate; candidate membership remains evidence only."""
    if row.pos in RAW_MATCH_POS:
        canonical = row.morpheme
        status: NormalizationStatus = (
            "matched" if canonical in game_words else "unmatched"
        )
    elif row.pos in BASEFORM_POS:
        canonical = f"{row.morpheme}다"
        status = "matched" if canonical in game_words else "review"
    elif row.pos in REVIEW_POS:
        canonical = row.morpheme
        status = "review"
    else:
        canonical = ""
        status = "unmatched"
    return NormalizedFrequencyRow(
        source_subtype=row.source_subtype,
        source_morpheme=row.morpheme,
        source_pos=row.pos,
        canonical_form=canonical,
        status=status,
        count=row.count,
        in_game_vocabulary=bool(canonical and canonical in game_words),
        in_answer_candidates=bool(canonical and canonical in answer_words),
    )


def normalize_frequencies(
    rows: Iterable[RawFrequencyRow],
    game_words: frozenset[str],
    answer_words: frozenset[str],
) -> tuple[NormalizedFrequencyRow, ...]:
    """Normalize rows deterministically without aggregating canonical collisions."""
    normalized = tuple(normalize_row(row, game_words, answer_words) for row in rows)
    if sum(row.count for row in normalized) != sum(row.count for row in rows):
        raise ModuNormalizationError("Normalization did not conserve token counts.")
    return normalized


def _collision_record(
    collision_type: str, key: str, rows: Sequence[NormalizedFrequencyRow]
) -> CollisionRecord:
    return CollisionRecord(
        collision_type=collision_type,
        key=key,
        source_subtypes=tuple(sorted({row.source_subtype for row in rows})),
        source_morphemes=tuple(sorted({row.source_morpheme for row in rows})),
        source_pos=tuple(sorted({row.source_pos for row in rows})),
        statuses=tuple(sorted({row.status for row in rows})),
        source_count=len(rows),
        total_count=sum(row.count for row in rows),
    )


def detect_collisions(
    rows: Sequence[NormalizedFrequencyRow],
) -> tuple[CollisionRecord, ...]:
    """Detect raw/POS, canonical-source, and cross-subtype role collisions."""
    raw_groups: dict[str, list[NormalizedFrequencyRow]] = defaultdict(list)
    canonical_groups: dict[str, list[NormalizedFrequencyRow]] = defaultdict(list)
    for row in rows:
        raw_groups[row.source_morpheme].append(row)
        if row.canonical_form:
            canonical_groups[row.canonical_form].append(row)

    collisions: list[CollisionRecord] = []
    for raw_form, group in raw_groups.items():
        if len({row.source_pos for row in group}) > 1:
            collisions.append(_collision_record("cross_pos", raw_form, group))
    for canonical, group in canonical_groups.items():
        sources = {(row.source_morpheme, row.source_pos) for row in group}
        if len(sources) > 1:
            collisions.append(_collision_record("canonical", canonical, group))
        subtype_roles: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for row in group:
            subtype_roles[row.source_subtype].add((row.source_pos, row.status))
        if len(subtype_roles) > 1 and len(set(map(frozenset, subtype_roles.values()))) > 1:
            collisions.append(_collision_record("source_subtype", canonical, group))
    for canonical in sorted(REQUIRED_AUDIT_CASES):
        if group := canonical_groups.get(canonical):
            collisions.append(_collision_record("required_case", canonical, group))
    return tuple(
        sorted(
            collisions,
            key=lambda item: (
                (
                    "cross_pos",
                    "canonical",
                    "source_subtype",
                    "required_case",
                ).index(item.collision_type),
                -item.total_count,
                item.key,
            ),
        )
    )


def _status_summary(rows: Sequence[NormalizedFrequencyRow]) -> dict[str, Any]:
    token_total = sum(row.count for row in rows)
    result: dict[str, Any] = {
        "total_row_count": len(rows),
        "total_token_count": token_total,
        "unique_canonical_matched_words": len(
            {row.canonical_form for row in rows if row.status == "matched"}
        ),
    }
    for status in STATUSES:
        status_rows = [row for row in rows if row.status == status]
        tokens = sum(row.count for row in status_rows)
        result[status] = {
            "row_count": len(status_rows),
            "token_count": tokens,
            "token_rate": tokens / token_total if token_total else 0.0,
        }
    result["count_conservation"] = (
        sum(result[status]["token_count"] for status in STATUSES) == token_total
    )
    return result


def _pos_summary(rows: Sequence[NormalizedFrequencyRow]) -> dict[str, Any]:
    by_pos: dict[str, list[NormalizedFrequencyRow]] = defaultdict(list)
    for row in rows:
        by_pos[row.source_pos].append(row)
    return {
        pos: _status_summary(pos_rows)
        for pos, pos_rows in sorted(by_pos.items())
    }


def _render_frequency_row(row: NormalizedFrequencyRow) -> dict[str, Any]:
    return {
        "source_subtype": row.source_subtype,
        "source_morpheme": row.source_morpheme,
        "source_pos": row.source_pos,
        "canonical_form": row.canonical_form,
        "status": row.status,
        "count": row.count,
        "in_game_vocabulary": row.in_game_vocabulary,
        "in_answer_candidates": row.in_answer_candidates,
    }


def build_normalization_report(
    rows: Sequence[NormalizedFrequencyRow],
    collisions: Sequence[CollisionRecord],
    *,
    input_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build conservation, subtype/POS, top-frequency, and collision summaries."""
    overall = _status_summary(rows)
    if not overall["count_conservation"]:
        raise ModuNormalizationError("Report totals do not conserve input counts.")
    collision_counts = Counter(collision.collision_type for collision in collisions)
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            -row.count,
            row.canonical_form,
            row.source_subtype,
            row.source_morpheme,
            row.source_pos,
        ),
    )
    subtype_reports: dict[str, Any] = {}
    for subtype in SOURCE_SUBTYPES:
        subtype_rows = [row for row in rows if row.source_subtype == subtype]
        subtype_reports[subtype] = {
            **_status_summary(subtype_rows),
            "by_pos": _pos_summary(subtype_rows),
        }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "inputs": dict(input_metadata or {}),
        "contract": {
            "normalization": "NFKC+strip",
            "answer_candidate_membership": "evidence_only",
            "canonical_collisions_merged": False,
        },
        "overall": {**overall, "by_pos": _pos_summary(rows)},
        "subtypes": subtype_reports,
        "collisions": {
            "counts": {
                collision_type: collision_counts[collision_type]
                for collision_type in ("cross_pos", "canonical", "source_subtype")
            },
            "required_case_audit": [
                collision.as_csv_row()
                for collision in collisions
                if collision.collision_type == "required_case"
            ],
        },
        "top_matched_rows": [
            _render_frequency_row(row)
            for row in sorted_rows
            if row.status == "matched"
        ][:30],
        "top_review_rows": [
            _render_frequency_row(row)
            for row in sorted_rows
            if row.status == "review"
        ][:30],
    }


def write_normalized_frequency(path: Path, rows: Iterable[NormalizedFrequencyRow]) -> None:
    """Atomically write provenance-preserving normalized frequency rows."""
    _write_csv(path, NORMALIZED_FIELDS, (row.as_csv_row() for row in rows))


def write_collision_audit(path: Path, collisions: Iterable[CollisionRecord]) -> None:
    """Atomically write collision groups without merging their source rows."""
    _write_csv(path, COLLISION_FIELDS, (row.as_csv_row() for row in collisions))


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Iterable[Mapping[str, str | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Atomically write a deterministic UTF-8 JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
