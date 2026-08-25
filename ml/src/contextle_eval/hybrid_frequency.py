"""Select exactly one raw or lemma frequency source per answer candidate."""

from __future__ import annotations

import contextlib
import csv
import json
import os
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from contextle_eval.lemma_frequency import normalize_word

DEFAULT_SAMPLE_SEED = 20260823
PREDICATE_POS = frozenset({"동사", "형용사"})
REPORT_POS = ("명사", "동사", "형용사", "부사", "관형사", "감탄사", "대명사", "수사")
SPECIAL_REVIEW_LEMMAS = (
    "하다",
    "되다",
    "있다",
    "보다",
    "않다",
    "감사하다",
    "진정하다",
    "달다",
)
OUTPUT_FIELDS = (
    "raw_frequency",
    "lemma_frequency",
    "lemma_frequency_pos",
    "lemma_source_surface_count",
    "frequency_policy_source",
    "selected_frequency",
    "selected_frequency_source",
    "frequency_found",
    "hybrid_risk_flags",
)


class HybridFrequencyError(RuntimeError):
    """Raised when a hybrid-frequency input or output is invalid."""


@dataclass(frozen=True, slots=True)
class LemmaBucket:
    lemma: str
    pos: str
    count: int
    source_surface_count: int

    def as_dict(self) -> dict[str, str | int]:
        return {
            "lemma": self.lemma,
            "pos": self.pos,
            "count": self.count,
            "source_surface_count": self.source_surface_count,
        }


def _parse_non_negative_int(raw: str, *, field: str, row_number: int) -> int:
    try:
        value = int(raw.strip())
    except (AttributeError, ValueError) as exc:
        raise HybridFrequencyError(
            f"Row {row_number} has an invalid integer {field}: {raw!r}."
        ) from exc
    if value < 0:
        raise HybridFrequencyError(
            f"Row {row_number} requires non-negative {field}; got {value}."
        )
    return value


def load_raw_frequency(path: Path) -> dict[str, int]:
    """Load a normalized, unique raw surface-frequency CSV."""
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise HybridFrequencyError(f"Could not read raw frequency {path}: {exc}") from exc
    records: dict[str, int] = {}
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"word", "count"} <= set(reader.fieldnames):
            raise HybridFrequencyError("Raw frequency CSV requires 'word' and 'count'.")
        for row_number, row in enumerate(reader, start=2):
            word = normalize_word(row.get("word", ""))
            if not word:
                raise HybridFrequencyError(f"Raw frequency row {row_number} has a blank word.")
            if word in records:
                raise HybridFrequencyError(f"Duplicate normalized raw word {word!r}.")
            records[word] = _parse_non_negative_int(
                row.get("count", ""), field="count", row_number=row_number
            )
    if not records:
        raise HybridFrequencyError("Raw frequency CSV contains no usable rows.")
    return records


def load_lemma_frequency(path: Path) -> dict[tuple[str, str], LemmaBucket]:
    """Load unique exact `(lemma, POS)` aggregate buckets."""
    required = {"lemma", "pos", "count", "source_surface_count", "analysis_status"}
    try:
        handle = path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise HybridFrequencyError(f"Could not read lemma frequency {path}: {exc}") from exc
    records: dict[tuple[str, str], LemmaBucket] = {}
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise HybridFrequencyError(f"Lemma frequency CSV requires {sorted(required)}.")
        for row_number, row in enumerate(reader, start=2):
            if row.get("analysis_status") != "exact":
                raise HybridFrequencyError(
                    f"Lemma frequency row {row_number} is not an exact aggregate."
                )
            lemma = normalize_word(row.get("lemma", ""))
            pos = normalize_word(row.get("pos", ""))
            if not lemma or not pos:
                raise HybridFrequencyError(
                    f"Lemma frequency row {row_number} has a blank lemma or POS."
                )
            key = lemma, pos
            if key in records:
                raise HybridFrequencyError(f"Duplicate lemma/POS bucket {key!r}.")
            records[key] = LemmaBucket(
                lemma=lemma,
                pos=pos,
                count=_parse_non_negative_int(
                    row.get("count", ""), field="count", row_number=row_number
                ),
                source_surface_count=_parse_non_negative_int(
                    row.get("source_surface_count", ""),
                    field="source_surface_count",
                    row_number=row_number,
                ),
            )
    if not records:
        raise HybridFrequencyError("Lemma frequency CSV contains no exact buckets.")
    return records


def candidate_pos_buckets(raw_pos: str) -> set[str]:
    """Map Wiktionary candidate POS labels to hybrid frequency buckets."""
    mapping = {
        "명사": "명사",
        "고유 명사": "고유명사",
        "고유명사": "고유명사",
        "의존 명사": "의존명사",
        "의존명사": "의존명사",
        "대명사": "대명사",
        "수사": "수사",
        "부사": "부사",
        "관형사": "관형사",
        "감탄사": "감탄사",
        "동사": "동사",
        "자동사": "동사",
        "타동사": "동사",
        "조동사": "동사",
        "보조 동사": "동사",
        "보조동사": "동사",
        "형용사": "형용사",
        "보조 형용사": "형용사",
        "보조형용사": "형용사",
    }
    return {mapping[pos] for pos in raw_pos.split("|") if pos in mapping}


def policy_source(raw_pos: str) -> str:
    """Choose one source policy; predicate membership has deterministic precedence."""
    buckets = candidate_pos_buckets(raw_pos)
    return "lemma" if buckets & PREDICATE_POS else "raw"


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise HybridFrequencyError(f"Could not write hybrid CSV {path}: {exc}") from exc


def join_hybrid_frequency(
    candidate_path: Path,
    raw_frequency: dict[str, int],
    lemma_frequency: dict[tuple[str, str], LemmaBucket],
    output_path: Path,
) -> list[dict[str, str]]:
    """Write one selected raw/lemma frequency per candidate without fallback or summing."""
    try:
        handle = candidate_path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise HybridFrequencyError(f"Could not read candidates {candidate_path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"word", "pos"} <= set(reader.fieldnames):
            raise HybridFrequencyError("Candidate CSV requires 'word' and 'pos'.")
        fieldnames = list(reader.fieldnames)
        collisions = set(fieldnames) & set(OUTPUT_FIELDS)
        if collisions:
            raise HybridFrequencyError(
                f"Candidate CSV already has hybrid output fields: {sorted(collisions)}."
            )
        candidates = [dict(row) for row in reader]

    joined: list[dict[str, str]] = []
    for candidate in candidates:
        row = dict(candidate)
        word = normalize_word(row.get("word", ""))
        row["word"] = word
        buckets = candidate_pos_buckets(row.get("pos", ""))
        lemma_matches = [
            lemma_frequency[(word, pos)]
            for pos in sorted(buckets)
            if (word, pos) in lemma_frequency
        ]
        raw_value = raw_frequency.get(word)
        lemma_value = sum(match.count for match in lemma_matches) if lemma_matches else None
        planned_source = policy_source(row.get("pos", ""))
        selected_value = lemma_value if planned_source == "lemma" else raw_value
        selected_source = planned_source if selected_value is not None else "none"

        non_predicate_buckets = buckets - PREDICATE_POS
        flags: list[str] = []
        if buckets & PREDICATE_POS and non_predicate_buckets:
            flags.append("mixed_candidate_pos_predicate_precedence")
        if len(lemma_matches) > 1:
            flags.append("cross_pos_lemma_buckets")
        if word in SPECIAL_REVIEW_LEMMAS:
            flags.append("special_lemma_review")

        row.update(
            {
                "raw_frequency": str(raw_value) if raw_value is not None else "",
                "lemma_frequency": str(lemma_value) if lemma_value is not None else "",
                "lemma_frequency_pos": "|".join(match.pos for match in lemma_matches),
                "lemma_source_surface_count": (
                    str(sum(match.source_surface_count for match in lemma_matches))
                    if lemma_matches
                    else ""
                ),
                "frequency_policy_source": planned_source,
                "selected_frequency": (
                    str(selected_value) if selected_value is not None else ""
                ),
                "selected_frequency_source": selected_source,
                "frequency_found": "true" if selected_value is not None else "false",
                "hybrid_risk_flags": "|".join(flags),
            }
        )
        joined.append(row)

    _atomic_csv(output_path, (*fieldnames, *OUTPUT_FIELDS), joined)
    return joined


def load_special_sources(
    audit_path: Path,
) -> dict[tuple[str, str], list[dict[str, str | int]]]:
    """Load exact source surfaces only for the explicitly reviewed lemmas."""
    required = {"surface", "count", "lemma", "pos", "analysis_status"}
    try:
        handle = audit_path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise HybridFrequencyError(f"Could not read lemma audit {audit_path}: {exc}") from exc
    sources: defaultdict[tuple[str, str], list[dict[str, str | int]]] = defaultdict(list)
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise HybridFrequencyError(f"Lemma audit CSV requires {sorted(required)}.")
        for row_number, row in enumerate(reader, start=2):
            lemma = normalize_word(row.get("lemma", ""))
            if row.get("analysis_status") != "exact" or lemma not in SPECIAL_REVIEW_LEMMAS:
                continue
            sources[(lemma, normalize_word(row.get("pos", "")))].append(
                {
                    "surface": normalize_word(row.get("surface", "")),
                    "count": _parse_non_negative_int(
                        row.get("count", ""), field="count", row_number=row_number
                    ),
                }
            )
    for rows in sources.values():
        rows.sort(key=lambda row: (-int(row["count"]), str(row["surface"])))
    return dict(sources)


def _group_stats(rows: Sequence[dict[str, str]]) -> dict[str, int | float | None]:
    total = len(rows)
    raw = [row for row in rows if row["raw_frequency"]]
    lemma = [row for row in rows if row["lemma_frequency"]]
    selected = [row for row in rows if row["frequency_found"] == "true"]
    selected_values = [int(row["selected_frequency"]) for row in selected]
    raw_coverage = len(raw) / total if total else 0.0
    lemma_coverage = len(lemma) / total if total else 0.0
    selected_coverage = len(selected) / total if total else 0.0
    return {
        "candidate_count": total,
        "raw_matched": len(raw),
        "raw_coverage_ratio": raw_coverage,
        "lemma_matched": len(lemma),
        "lemma_coverage_ratio": lemma_coverage,
        "hybrid_matched": len(selected),
        "hybrid_unmatched": total - len(selected),
        "hybrid_coverage_ratio": selected_coverage,
        "hybrid_point_change_vs_raw": (selected_coverage - raw_coverage) * 100,
        "hybrid_point_change_vs_lemma": (selected_coverage - lemma_coverage) * 100,
        "selected_frequency_median": (
            statistics.median(selected_values) if selected_values else None
        ),
        "selected_raw_count": sum(
            row["selected_frequency_source"] == "raw" for row in rows
        ),
        "selected_lemma_count": sum(
            row["selected_frequency_source"] == "lemma" for row in rows
        ),
        "selected_none_count": sum(
            row["selected_frequency_source"] == "none" for row in rows
        ),
    }


def _sample_view(row: dict[str, str]) -> dict[str, str]:
    return {
        key: row.get(key, "")
        for key in (
            "word",
            "pos",
            "selected_frequency",
            "selected_frequency_source",
            "raw_frequency",
            "lemma_frequency",
            "review_required",
            "review_reason",
            "hybrid_risk_flags",
        )
    }


def _random_sample(
    rows: Sequence[dict[str, str]], *, seed: int, size: int
) -> list[dict[str, str]]:
    if not rows:
        return []
    return [_sample_view(row) for row in random.Random(seed).sample(list(rows), min(size, len(rows)))]


def build_hybrid_report(
    joined: Sequence[dict[str, str]],
    raw_frequency: dict[str, int],
    lemma_frequency: dict[tuple[str, str], LemmaBucket],
    special_sources: dict[tuple[str, str], list[dict[str, str | int]]],
    *,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> dict[str, Any]:
    """Build deterministic coverage, sample, review, and concentration analysis."""
    matched = sorted(
        (row for row in joined if row["frequency_found"] == "true"),
        key=lambda row: (-int(row["selected_frequency"]), row["word"]),
    )
    unmatched = [row for row in joined if row["frequency_found"] == "false"]
    middle_start = len(matched) * 45 // 100
    middle_end = -(-len(matched) * 55 // 100)
    low_start = len(matched) * 90 // 100

    pos_stats = {
        pos: _group_stats([row for row in joined if pos in row.get("pos", "").split("|")])
        for pos in REPORT_POS
    }
    review_stats = {
        flag: _group_stats(
            [row for row in joined if row.get("review_required", "").lower() == flag]
        )
        for flag in ("true", "false")
    }
    reasons = (
        "technical_term",
        "historical_term",
        "archaic",
        "long_word",
        "explicit_rare_label",
        "dialect",
    )
    reason_stats = {
        reason: _group_stats(
            [row for row in joined if reason in row.get("review_reason", "").split("|")]
        )
        for reason in reasons
    }

    special_review: dict[str, Any] = {}
    by_word = {row["word"]: row for row in joined}
    for lemma in SPECIAL_REVIEW_LEMMAS:
        candidate = by_word.get(lemma)
        buckets = sorted(
            (bucket for (word, _), bucket in lemma_frequency.items() if word == lemma),
            key=lambda bucket: bucket.pos,
        )
        special_review[lemma] = {
            "candidate_pos": candidate.get("pos", "") if candidate else "",
            "raw_frequency": raw_frequency.get(lemma),
            "lemma_buckets": [bucket.as_dict() for bucket in buckets],
            "selected_frequency": candidate.get("selected_frequency", "") if candidate else "",
            "selected_frequency_source": (
                candidate.get("selected_frequency_source", "none") if candidate else "none"
            ),
            "risk_flags": candidate.get("hybrid_risk_flags", "") if candidate else "",
            "top_source_surfaces": {
                bucket.pos: special_sources.get((lemma, bucket.pos), [])[:20]
                for bucket in buckets
            },
        }

    mixed_pos = [
        row
        for row in joined
        if "mixed_candidate_pos_predicate_precedence" in row["hybrid_risk_flags"]
    ]
    cross_pos = [
        row for row in joined if "cross_pos_lemma_buckets" in row["hybrid_risk_flags"]
    ]
    selected_lemma = [
        row for row in matched if row["selected_frequency_source"] == "lemma"
    ]
    selected_lemma.sort(
        key=lambda row: (
            -int(row["selected_frequency"]),
            -int(row["lemma_source_surface_count"] or 0),
            row["word"],
        )
    )

    predicate_samples = {}
    for offset, pos in enumerate(("동사", "형용사"), start=5):
        pool = sorted(
            (
                row
                for row in joined
                if pos in row.get("pos", "").split("|")
                and row["frequency_found"] == "true"
                and row["selected_frequency_source"] == "lemma"
            ),
            key=lambda row: row["word"],
        )
        predicate_samples[pos] = _random_sample(pool, seed=seed + offset, size=50)

    return {
        "policy": {
            "predicate_pos": sorted(PREDICATE_POS),
            "predicate_source": "lemma",
            "other_pos_source": "raw",
            "mixed_pos_precedence": "lemma",
            "fallback_to_other_source": False,
            "missing_frequency_value": "blank",
        },
        "overall_coverage": _group_stats(joined),
        "pos_coverage": pos_stats,
        "review_required_coverage": review_stats,
        "review_reason_coverage": reason_stats,
        "selected_source_counts": dict(
            Counter(row["selected_frequency_source"] for row in joined)
        ),
        "top_100": [_sample_view(row) for row in matched[:100]],
        "middle_100": _random_sample(
            matched[middle_start:middle_end], seed=seed + 1, size=100
        ),
        "low_100": _random_sample(matched[low_start:], seed=seed + 2, size=100),
        "unmatched_100": _random_sample(unmatched, seed=seed + 3, size=100),
        "verb_50": predicate_samples["동사"],
        "adjective_50": predicate_samples["형용사"],
        "special_lemma_review": special_review,
        "mixed_candidate_pos_count": len(mixed_pos),
        "mixed_candidate_pos_sample": [_sample_view(row) for row in mixed_pos[:100]],
        "cross_pos_lemma_count": len(cross_pos),
        "cross_pos_lemma_sample": [_sample_view(row) for row in cross_pos[:100]],
        "high_lemma_frequency_risks": [
            _sample_view(row) for row in selected_lemma[:50]
        ],
        "sample_seed": seed,
    }


def format_hybrid_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)
