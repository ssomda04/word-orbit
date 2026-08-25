"""Conservatively aggregate surface frequencies into Kiwi lemma/POS buckets."""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import random
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

ANALYSIS_STATUSES = ("exact", "ambiguous", "multi_morpheme", "unanalyzed")
DEFAULT_AMBIGUITY_MARGIN = 3.0
DEFAULT_TOP_N = 3
DEFAULT_SAMPLE_SEED = 20260823

ENDING_TAGS = frozenset({"EP", "EF", "EC", "ETN", "ETM", "Z_CODA", "Z_SIOT"})
PREDICATE_TAGS = {"VV": "동사", "VA": "형용사", "VX": "동사", "VCN": "형용사"}
DERIVATIONAL_TAGS = {"XSV": "동사", "XSA": "형용사"}
DERIVATIONAL_ROOT_TAGS = frozenset({"NNG", "NNP", "NNB", "XR", "SL", "SH"})
SINGLE_TOKEN_POS = {
    "NNG": "명사",
    "NNP": "고유명사",
    "NNB": "의존명사",
    "NR": "수사",
    "NP": "대명사",
    "MAG": "부사",
    "MAJ": "부사",
    "MM": "관형사",
    "IC": "감탄사",
}
POS_ORDER = ("명사", "동사", "형용사", "부사", "관형사", "감탄사", "대명사", "수사")
PREDICATE_POS = frozenset({"동사", "형용사"})
MANDATORY_EXAMPLES = (
    "먹다",
    "가다",
    "알다",
    "모르다",
    "주다",
    "사랑하다",
    "말하다",
    "생각하다",
    "행복하다",
    "아름답다",
    "작다",
    "웃다",
)


class LemmaFrequencyError(RuntimeError):
    """Raised when lemma-frequency input or output is invalid."""


class TokenLike(Protocol):
    form: str
    tag: str


class KiwiLike(Protocol):
    def analyze(
        self, text: str, top_n: int = 1
    ) -> Sequence[tuple[Sequence[TokenLike], float]]: ...


@dataclass(frozen=True, slots=True)
class ParsedAssignment:
    status: str
    lemma: str = ""
    pos: str = ""
    kiwi_tag: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        if self.status == "exact":
            return self.status, self.lemma, self.pos
        return self.status, "", ""


@dataclass(frozen=True, slots=True)
class SurfaceAnalysis:
    surface: str
    count: int
    lemma: str
    pos: str
    kiwi_tag: str
    analysis_status: str
    best_score: float | None
    score_margin: float | None
    analysis_candidates: str

    def as_row(self) -> dict[str, str | int]:
        return {
            "surface": self.surface,
            "count": self.count,
            "lemma": self.lemma,
            "pos": self.pos,
            "kiwi_tag": self.kiwi_tag,
            "analysis_status": self.analysis_status,
            "best_score": _optional_float(self.best_score),
            "score_margin": _optional_float(self.score_margin),
            "analysis_candidates": self.analysis_candidates,
        }


@dataclass(frozen=True, slots=True)
class LemmaAggregate:
    lemma: str
    pos: str
    count: int
    source_surface_count: int
    analysis_status: str = "exact"

    def as_row(self) -> dict[str, str | int]:
        return {
            "lemma": self.lemma,
            "pos": self.pos,
            "count": self.count,
            "source_surface_count": self.source_surface_count,
            "analysis_status": self.analysis_status,
        }


def normalize_word(word: str) -> str:
    return unicodedata.normalize("NFKC", word).strip()


def _base_tag(tag: str) -> str:
    return tag.split("-", maxsplit=1)[0]


def _optional_float(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.6f}"


def load_surface_frequency(path: Path) -> list[tuple[str, int]]:
    """Load and validate a UTF-8 ``surface count`` frequency list."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise LemmaFrequencyError(f"Could not read UTF-8 frequency input {path}: {exc}") from exc

    rows: list[tuple[str, int]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.rsplit(maxsplit=1)
        if len(parts) != 2:
            raise LemmaFrequencyError(
                f"Frequency line {line_number} is not in 'word count' format."
            )
        surface = normalize_word(parts[0])
        if not surface:
            raise LemmaFrequencyError(f"Frequency line {line_number} has a blank word.")
        try:
            count = int(parts[1])
        except ValueError as exc:
            raise LemmaFrequencyError(
                f"Frequency line {line_number} has a non-integer count: {parts[1]!r}."
            ) from exc
        if count < 0:
            raise LemmaFrequencyError(
                f"Frequency line {line_number} has a negative count: {count}."
            )
        if surface in seen:
            raise LemmaFrequencyError(
                f"Frequency input has duplicate normalized surface {surface!r}."
            )
        seen.add(surface)
        rows.append((surface, count))
    if not rows:
        raise LemmaFrequencyError("Frequency input contains no usable rows.")
    return rows


def classify_tokens(tokens: Sequence[TokenLike], surface: str) -> ParsedAssignment:
    """Map one Kiwi parse to one conservative lexical assignment or a non-exact status."""
    if not tokens:
        return ParsedAssignment("unanalyzed")

    forms = [normalize_word(token.form) for token in tokens]
    tags = [_base_tag(token.tag) for token in tokens]
    if any(not form for form in forms) or any(tag in {"UNK", "W_UNKNOWN"} for tag in tags):
        return ParsedAssignment("unanalyzed")

    first_pos = PREDICATE_TAGS.get(tags[0])
    if first_pos and all(tag in ENDING_TAGS for tag in tags[1:]):
        return ParsedAssignment("exact", f"{forms[0]}다", first_pos, tags[0])

    last_core = len(tags) - 1
    while last_core >= 0 and tags[last_core] in ENDING_TAGS:
        last_core -= 1
    if last_core >= 1 and all(
        tag in DERIVATIONAL_ROOT_TAGS for tag in tags[:last_core]
    ):
        derivation_pos = DERIVATIONAL_TAGS.get(tags[last_core])
        if derivation_pos is None and forms[last_core] == "하":
            derivation_pos = PREDICATE_TAGS.get(tags[last_core])
        if derivation_pos:
            lemma = "".join(forms[: last_core + 1]) + "다"
            return ParsedAssignment("exact", lemma, derivation_pos, tags[last_core])

    if len(tokens) == 1:
        pos = SINGLE_TOKEN_POS.get(tags[0])
        if pos:
            return ParsedAssignment("exact", normalize_word(surface), pos, tags[0])

    return ParsedAssignment("multi_morpheme")


def analyze_surface(
    kiwi: KiwiLike,
    surface: str,
    count: int,
    *,
    top_n: int = DEFAULT_TOP_N,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> SurfaceAnalysis:
    """Analyze one surface without ever assigning its count to multiple lemmas."""
    if top_n < 1:
        raise LemmaFrequencyError("top_n must be at least 1.")
    if ambiguity_margin < 0 or not math.isfinite(ambiguity_margin):
        raise LemmaFrequencyError("ambiguity_margin must be finite and non-negative.")

    raw_analyses = list(kiwi.analyze(surface, top_n=top_n))
    if not raw_analyses:
        return SurfaceAnalysis(surface, count, "", "", "", "unanalyzed", None, None, "[]")

    candidates: list[tuple[ParsedAssignment, float, list[list[str]]]] = []
    for tokens, raw_score in raw_analyses:
        score = float(raw_score)
        if not math.isfinite(score):
            continue
        assignment = classify_tokens(tokens, surface)
        token_view = [[normalize_word(token.form), token.tag] for token in tokens]
        candidates.append((assignment, score, token_view))
    if not candidates:
        return SurfaceAnalysis(surface, count, "", "", "", "unanalyzed", None, None, "[]")

    candidates.sort(key=lambda item: item[1], reverse=True)
    best_assignment, best_score, _ = candidates[0]
    score_margin = best_score - candidates[1][1] if len(candidates) > 1 else None
    status = best_assignment.status
    lemma = best_assignment.lemma
    pos = best_assignment.pos
    kiwi_tag = best_assignment.kiwi_tag
    for competitor, score, _ in candidates[1:]:
        if best_score - score > ambiguity_margin:
            continue
        if competitor.key != best_assignment.key:
            status = "ambiguous"
            lemma = ""
            pos = ""
            kiwi_tag = ""
            break

    candidate_view = [
        {
            "score": round(score, 6),
            "status": assignment.status,
            "lemma": assignment.lemma,
            "pos": assignment.pos,
            "tokens": token_view,
        }
        for assignment, score, token_view in candidates
    ]
    return SurfaceAnalysis(
        surface=surface,
        count=count,
        lemma=lemma,
        pos=pos,
        kiwi_tag=kiwi_tag,
        analysis_status=status,
        best_score=best_score,
        score_margin=score_margin,
        analysis_candidates=json.dumps(candidate_view, ensure_ascii=False, separators=(",", ":")),
    )


def analyze_frequencies(
    kiwi: KiwiLike,
    rows: Sequence[tuple[str, int]],
    *,
    top_n: int = DEFAULT_TOP_N,
    ambiguity_margin: float = DEFAULT_AMBIGUITY_MARGIN,
) -> tuple[list[SurfaceAnalysis], list[LemmaAggregate]]:
    """Analyze every surface and aggregate only exact single assignments."""
    analyses = [
        analyze_surface(
            kiwi,
            surface,
            count,
            top_n=top_n,
            ambiguity_margin=ambiguity_margin,
        )
        for surface, count in rows
    ]
    totals: defaultdict[tuple[str, str], int] = defaultdict(int)
    sources: Counter[tuple[str, str]] = Counter()
    for analysis in analyses:
        if analysis.analysis_status != "exact":
            continue
        key = analysis.lemma, analysis.pos
        totals[key] += analysis.count
        sources[key] += 1
    aggregates = [
        LemmaAggregate(lemma, pos, count, sources[(lemma, pos)])
        for (lemma, pos), count in totals.items()
    ]
    aggregates.sort(key=lambda row: (row.lemma, row.pos))
    return analyses, aggregates


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
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
        raise LemmaFrequencyError(f"Could not write CSV {path}: {exc}") from exc


def write_analysis_csv(path: Path, analyses: Sequence[SurfaceAnalysis]) -> None:
    _atomic_csv(
        path,
        (
            "surface",
            "count",
            "lemma",
            "pos",
            "kiwi_tag",
            "analysis_status",
            "best_score",
            "score_margin",
            "analysis_candidates",
        ),
        [row.as_row() for row in analyses],
    )


def write_lemma_csv(path: Path, aggregates: Sequence[LemmaAggregate]) -> None:
    _atomic_csv(
        path,
        ("lemma", "pos", "count", "source_surface_count", "analysis_status"),
        [row.as_row() for row in aggregates],
    )


def _candidate_pos_buckets(raw_pos: str) -> set[str]:
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


def join_candidates(
    candidate_path: Path,
    aggregates: Sequence[LemmaAggregate],
    output_path: Path,
) -> list[dict[str, str]]:
    """Join exact lemma/POS buckets without producing a rank artifact."""
    try:
        handle = candidate_path.open(encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise LemmaFrequencyError(f"Could not read candidates {candidate_path}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "word" not in reader.fieldnames or "pos" not in reader.fieldnames:
            raise LemmaFrequencyError("Candidate CSV requires 'word' and 'pos' columns.")
        fieldnames = list(reader.fieldnames)
        candidates = [dict(row) for row in reader]

    output_fields = (
        "lemma_frequency",
        "lemma_frequency_found",
        "lemma_frequency_pos",
        "lemma_source_surface_count",
    )
    collisions = set(fieldnames) & set(output_fields)
    if collisions:
        raise LemmaFrequencyError(f"Candidate CSV already has lemma output fields: {sorted(collisions)}")

    lookup = {(row.lemma, row.pos): row for row in aggregates}
    joined: list[dict[str, str]] = []
    for candidate in candidates:
        word = normalize_word(candidate.get("word", ""))
        buckets = _candidate_pos_buckets(candidate.get("pos", ""))
        matches = [lookup[(word, pos)] for pos in sorted(buckets) if (word, pos) in lookup]
        row = dict(candidate)
        row["word"] = word
        row["lemma_frequency"] = str(sum(match.count for match in matches)) if matches else ""
        row["lemma_frequency_found"] = "true" if matches else "false"
        row["lemma_frequency_pos"] = "|".join(match.pos for match in matches)
        row["lemma_source_surface_count"] = (
            str(sum(match.source_surface_count for match in matches)) if matches else ""
        )
        joined.append(row)
    _atomic_csv(output_path, (*fieldnames, *output_fields), joined)
    return joined


def _group_stats(
    rows: Sequence[dict[str, str]], raw_surfaces: set[str]
) -> dict[str, int | float | str | None]:
    raw_matched = [row for row in rows if normalize_word(row.get("word", "")) in raw_surfaces]
    lemma_matched = [row for row in rows if row["lemma_frequency_found"] == "true"]
    lemma_values = [int(row["lemma_frequency"]) for row in lemma_matched]
    total = len(rows)
    raw_coverage = len(raw_matched) / total if total else 0.0
    lemma_coverage = len(lemma_matched) / total if total else 0.0
    return {
        "candidate_count": total,
        "raw_matched": len(raw_matched),
        "raw_coverage_ratio": raw_coverage,
        "lemma_matched": len(lemma_matched),
        "lemma_unmatched": total - len(lemma_matched),
        "lemma_coverage_ratio": lemma_coverage,
        "coverage_point_change": (lemma_coverage - raw_coverage) * 100,
        "lemma_frequency_median": statistics.median(lemma_values) if lemma_values else None,
    }


def build_report(
    analyses: Sequence[SurfaceAnalysis],
    aggregates: Sequence[LemmaAggregate],
    joined: Sequence[dict[str, str]],
    raw_rows: Sequence[tuple[str, int]],
    *,
    kiwi_version: str,
    top_n: int,
    ambiguity_margin: float,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> dict[str, Any]:
    """Build deterministic coverage, audit, sample, and concentration statistics."""
    raw_frequency = dict(raw_rows)
    raw_surfaces = set(raw_frequency)
    status_counts = Counter(row.analysis_status for row in analyses)
    {(row.lemma, row.pos): row for row in aggregates}
    source_lookup: defaultdict[tuple[str, str], list[SurfaceAnalysis]] = defaultdict(list)
    for analysis in analyses:
        if analysis.analysis_status == "exact":
            source_lookup[(analysis.lemma, analysis.pos)].append(analysis)

    pos_stats = {
        pos: _group_stats(
            [row for row in joined if pos in row.get("pos", "").split("|")], raw_surfaces
        )
        for pos in POS_ORDER
    }
    review_stats = {
        flag: _group_stats(
            [row for row in joined if row.get("review_required", "").lower() == flag],
            raw_surfaces,
        )
        for flag in ("true", "false")
    }
    review_reasons = (
        "technical_term",
        "historical_term",
        "archaic",
        "long_word",
        "explicit_rare_label",
        "dialect",
    )
    reason_stats = {
        reason: _group_stats(
            [row for row in joined if reason in row.get("review_reason", "").split("|")],
            raw_surfaces,
        )
        for reason in review_reasons
    }

    def sample_pos(pos: str, offset: int) -> list[dict[str, int | str]]:
        pool = sorted((row for row in aggregates if row.pos == pos), key=lambda row: row.lemma)
        chosen = random.Random(seed + offset).sample(pool, min(50, len(pool)))
        return [row.as_row() for row in chosen]

    examples: dict[str, Any] = {}
    for lemma in MANDATORY_EXAMPLES:
        matching = [row for row in aggregates if row.lemma == lemma and row.pos in PREDICATE_POS]
        sources = sorted(
            (surface for row in matching for surface in source_lookup[(row.lemma, row.pos)]),
            key=lambda row: (-row.count, row.surface),
        )
        base_analysis = next((row for row in analyses if row.surface == lemma), None)
        examples[lemma] = {
            "raw_surface_count": raw_frequency.get(lemma),
            "base_analysis_status": (
                base_analysis.analysis_status if base_analysis is not None else "not_in_50k"
            ),
            "lemma_aggregate_count": sum(row.count for row in matching) or None,
            "lemma_pos": sorted({row.pos for row in matching}),
            "related_surfaces": [
                {"surface": row.surface, "count": row.count} for row in sources
            ],
        }

    ambiguous = [row for row in analyses if row.analysis_status == "ambiguous"]
    multi = [row for row in analyses if row.analysis_status == "multi_morpheme"]
    cross_pos: defaultdict[str, list[LemmaAggregate]] = defaultdict(list)
    for aggregate in aggregates:
        cross_pos[aggregate.lemma].append(aggregate)
    cross_pos_risks = [
        {
            "lemma": lemma,
            "buckets": [row.as_row() for row in sorted(rows, key=lambda row: row.pos)],
        }
        for lemma, rows in cross_pos.items()
        if len(rows) > 1
    ]
    cross_pos_risks.sort(
        key=lambda item: -sum(int(bucket["count"]) for bucket in item["buckets"])
    )
    concentrated = sorted(
        aggregates, key=lambda row: (-row.count, -row.source_surface_count, row.lemma, row.pos)
    )[:25]

    return {
        "kiwi_version": kiwi_version,
        "policy": {"top_n": top_n, "ambiguity_score_margin": ambiguity_margin},
        "surface_count": len(analyses),
        "surface_status_counts": {
            status: status_counts.get(status, 0) for status in ANALYSIS_STATUSES
        },
        "exact_assigned_count_total": sum(
            row.count for row in analyses if row.analysis_status == "exact"
        ),
        "non_exact_preserved_count_total": sum(
            row.count for row in analyses if row.analysis_status != "exact"
        ),
        "lemma_unique_count": len({row.lemma for row in aggregates}),
        "lemma_pos_bucket_count": len(aggregates),
        "overall_coverage": _group_stats(joined, raw_surfaces),
        "pos_coverage": pos_stats,
        "review_required_coverage": review_stats,
        "review_reason_coverage": reason_stats,
        "mandatory_examples": examples,
        "verb_sample": sample_pos("동사", 1),
        "adjective_sample": sample_pos("형용사", 2),
        "noun_sample": sample_pos("명사", 3),
        "ambiguous_sample": [row.as_row() for row in ambiguous[:50]],
        "multi_morpheme_sample": [row.as_row() for row in multi[:50]],
        "cross_pos_merge_risks": cross_pos_risks[:25],
        "highest_aggregate_counts": [row.as_row() for row in concentrated],
        "sample_seed": seed,
    }


def format_report(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2)
