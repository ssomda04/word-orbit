"""Analyze conservative lexical base-form mappings for raw Modu MP counts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

SourceSubtype = Literal["NXMP", "SXMP"]
SOURCE_SUBTYPES: tuple[SourceSubtype, ...] = ("NXMP", "SXMP")

REPORT_SCHEMA_VERSION = "1.0"
TARGET_POS = (
    "NNG",
    "NNP",
    "VV",
    "VA",
    "MAG",
    "VX",
    "XSV",
    "XSA",
    "VCP",
    "VCN",
    "NNB",
    "NP",
    "IC",
)
PREDICATE_POS = frozenset({"VV", "VA", "VX", "XSV", "XSA"})
RAW_LEXICAL_POS = frozenset({"NNG", "MAG"})
BASEFORM_LEXICAL_POS = frozenset({"VV", "VA"})
POS_ROLES: Mapping[str, str] = {
    "NNG": "lexical content candidate",
    "NNP": "proper noun",
    "VV": "lexical content candidate",
    "VA": "lexical content candidate",
    "MAG": "lexical content candidate",
    "VX": "auxiliary/functional",
    "XSV": "auxiliary/functional",
    "XSA": "auxiliary/functional",
    "VCP": "grammatical/function word",
    "VCN": "grammatical/function word",
    "NNB": "grammatical/function word",
    "NP": "grammatical/function word",
    "IC": "review required",
}


class ModuBaseformAnalysisError(ValueError):
    """Raised when base-form analysis inputs violate their contracts."""


@dataclass(frozen=True, slots=True)
class FrequencyEntry:
    """One validated frequency row for a target POS."""

    source_subtype: SourceSubtype
    morpheme: str
    pos: str
    count: int


def normalize_form(value: str) -> str:
    """Apply the frequency key's NFKC+strip normalization."""
    return unicodedata.normalize("NFKC", value).strip()


def load_frequency_entries(path: Path) -> tuple[FrequencyEntry, ...]:
    """Load target-POS rows from the deterministic Modu frequency CSV."""
    try:
        handle = path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise ModuBaseformAnalysisError(f"Could not read frequency CSV: {exc}") from exc
    entries: list[FrequencyEntry] = []
    seen: set[tuple[str, str, str]] = set()
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["source_subtype", "morpheme", "pos", "count"]:
            raise ModuBaseformAnalysisError("Frequency CSV header is invalid.")
        for line_number, row in enumerate(reader, start=2):
            subtype = row["source_subtype"]
            morpheme = normalize_form(row["morpheme"])
            pos = row["pos"].strip()
            if subtype not in SOURCE_SUBTYPES:
                raise ModuBaseformAnalysisError(
                    f"Frequency row {line_number} has an invalid source subtype."
                )
            if not morpheme or morpheme != row["morpheme"]:
                raise ModuBaseformAnalysisError(
                    f"Frequency row {line_number} has a noncanonical morpheme."
                )
            try:
                count = int(row["count"])
            except ValueError as exc:
                raise ModuBaseformAnalysisError(
                    f"Frequency row {line_number} has an invalid count."
                ) from exc
            if count < 1:
                raise ModuBaseformAnalysisError(
                    f"Frequency row {line_number} has a non-positive count."
                )
            key = (subtype, morpheme, pos)
            if key in seen:
                raise ModuBaseformAnalysisError("Frequency CSV contains a duplicate key.")
            seen.add(key)
            if pos in TARGET_POS:
                entries.append(FrequencyEntry(subtype, morpheme, pos, count))
    if not entries:
        raise ModuBaseformAnalysisError("Frequency CSV has no target-POS rows.")
    return tuple(entries)


def load_vocabulary(path: Path) -> frozenset[str]:
    """Load the canonical game vocabulary as a normalized set."""
    try:
        words = [normalize_form(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeError) as exc:
        raise ModuBaseformAnalysisError(f"Could not read game vocabulary: {exc}") from exc
    vocabulary = frozenset(word for word in words if word)
    if not vocabulary:
        raise ModuBaseformAnalysisError("Game vocabulary is empty.")
    return vocabulary


def load_answer_candidates(path: Path) -> frozenset[str]:
    """Load normalized words from the answer-candidate CSV."""
    try:
        handle = path.open(encoding="utf-8", newline="")
    except OSError as exc:
        raise ModuBaseformAnalysisError(f"Could not read answer candidates: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "word" not in reader.fieldnames:
            raise ModuBaseformAnalysisError("Answer-candidate CSV has no word column.")
        candidates = frozenset(
            word
            for row in reader
            if (word := normalize_form(row.get("word", "")))
        )
    if not candidates:
        raise ModuBaseformAnalysisError("Answer-candidate CSV is empty.")
    return candidates


def conservative_baseform(
    morpheme: str, pos: str, vocabulary: frozenset[str]
) -> tuple[str, str]:
    """Return a vocabulary-backed lexical form or an explicit non-match status."""
    if pos in RAW_LEXICAL_POS:
        return (morpheme, "matched") if morpheme in vocabulary else ("", "unmatched")
    if pos in BASEFORM_LEXICAL_POS:
        candidate = f"{morpheme}다"
        return (candidate, "matched") if candidate in vocabulary else ("", "review")
    if pos in {"NNP", "IC"}:
        return "", "review"
    return "", "unmatched"


def _rate(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def _coverage(
    counts: Mapping[str, int], game_words: frozenset[str], answer_words: frozenset[str], pos: str
) -> dict[str, Any]:
    total_count = sum(counts.values())
    unique_count = len(counts)

    def membership(forms: Mapping[str, str], lexicon: frozenset[str]) -> dict[str, Any]:
        hit_forms = {form for form, candidate in forms.items() if candidate in lexicon}
        hit_tokens = sum(counts[form] for form in hit_forms)
        return {
            "unique_hits": len(hit_forms),
            "unique_rate": _rate(len(hit_forms), unique_count),
            "token_hits": hit_tokens,
            "token_rate": _rate(hit_tokens, total_count),
        }

    raw_forms = {form: form for form in counts}
    da_forms = {form: f"{form}다" for form in counts}
    conservative_forms = {
        form: canonical
        for form in counts
        if (canonical := conservative_baseform(form, pos, game_words)[0])
    }
    return {
        "unique_forms": unique_count,
        "token_count": total_count,
        "raw_in_game_vocabulary": membership(raw_forms, game_words),
        "plus_da_in_game_vocabulary": membership(da_forms, game_words),
        "raw_in_answer_candidates": membership(raw_forms, answer_words),
        "plus_da_in_answer_candidates": membership(da_forms, answer_words),
        "conservative_in_game_vocabulary": membership(conservative_forms, game_words),
        "conservative_in_answer_candidates": membership(conservative_forms, answer_words),
    }


def _samples(counts: Mapping[str, int], *, sample_size: int = 10) -> dict[str, Any]:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    middle_start = max(0, (len(ordered) - sample_size) // 2)

    def render(items: Iterable[tuple[str, int]]) -> list[dict[str, Any]]:
        return [{"morpheme": form, "count": count} for form, count in items]

    return {
        "top": render(ordered[:sample_size]),
        "middle_rank": render(ordered[middle_start : middle_start + sample_size]),
    }


def _top_entries(
    entries: Iterable[FrequencyEntry], predicate: Any, *, limit: int = 20
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for entry in entries:
        if predicate(entry):
            counts[(entry.morpheme, entry.pos)] += entry.count
    return [
        {"morpheme": form, "pos": pos, "count": count}
        for (form, pos), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )[:limit]
    ]


def _predicate_audit_entries(
    entries: Iterable[FrequencyEntry],
    predicate: Any,
    game_words: frozenset[str],
    answer_words: frozenset[str],
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for entry in entries:
        if predicate(entry):
            counts[(entry.morpheme, entry.pos)] += entry.count
    return [
        {
            "morpheme": form,
            "pos": pos,
            "count": count,
            "plus_da": f"{form}다",
            "raw_in_game_vocabulary": form in game_words,
            "plus_da_in_game_vocabulary": f"{form}다" in game_words,
            "plus_da_in_answer_candidates": f"{form}다" in answer_words,
        }
        for (form, pos), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )[:limit]
    ]


def build_analysis_report(
    entries: tuple[FrequencyEntry, ...],
    game_words: frozenset[str],
    answer_words: frozenset[str],
    *,
    input_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build coverage, POS evidence, collision, and risk-audit sections."""
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    overall: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in entries:
        grouped[(entry.source_subtype, entry.pos)][entry.morpheme] += entry.count
        overall[entry.pos][entry.morpheme] += entry.count

    pos_analysis: dict[str, Any] = {}
    subtype_coverage: dict[str, Any] = {}
    for pos in TARGET_POS:
        counts = overall[pos]
        pos_analysis[pos] = {
            "recommended_role": POS_ROLES[pos],
            "coverage": _coverage(counts, game_words, answer_words, pos),
            "samples": _samples(counts),
        }
    for subtype in SOURCE_SUBTYPES:
        subtype_coverage[subtype] = {
            pos: _coverage(grouped[(subtype, pos)], game_words, answer_words, pos)
            for pos in TARGET_POS
        }

    form_pos: dict[str, set[str]] = defaultdict(set)
    form_counts: Counter[str] = Counter()
    for entry in entries:
        form_pos[entry.morpheme].add(entry.pos)
        form_counts[entry.morpheme] += entry.count
    cross_pos = [form for form, positions in form_pos.items() if len(positions) > 1]
    cross_pos_examples = [
        {
            "morpheme": form,
            "positions": sorted(form_pos[form]),
            "count": form_counts[form],
        }
        for form in sorted(cross_pos, key=lambda form: (-form_counts[form], form))[:50]
    ]

    raw_plus_da_collisions = [
        (form, pos, count)
        for pos in PREDICATE_POS
        for form, count in overall[pos].items()
        if form in game_words and f"{form}다" in game_words
    ]
    collision_examples = [
        {"morpheme": form, "pos": pos, "count": count, "plus_da": f"{form}다"}
        for form, pos, count in sorted(
            raw_plus_da_collisions, key=lambda item: (-item[2], item[0], item[1])
        )[:50]
    ]

    canonical_sources: dict[str, set[tuple[str, str]]] = defaultdict(set)
    canonical_counts: Counter[str] = Counter()
    for pos in TARGET_POS:
        for form, count in overall[pos].items():
            canonical, status = conservative_baseform(form, pos, game_words)
            if status == "matched":
                canonical_sources[canonical].add((form, pos))
                canonical_counts[canonical] += count
    canonical_collisions = [
        canonical for canonical, sources in canonical_sources.items() if len(sources) > 1
    ]

    irregular_stems = frozenset(
        {
            "듣",
            "걷",
            "돕",
            "곱",
            "낫",
            "헛",
            "짓",
            "붓",
            "잇",
            "모르",
            "부르",
            "흐르",
            "이르",
            "푸",
            "끄",
            "쓰",
        }
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": {
            "target_pos": list(TARGET_POS),
            "strategies": {
                "A": "raw morpheme membership",
                "B": "morpheme + 다 membership (measurement only)",
                "C": "POS-gated, game-vocabulary-backed conservative mapping",
            },
            "no_cutoff_or_pool_selection": True,
        },
        "inputs": dict(input_metadata or {}),
        "lexicons": {
            "game_vocabulary_size": len(game_words),
            "answer_candidate_size": len(answer_words),
        },
        "pos_analysis": pos_analysis,
        "subtype_coverage": subtype_coverage,
        "cross_pos": {
            "form_count": len(cross_pos),
            "examples": cross_pos_examples,
        },
        "collision_risks": {
            "raw_and_plus_da_both_in_game_vocabulary_count": len(raw_plus_da_collisions),
            "raw_and_plus_da_examples": collision_examples,
            "canonical_form_multi_source_count": len(canonical_collisions),
            "canonical_form_multi_source_examples": [
                {
                    "canonical": canonical,
                    "sources": [
                        {"morpheme": form, "pos": pos}
                        for form, pos in sorted(canonical_sources[canonical])
                    ],
                    "count": canonical_counts[canonical],
                }
                for canonical in sorted(
                    canonical_collisions,
                    key=lambda value: (-canonical_counts[value], value),
                )[:50]
            ],
        },
        "risk_audit": {
            "irregular_stems": _predicate_audit_entries(
                entries,
                lambda entry: entry.pos in {"VV", "VA"}
                and entry.morpheme in irregular_stems,
                game_words,
                answer_words,
            ),
            "hada_family": _predicate_audit_entries(
                entries,
                lambda entry: entry.pos in PREDICATE_POS
                and entry.morpheme.endswith("하"),
                game_words,
                answer_words,
            ),
            "doeda_family": _predicate_audit_entries(
                entries,
                lambda entry: entry.pos in PREDICATE_POS
                and entry.morpheme.endswith("되"),
                game_words,
                answer_words,
            ),
            "vv_va_plus_da_unmatched": _predicate_audit_entries(
                entries,
                lambda entry: entry.pos in {"VV", "VA"}
                and f"{entry.morpheme}다" not in game_words,
                game_words,
                answer_words,
            ),
            "itda_eopda_gatda": _top_entries(
                entries, lambda entry: entry.morpheme in {"있", "없", "같"}
            ),
            "vx": _top_entries(entries, lambda entry: entry.pos == "VX"),
            "xsv_xsa": _top_entries(
                entries, lambda entry: entry.pos in {"XSV", "XSA"}
            ),
            "nnp_nng_ambiguity": _top_entries(
                entries,
                lambda entry: entry.pos in {"NNP", "NNG"}
                and {"NNP", "NNG"}.issubset(form_pos[entry.morpheme]),
            ),
        },
        "recommended_contract": {
            "normalization": "NFKC+strip",
            "NNG": "raw form iff present in game vocabulary; otherwise unmatched",
            "MAG": "raw form iff present in game vocabulary; otherwise unmatched",
            "VV": "form+다 iff present in game vocabulary; otherwise review",
            "VA": "form+다 iff present in game vocabulary; otherwise review",
            "NNP": "review; never auto-admit as a common lexical candidate",
            "VX": "unmatched; auxiliary use is not an independent lexical count",
            "XSV": "unmatched; left derivational base is absent from the aggregation key",
            "XSA": "unmatched; left derivational base is absent from the aggregation key",
            "other_target_pos": "unmatched, except IC remains review",
            "answer_candidate_use": (
                "answer-candidate membership is reported separately and does not override "
                "the game-vocabulary gate"
            ),
        },
    }


def file_metadata(path: Path) -> dict[str, Any]:
    """Return reproducibility metadata for one analysis input."""
    payload = path.read_bytes()
    return {
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Atomically write a deterministic UTF-8 JSON analysis report."""
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
