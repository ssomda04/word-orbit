"""Build reviewable answer candidates from the game vocabulary and Wiktionary."""

from __future__ import annotations

import bz2
import contextlib
import csv
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from contextle_eval.wiktionary_words import (
    ALLOWED_POS,
    WiktionaryExtractionError,
    _title_exclusion,
    detected_parts_of_speech,
    iter_wiktionary_pages,
    korean_sections,
)

DEFAULT_LONG_WORD_LENGTH = 8
PROPER_NOUN_POS = frozenset({"고유 명사", "고유명사"})
GENERAL_LEXICAL_POS = frozenset(ALLOWED_POS - PROPER_NOUN_POS)

# These are explicit subject labels used by Korean Wiktionary, not inferred topics.
TECHNICAL_DOMAINS = frozenset(
    {
        "건축",
        "경제",
        "공학",
        "광학",
        "군사",
        "금융",
        "기상",
        "농업",
        "동물",
        "동물학",
        "물리",
        "법률",
        "법의학",
        "생물",
        "생물학",
        "수학",
        "식물",
        "식물학",
        "약학",
        "언어학",
        "의학",
        "전기",
        "전자",
        "전산학",
        "지리",
        "지질",
        "천문",
        "천문학",
        "컴퓨터",
        "한의학",
        "해부학",
        "행정",
        "화학",
    }
)

CSV_FIELDS = (
    "word",
    "pos",
    "length",
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
)

LABEL_TEMPLATE_RE = re.compile(
    r"\{\{\s*(?:라벨|lb|tlb)\s*\|(?P<arguments>[^{}\r\n]*)\}\}", re.IGNORECASE
)
ALTERNATIVE_FORM_TEMPLATE_RE = re.compile(
    r"\{\{\s*alternative form of\s*\|(?P<arguments>[^{}\r\n]*)\}\}",
    re.IGNORECASE,
)
CATEGORY_RE = re.compile(
    r"\[\[\s*(?:분류|category)\s*:\s*(?P<category>[^\]|\r\n]+)", re.IGNORECASE
)


class AnswerCandidateError(RuntimeError):
    """Raised when candidate input cannot be read or output cannot be written."""


@dataclass(frozen=True, slots=True)
class QualityMetadata:
    """Explicit quality metadata found in one Korean Wiktionary entry."""

    labels: frozenset[str]
    domain_labels: frozenset[str]
    is_archaic: bool
    is_dialect: bool
    is_historical: bool
    has_explicit_rare_label: bool

    @property
    def is_technical(self) -> bool:
        return bool(self.domain_labels)


@dataclass(frozen=True, slots=True)
class AnswerCandidate:
    """One CSV row retained for human answer-pool review."""

    word: str
    parts_of_speech: frozenset[str]
    quality: QualityMetadata
    review_reasons: tuple[str, ...]

    def as_csv_row(self) -> dict[str, str | int]:
        is_proper_noun = bool(self.parts_of_speech & PROPER_NOUN_POS)
        is_general = bool(self.parts_of_speech & GENERAL_LEXICAL_POS)
        return {
            "word": self.word,
            "pos": "|".join(sorted(self.parts_of_speech)),
            "length": len(self.word),
            "is_proper_noun": _csv_bool(is_proper_noun),
            "is_general_lexical_pos": _csv_bool(is_general),
            "is_archaic": _csv_bool(self.quality.is_archaic),
            "is_dialect": _csv_bool(self.quality.is_dialect),
            "is_historical": _csv_bool(self.quality.is_historical),
            "is_technical": _csv_bool(self.quality.is_technical),
            "wiktionary_labels": "|".join(sorted(self.quality.labels)),
            "domain_labels": "|".join(sorted(self.quality.domain_labels)),
            "review_required": _csv_bool(bool(self.review_reasons)),
            "review_reason": "|".join(self.review_reasons),
        }


@dataclass
class CandidateStats:
    """Deterministic candidate, exclusion, POS, and review counts."""

    vocabulary_words: int = 0
    candidate_words: int = 0
    review_required_words: int = 0
    excluded: Counter[str] = field(default_factory=Counter)
    candidate_pos: Counter[str] = field(default_factory=Counter)
    review_reasons: Counter[str] = field(default_factory=Counter)

    @property
    def excluded_words(self) -> int:
        return sum(self.excluded.values())


@dataclass
class _EntryMetadata:
    parts_of_speech: set[str] = field(default_factory=set)
    sections: list[str] = field(default_factory=list)


def _csv_bool(value: bool) -> str:
    return "true" if value else "false"


def _normalize_marker(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip().lower())


def _template_labels(text: str) -> set[str]:
    labels: set[str] = set()
    for match in LABEL_TEMPLATE_RE.finditer(text):
        arguments = [
            _normalize_marker(part) for part in match.group("arguments").split("|")
        ]
        if not arguments or arguments[0] not in {"ko", "한국어"}:
            continue
        labels.update(
            argument for argument in arguments[1:] if argument and "=" not in argument
        )
    return labels


def _categories(text: str) -> set[str]:
    return {
        _normalize_marker(match.group("category"))
        for match in CATEGORY_RE.finditer(text)
        if match.group("category").strip()
    }


def _has_explicit_dialect_form(text: str) -> bool:
    for match in ALTERNATIVE_FORM_TEMPLATE_RE.finditer(text):
        arguments = [
            _normalize_marker(part) for part in match.group("arguments").split("|")
        ]
        if (
            arguments
            and arguments[0] in {"ko", "한국어"}
            and any("방언" in argument for argument in arguments[1:])
        ):
            return True
    return False


def _matching_domains(labels: set[str], categories: set[str]) -> frozenset[str]:
    searchable = set(labels)
    for category in categories:
        for prefix in ("한국어 ", "표준어 "):
            if category.startswith(prefix):
                searchable.add(category.removeprefix(prefix))
    return frozenset(
        domain
        for domain in TECHNICAL_DOMAINS
        if any(domain in marker for marker in searchable)
    )


def extract_quality_metadata(sections: tuple[str, ...]) -> QualityMetadata:
    """Read only explicit labels and categories from Korean language sections."""
    text = "\n".join(sections)
    labels = _template_labels(text)
    categories = _categories(text)
    combined = labels | categories
    domain_labels = _matching_domains(labels, categories)
    return QualityMetadata(
        labels=frozenset(labels),
        domain_labels=domain_labels,
        is_archaic=any("고어" in marker or "옛말" in marker for marker in combined),
        is_dialect=(
            any("방언" in marker for marker in combined)
            or _has_explicit_dialect_form(text)
        ),
        is_historical=(
            "역사" in labels
            or any(
                category == "한국어 역사" or category.startswith("한국어 역사 ")
                for category in categories
            )
        ),
        has_explicit_rare_label=any(
            marker in {"드물게", "드문", "희귀", "희귀어"} for marker in combined
        ),
    )


def _load_vocabulary(path: Path) -> tuple[str, ...]:
    try:
        raw_words = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError as exc:
        raise AnswerCandidateError(f"Vocabulary file not found: {path}") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise AnswerCandidateError(
            f"Could not read vocabulary file {path}: {exc}"
        ) from exc

    unique: dict[str, None] = {}
    for raw_word in raw_words:
        word = unicodedata.normalize("NFKC", raw_word).strip()
        if word:
            unique.setdefault(word, None)
    return tuple(unique)


def _collect_entry_metadata(
    dump_path: Path, vocabulary: set[str]
) -> dict[str, _EntryMetadata]:
    metadata: dict[str, _EntryMetadata] = {}
    try:
        with bz2.open(dump_path, "rb") as stream:
            for page in iter_wiktionary_pages(stream):
                if (
                    page.namespace != 0
                    or page.is_redirect
                    or page.title is None
                    or page.text is None
                ):
                    continue
                word = unicodedata.normalize("NFKC", page.title).strip()
                if word not in vocabulary:
                    continue
                sections = korean_sections(page.text)
                parts_of_speech = detected_parts_of_speech(sections)
                if not sections:
                    continue
                entry = metadata.setdefault(word, _EntryMetadata())
                entry.parts_of_speech.update(parts_of_speech)
                entry.sections.extend(sections)
    except (OSError, EOFError, ET.ParseError, WiktionaryExtractionError) as exc:
        raise AnswerCandidateError(
            f"Could not read Wiktionary dump {dump_path}: {exc}"
        ) from exc
    return metadata


def _review_reasons(
    word: str, quality: QualityMetadata, long_word_length: int
) -> tuple[str, ...]:
    reasons: list[str] = []
    if len(word) >= long_word_length:
        reasons.append("long_word")
    if quality.is_technical:
        reasons.append("technical_term")
    if quality.is_archaic:
        reasons.append("archaic")
    if quality.is_dialect:
        reasons.append("dialect")
    if quality.is_historical:
        reasons.append("historical_term")
    if quality.has_explicit_rare_label:
        reasons.append("explicit_rare_label")
    return tuple(reasons)


def build_answer_candidates(
    dump_path: Path,
    vocabulary_path: Path,
    output_path: Path,
    *,
    long_word_length: int = DEFAULT_LONG_WORD_LENGTH,
) -> CandidateStats:
    """Build a candidate CSV without changing or filtering the game vocabulary file."""
    if long_word_length < 1:
        raise AnswerCandidateError("Long-word review length must be at least 1.")
    if not dump_path.is_file():
        raise AnswerCandidateError(f"Wiktionary dump file not found: {dump_path}")

    vocabulary = _load_vocabulary(vocabulary_path)
    metadata = _collect_entry_metadata(dump_path, set(vocabulary))
    stats = CandidateStats(vocabulary_words=len(vocabulary))
    candidates: list[AnswerCandidate] = []

    for word in vocabulary:
        title_exclusion = _title_exclusion(word)
        if title_exclusion is not None:
            stats.excluded[title_exclusion] += 1
            continue
        entry = metadata.get(word)
        if entry is None:
            stats.excluded["missing_wiktionary_metadata"] += 1
            continue
        parts_of_speech = frozenset(entry.parts_of_speech)
        if parts_of_speech & PROPER_NOUN_POS:
            stats.excluded["proper_noun"] += 1
            continue
        if not parts_of_speech & GENERAL_LEXICAL_POS:
            stats.excluded["not_general_lexical_pos"] += 1
            continue

        quality = extract_quality_metadata(tuple(entry.sections))
        review_reasons = _review_reasons(word, quality, long_word_length)
        candidate = AnswerCandidate(word, parts_of_speech, quality, review_reasons)
        candidates.append(candidate)
        stats.candidate_pos.update(parts_of_speech)
        stats.review_reasons.update(review_reasons)
        if review_reasons:
            stats.review_required_words += 1

    stats.candidate_words = len(candidates)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file, fieldnames=CSV_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(candidate.as_csv_row() for candidate in candidates)
        temporary_path.replace(output_path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise AnswerCandidateError(
            f"Could not write candidate CSV {output_path}: {exc}"
        ) from exc
    return stats


def format_candidate_statistics(stats: CandidateStats) -> str:
    """Render deterministic summary counts for the candidate-generation CLI."""
    lines = [
        f"Vocabulary words: {stats.vocabulary_words}",
        f"Excluded words: {stats.excluded_words}",
        f"Answer candidates: {stats.candidate_words}",
        f"Review required: {stats.review_required_words}",
        "Excluded by reason:",
    ]
    lines.extend(
        f"  {reason}: {count}" for reason, count in sorted(stats.excluded.items())
    )
    lines.append("Candidate POS counts (multi-label):")
    lines.extend(
        f"  {part}: {count}" for part, count in sorted(stats.candidate_pos.items())
    )
    lines.append("Review reasons (multi-label):")
    lines.extend(
        f"  {reason}: {count}" for reason, count in sorted(stats.review_reasons.items())
    )
    return "\n".join(lines)
