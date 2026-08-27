"""Stream Korean game-word headwords from a Wikimedia XML dump."""

from __future__ import annotations

import bz2
import contextlib
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

KOREAN_LANGUAGE_HEADING = "한국어"

# The nine commonly recognized Korean parts of speech, plus heading variants
# used for noun and predicate subtypes on Korean Wiktionary.
NON_PREDICATE_POS = {
    "명사",
    "보통 명사",
    "보통명사",
    "고유 명사",
    "고유명사",
    "의존 명사",
    "의존명사",
    "대명사",
    "수사",
    "관형사",
    "부사",
    "감탄사",
}
PREDICATE_POS = {
    "동사",
    "자동사",
    "타동사",
    "보조 동사",
    "보조동사",
    "조동사",
    "형용사",
    "보조 형용사",
    "보조형용사",
}
ALLOWED_POS = NON_PREDICATE_POS | PREDICATE_POS
PREDICATE_INFLECTION_HEADINGS = {
    "동사 활용형",
    "동사활용형",
    "형용사 활용형",
    "형용사활용형",
    "활용형",
}
INFLECTED_PREDICATE_ENDINGS = ("았다", "었다", "였다", "겠다", "습니다", "ㅂ니다")

HEADING_RE = re.compile(
    r"^(?P<marks>={2,6})[ \t]*(?P<label>[^=\r\n]+?)[ \t]*(?P=marks)[ \t]*$",
    re.MULTILINE,
)
POS_NUMBER_RE = re.compile(r"\s*(?:\(?\d+\)?)\s*$")
REDIRECT_RE = re.compile(r"^\s*#(?:redirect|넘겨주기)\b", re.IGNORECASE)

EXCLUSION_REASONS = (
    "non_main_namespace",
    "redirect",
    "missing_title",
    "missing_text",
    "no_korean_section",
    "no_allowed_pos",
    "empty_after_normalization",
    "contains_internal_whitespace",
    "contains_number",
    "contains_special_character",
    "contains_non_hangul_character",
    "non_lemma_predicate",
    "duplicate",
)


class WiktionaryExtractionError(RuntimeError):
    """Raised when the dump cannot be read or the output cannot be written."""


@dataclass(frozen=True)
class WiktionaryPage:
    """The page fields needed by the headword filter."""

    title: str | None
    namespace: int | None
    text: str | None
    is_redirect: bool = False


@dataclass
class ExtractionStats:
    """Deterministic extraction counts, including one reason per excluded page."""

    pages_seen: int = 0
    normalized_titles: int = 0
    unique_words: int = 0
    excluded: Counter[str] = field(default_factory=Counter)

    @property
    def excluded_pages(self) -> int:
        return sum(self.excluded.values())

    def as_dict(self) -> dict[str, int | dict[str, int]]:
        return {
            "pages_seen": self.pages_seen,
            "unique_words": self.unique_words,
            "normalized_titles": self.normalized_titles,
            "excluded_pages": self.excluded_pages,
            "excluded_by_reason": {
                reason: self.excluded.get(reason, 0) for reason in EXCLUSION_REASONS
            },
        }


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _direct_child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in element if _local_name(child.tag) == name), None)


def _revision_text(revision: ET.Element | None) -> str | None:
    if revision is None:
        return None
    direct_text = _direct_child(revision, "text")
    if direct_text is not None:
        return direct_text.text
    for descendant in revision.iter():
        if _local_name(descendant.tag) == "text":
            return descendant.text
    return None


def iter_wiktionary_pages(stream: BinaryIO) -> Iterator[WiktionaryPage]:
    """Yield namespace-agnostic page records while releasing parsed XML nodes."""
    context = ET.iterparse(stream, events=("start", "end"))
    _, root = next(context)
    for event, element in context:
        if event != "end" or _local_name(element.tag) != "page":
            continue

        title_node = _direct_child(element, "title")
        namespace_node = _direct_child(element, "ns")
        revision_node = _direct_child(element, "revision")
        redirect_node = _direct_child(element, "redirect")

        try:
            namespace = int(namespace_node.text) if namespace_node is not None else None
        except (TypeError, ValueError):
            namespace = None

        text = _revision_text(revision_node)
        yield WiktionaryPage(
            title=title_node.text if title_node is not None else None,
            namespace=namespace,
            text=text,
            is_redirect=redirect_node is not None or bool(text and REDIRECT_RE.match(text)),
        )
        element.clear()
        root.clear()


def _normalize_heading(label: str) -> str:
    normalized = unicodedata.normalize("NFKC", label).strip().lower()
    return re.sub(r"\s+", " ", normalized)


def korean_sections(text: str) -> tuple[str, ...]:
    """Return only explicitly marked Korean level-two language sections."""
    headings = list(HEADING_RE.finditer(text))
    sections: list[str] = []
    for index, heading in enumerate(headings):
        if len(heading.group("marks")) != 2:
            continue
        if _normalize_heading(heading.group("label")) != KOREAN_LANGUAGE_HEADING:
            continue

        end = len(text)
        for following in headings[index + 1 :]:
            if len(following.group("marks")) == 2:
                end = following.start()
                break
        sections.append(text[heading.end() : end])
    return tuple(sections)


def detected_parts_of_speech(sections: tuple[str, ...]) -> frozenset[str]:
    """Return exact major-POS headings found inside Korean sections."""
    detected: set[str] = set()
    for section in sections:
        for heading in HEADING_RE.finditer(section):
            if len(heading.group("marks")) < 3:
                continue
            label = _normalize_heading(heading.group("label"))
            label = POS_NUMBER_RE.sub("", label).strip()
            if label in ALLOWED_POS:
                detected.add(label)
    return frozenset(detected)


def _contains_predicate_inflection_heading(sections: tuple[str, ...]) -> bool:
    for section in sections:
        for heading in HEADING_RE.finditer(section):
            if len(heading.group("marks")) < 3:
                continue
            label = _normalize_heading(heading.group("label"))
            label = POS_NUMBER_RE.sub("", label).strip()
            if label in PREDICATE_INFLECTION_HEADINGS:
                return True
    return False


def _is_hangul_syllable(character: str) -> bool:
    codepoint = ord(character)
    return 0xAC00 <= codepoint <= 0xD7A3


def _title_exclusion(title: str) -> str | None:
    if not title:
        return "empty_after_normalization"
    if any(character.isspace() for character in title):
        return "contains_internal_whitespace"
    if any(character.isnumeric() for character in title):
        return "contains_number"
    if any(unicodedata.category(character)[0] in {"P", "S"} for character in title):
        return "contains_special_character"
    if any(not _is_hangul_syllable(character) for character in title):
        return "contains_non_hangul_character"
    return None


def _classify_page(page: WiktionaryPage) -> tuple[str | None, str | None, bool]:
    if page.namespace != 0:
        return None, "non_main_namespace", False
    if page.is_redirect:
        return None, "redirect", False
    if page.title is None:
        return None, "missing_title", False
    if page.text is None:
        return None, "missing_text", False

    sections = korean_sections(page.text)
    if not sections:
        return None, "no_korean_section", False
    parts_of_speech = detected_parts_of_speech(sections)
    if not parts_of_speech:
        if _contains_predicate_inflection_heading(sections):
            return None, "non_lemma_predicate", False
        return None, "no_allowed_pos", False

    title = unicodedata.normalize("NFKC", page.title).strip()
    changed = title != page.title
    exclusion = _title_exclusion(title)
    if exclusion is not None:
        return None, exclusion, changed

    if parts_of_speech <= PREDICATE_POS and (
        not title.endswith("다") or title.endswith(INFLECTED_PREDICATE_ENDINGS)
    ):
        return None, "non_lemma_predicate", changed
    return title, None, changed


def extract_headwords(
    pages: Iterable[WiktionaryPage],
) -> tuple[list[str], ExtractionStats]:
    """Filter, normalize, deduplicate, and sort Korean headwords."""
    stats = ExtractionStats()
    words: set[str] = set()
    for page in pages:
        stats.pages_seen += 1
        word, exclusion, normalized = _classify_page(page)
        if normalized:
            stats.normalized_titles += 1
        if exclusion is not None:
            stats.excluded[exclusion] += 1
            continue
        assert word is not None
        if word in words:
            stats.excluded["duplicate"] += 1
            continue
        words.add(word)

    result = sorted(words)
    stats.unique_words = len(result)
    return result, stats


def extract_dump(dump_path: Path, output_path: Path) -> ExtractionStats:
    """Read a bz2 MediaWiki dump and atomically replace the UTF-8 word list."""
    if not dump_path.is_file():
        raise WiktionaryExtractionError(f"Wiktionary dump file not found: {dump_path}")

    try:
        with bz2.open(dump_path, "rb") as stream:
            words, stats = extract_headwords(iter_wiktionary_pages(stream))
    except (OSError, EOFError, ET.ParseError) as exc:
        raise WiktionaryExtractionError(
            f"Could not read Wiktionary dump {dump_path}: {exc}"
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    try:
        payload = "".join(f"{word}\n" for word in words).encode("utf-8")
        temporary_path.write_bytes(payload)
        temporary_path.replace(output_path)
    except OSError as exc:
        with contextlib.suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise WiktionaryExtractionError(f"Could not write word list {output_path}: {exc}") from exc
    return stats


def format_statistics(stats: ExtractionStats) -> str:
    """Render stable, human-readable extraction and exclusion counts."""
    lines = [
        f"Pages scanned: {stats.pages_seen}",
        f"Unique headwords: {stats.unique_words}",
        f"Normalized titles: {stats.normalized_titles}",
        f"Excluded pages: {stats.excluded_pages}",
        "Excluded by reason:",
    ]
    lines.extend(f"  {reason}: {stats.excluded.get(reason, 0)}" for reason in EXCLUSION_REASONS)
    return "\n".join(lines)
