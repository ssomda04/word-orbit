"""Stream and validate NIKL Modu morphological-analysis corpus records."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Literal
from zipfile import BadZipFile, ZipFile

try:
    import ijson
    from ijson.common import ObjectBuilder
except ImportError as exc:  # pragma: no cover - exercised only in an incomplete environment
    raise ImportError(
        "Modu corpus streaming requires ijson. Run the inspection script with uv, "
        "or install ijson>=3.4,<4."
    ) from exc

SourceSubtype = Literal["NXMP", "SXMP"]
SOURCE_SUBTYPES: tuple[SourceSubtype, ...] = ("NXMP", "SXMP")


class ModuCorpusError(ValueError):
    """Raised when a Modu MP archive or record cannot be parsed safely."""


@dataclass(frozen=True)
class MorphemeRecord:
    """One NIKL-provided morpheme form and POS label."""

    corpus_id: str
    document_id: str
    sentence_id: str
    mp_id: int
    word_id: int | None
    word_form: str | None
    morpheme: str
    pos: str
    position: int | None
    source_subtype: SourceSubtype


@dataclass(frozen=True)
class ValidationIssue:
    """A recoverable relationship or value problem in one sentence."""

    code: str
    sentence_id: str
    mp_id: int | None
    record_index: int
    detail: str


@dataclass(frozen=True)
class ParsedSentence:
    """Validated records and minimal metadata for one streamed sentence."""

    corpus_id: str
    document_id: str
    sentence_id: str
    source_subtype: SourceSubtype
    sentence_form: str
    original_form: str | None
    word_count: int
    records: tuple[MorphemeRecord, ...]
    issues: tuple[ValidationIssue, ...]


def select_mp_entry(archive: ZipFile, source_subtype: SourceSubtype) -> str:
    """Return the archive's unique subtype JSON entry without extracting it."""
    if source_subtype not in SOURCE_SUBTYPES:
        raise ValueError(f"Unsupported MP source subtype: {source_subtype!r}")
    candidates = sorted(
        info.filename
        for info in archive.infolist()
        if not info.is_dir()
        and Path(info.filename).name.startswith(source_subtype)
        and info.filename.lower().endswith(".json")
    )
    if len(candidates) != 1:
        raise ModuCorpusError(
            f"Expected exactly one {source_subtype} JSON entry, "
            f"found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def select_nxmp_entry(archive: ZipFile) -> str:
    """Return the archive's unique NXMP JSON entry without extracting it."""
    return select_mp_entry(archive, "NXMP")


def select_sxmp_entry(archive: ZipFile) -> str:
    """Return the archive's unique SXMP JSON entry without extracting it."""
    return select_mp_entry(archive, "SXMP")


def iter_mp_sentences(
    zip_path: Path,
    *,
    source_subtype: SourceSubtype,
    limit_sentences: int | None = None,
    entry_name: str | None = None,
) -> Iterator[ParsedSentence]:
    """Stream validated NXMP or SXMP sentences directly from a ZIP entry.

    Only the current sentence object is materialized. The ZIP is never extracted,
    and neither the complete JSON document nor a complete corpus document is loaded.
    """
    if source_subtype not in SOURCE_SUBTYPES:
        raise ValueError(f"Unsupported MP source subtype: {source_subtype!r}")
    if limit_sentences is not None and limit_sentences < 1:
        raise ValueError("limit_sentences must be at least 1 or None")

    try:
        with ZipFile(zip_path, "r") as archive:
            selected = entry_name or select_mp_entry(archive, source_subtype)
            selected_name = Path(selected).name
            if not selected_name.startswith(source_subtype) or not selected_name.lower().endswith(
                ".json"
            ):
                raise ModuCorpusError(f"Entry is not {source_subtype} JSON: {selected}")
            try:
                with archive.open(selected, "r") as stream:
                    yield from _iter_sentence_stream(
                        stream, limit_sentences, source_subtype
                    )
            except KeyError as exc:
                raise ModuCorpusError(f"ZIP entry not found: {selected}") from exc
    except FileNotFoundError as exc:
        raise ModuCorpusError(f"Corpus ZIP not found: {zip_path}") from exc
    except BadZipFile as exc:
        raise ModuCorpusError(f"Invalid ZIP archive: {zip_path}") from exc


def iter_nxmp_sentences(
    zip_path: Path,
    *,
    limit_sentences: int | None = None,
    entry_name: str | None = None,
) -> Iterator[ParsedSentence]:
    """Stream validated newspaper MP sentences using the shared parser."""
    yield from iter_mp_sentences(
        zip_path,
        source_subtype="NXMP",
        limit_sentences=limit_sentences,
        entry_name=entry_name,
    )


def iter_sxmp_sentences(
    zip_path: Path,
    *,
    limit_sentences: int | None = None,
    entry_name: str | None = None,
) -> Iterator[ParsedSentence]:
    """Stream validated spoken-language MP sentences using the shared parser."""
    yield from iter_mp_sentences(
        zip_path,
        source_subtype="SXMP",
        limit_sentences=limit_sentences,
        entry_name=entry_name,
    )


def _iter_sentence_stream(
    stream: BinaryIO,
    limit_sentences: int | None,
    source_subtype: SourceSubtype,
) -> Iterator[ParsedSentence]:
    corpus_id: str | None = None
    document_id: str | None = None
    builder: ObjectBuilder | None = None
    builder_depth = 0
    yielded = 0

    try:
        events = ijson.parse(stream)
        for prefix, event, value in events:
            if builder is not None:
                builder.event(event, value)
                if event in {"start_map", "start_array"}:
                    builder_depth += 1
                elif event in {"end_map", "end_array"}:
                    builder_depth -= 1
                if builder_depth == 0:
                    if corpus_id is None or document_id is None:
                        raise ModuCorpusError(
                            "Encountered sentence before root id or document id."
                        )
                    if not corpus_id.startswith(source_subtype):
                        raise ModuCorpusError(
                            f"{source_subtype} entry has mismatched root id: "
                            f"{corpus_id!r}."
                        )
                    yield _parse_sentence(
                        builder.value, corpus_id, document_id, source_subtype
                    )
                    yielded += 1
                    builder = None
                    if limit_sentences is not None and yielded >= limit_sentences:
                        return
                continue

            if prefix == "id" and event == "string" and value.strip():
                corpus_id = value
            elif prefix == "document.item" and event == "start_map":
                document_id = None
            elif prefix == "document.item.id" and event == "string" and value.strip():
                document_id = value
            elif prefix == "document.item.sentence.item" and event == "start_map":
                builder = ObjectBuilder()
                builder.event(event, value)
                builder_depth = 1
    except ijson.JSONError as exc:
        raise ModuCorpusError(f"Malformed {source_subtype} JSON: {exc}") from exc

    if builder is not None:
        raise ModuCorpusError(
            f"Malformed {source_subtype} JSON: incomplete sentence object."
        )
    if corpus_id is None:
        raise ModuCorpusError(
            f"{source_subtype} root is missing non-empty string field 'id'."
        )
    if not corpus_id.startswith(source_subtype):
        raise ModuCorpusError(
            f"{source_subtype} entry has mismatched root id: {corpus_id!r}."
        )
    if yielded == 0:
        raise ModuCorpusError(f"{source_subtype} entry contains no sentences.")


def _required_string(raw: Mapping[str, Any], field: str, location: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ModuCorpusError(f"{location}.{field} must be a non-empty string.")
    return value


def _required_list(raw: Mapping[str, Any], field: str, location: str) -> list[Any]:
    value = raw.get(field)
    if not isinstance(value, list):
        raise ModuCorpusError(f"{location}.{field} must be an array.")
    return value


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _parse_sentence(
    raw: Any,
    corpus_id: str,
    document_id: str,
    source_subtype: SourceSubtype,
) -> ParsedSentence:
    if not isinstance(raw, Mapping):
        raise ModuCorpusError("document[].sentence[] must contain objects.")

    sentence_id = _required_string(raw, "id", "sentence")
    sentence_form = _required_string(raw, "form", f"sentence {sentence_id}")
    original = raw.get("original_form")
    if original is not None and not isinstance(original, str):
        raise ModuCorpusError(
            f"sentence {sentence_id}.original_form must be a string or null."
        )
    words = _required_list(raw, "word", f"sentence {sentence_id}")
    morphemes = _required_list(raw, "MP", f"sentence {sentence_id}")

    word_forms: dict[int, str] = {}
    for index, word in enumerate(words):
        location = f"sentence {sentence_id}.word[{index}]"
        if not isinstance(word, Mapping):
            raise ModuCorpusError(f"{location} must be an object.")
        word_id = _optional_int(word.get("id"))
        if word_id is None:
            raise ModuCorpusError(f"{location}.id must be an integer.")
        if word_id in word_forms:
            raise ModuCorpusError(f"{location}.id duplicates word id {word_id}.")
        word_forms[word_id] = _required_string(word, "form", location)

    records: list[MorphemeRecord] = []
    issues: list[ValidationIssue] = []
    seen_mp_ids: set[int] = set()
    expected_positions: dict[int, int] = {}
    for index, mp in enumerate(morphemes):
        location = f"sentence {sentence_id}.MP[{index}]"
        if not isinstance(mp, Mapping):
            raise ModuCorpusError(f"{location} must be an object.")
        mp_id = _optional_int(mp.get("id"))
        if mp_id is None:
            raise ModuCorpusError(f"{location}.id must be an integer.")
        morpheme = mp.get("form")
        pos = mp.get("label")
        word_id = _optional_int(mp.get("word_id"))
        position = _optional_int(mp.get("position"))

        if mp_id in seen_mp_ids:
            issues.append(
                ValidationIssue(
                    "duplicate_mp_id",
                    sentence_id,
                    mp_id,
                    index,
                    f"duplicate MP id {mp_id}",
                )
            )
        seen_mp_ids.add(mp_id)
        if not isinstance(morpheme, str) or not morpheme.strip():
            issues.append(
                ValidationIssue(
                    "empty_morpheme",
                    sentence_id,
                    mp_id,
                    index,
                    "form is missing or blank",
                )
            )
            morpheme = morpheme if isinstance(morpheme, str) else ""
        if not isinstance(pos, str) or not pos.strip():
            issues.append(
                ValidationIssue(
                    "empty_pos",
                    sentence_id,
                    mp_id,
                    index,
                    "label is missing or blank",
                )
            )
            pos = pos if isinstance(pos, str) else ""
        if word_id is None:
            issues.append(
                ValidationIssue(
                    "missing_word_id",
                    sentence_id,
                    mp_id,
                    index,
                    "word_id is not an integer",
                )
            )
        elif word_id not in word_forms:
            issues.append(
                ValidationIssue(
                    "orphan_word_id",
                    sentence_id,
                    mp_id,
                    index,
                    f"unknown word_id {word_id}",
                )
            )

        if word_id is not None:
            expected = expected_positions.get(word_id, 0) + 1
            if position != expected:
                issues.append(
                    ValidationIssue(
                        "position_order",
                        sentence_id,
                        mp_id,
                        index,
                        f"word_id {word_id}: expected position {expected}, got {position!r}",
                    )
                )
            expected_positions[word_id] = position if position is not None else expected

        records.append(
            MorphemeRecord(
                corpus_id=corpus_id,
                document_id=document_id,
                sentence_id=sentence_id,
                mp_id=mp_id,
                word_id=word_id,
                word_form=word_forms.get(word_id) if word_id is not None else None,
                morpheme=morpheme,
                pos=pos,
                position=position,
                source_subtype=source_subtype,
            )
        )

    return ParsedSentence(
        corpus_id=corpus_id,
        document_id=document_id,
        sentence_id=sentence_id,
        source_subtype=source_subtype,
        sentence_form=sentence_form,
        original_form=original,
        word_count=len(words),
        records=tuple(records),
        issues=tuple(issues),
    )
