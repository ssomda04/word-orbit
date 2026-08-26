"""Bounded streaming access to raw NIKL newspaper, dialogue, and online text."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from zipfile import BadZipFile, ZipFile

import ijson

RawSource = Literal["newspaper", "dialogue", "online_ebrw", "online_esrw"]
SOURCE_PREFIX = {
    "newspaper": "N",
    "dialogue": "SARW",
    "online_ebrw": "EBRW",
    "online_esrw": "ESRW",
}


class ModuRawCorpusError(ValueError):
    """Raised when a raw archive or selected text-unit schema is invalid."""


@dataclass(frozen=True, slots=True)
class RawTextUnit:
    source: RawSource
    entry_name: str
    text_id: str
    form: str
    schema_path: str


def _json_entries(archive: ZipFile, source: RawSource) -> list[str]:
    prefix = SOURCE_PREFIX[source]
    candidates = sorted(
        item.filename
        for item in archive.infolist()
        if not item.is_dir()
        and item.filename.lower().endswith(".json")
        and Path(item.filename).name.startswith(prefix)
    )
    if not candidates:
        raise ModuRawCorpusError(f"No {source} JSON entries found in archive.")
    return candidates


def _required_unit(raw: Any, source: RawSource, entry_name: str, path: str) -> RawTextUnit:
    if not isinstance(raw, dict):
        raise ModuRawCorpusError(f"{path} must contain objects.")
    text_id = raw.get("id")
    form = raw.get("form")
    if not isinstance(text_id, str) or not text_id.strip():
        raise ModuRawCorpusError(f"{path}.id must be a non-empty string.")
    if not isinstance(form, str) or not form.strip():
        raise ModuRawCorpusError(f"{path}.form must be a non-empty string.")
    return RawTextUnit(source, entry_name, text_id, form, path)


def _iter_entry_units(
    archive: ZipFile, entry_name: str, source: RawSource
) -> Iterator[RawTextUnit]:
    with archive.open(entry_name) as stream:
        if source == "dialogue":
            path = "document[].utterance[]"
            for raw in ijson.items(stream, "document.item.utterance.item"):
                yield _required_unit(raw, source, entry_name, path)
            return
        path = "document[].paragraph[]"
        for raw in ijson.items(stream, "document.item.paragraph.item"):
            if source != "online_esrw":
                yield _required_unit(raw, source, entry_name, path)
                continue
            sentences = raw.get("sentence") if isinstance(raw, dict) else None
            usable = [
                sentence
                for sentence in sentences or []
                if isinstance(sentence, dict)
                and isinstance(sentence.get("form"), str)
                and sentence["form"].strip()
            ]
            if usable:
                for sentence in usable:
                    yield _required_unit(
                        sentence,
                        source,
                        entry_name,
                        "document[].paragraph[].sentence[]",
                    )
            else:
                yield _required_unit(raw, source, entry_name, path)


def iter_raw_text_units(
    zip_path: Path, *, source: RawSource, limit: int
) -> Iterator[RawTextUnit]:
    """Yield at most ``limit`` units without extracting or loading a whole entry."""
    if source not in SOURCE_PREFIX:
        raise ValueError(f"Unsupported raw source: {source!r}")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    yielded = 0
    try:
        with ZipFile(zip_path) as archive:
            for entry_name in _json_entries(archive, source):
                for unit in _iter_entry_units(archive, entry_name, source):
                    yield unit
                    yielded += 1
                    if yielded >= limit:
                        return
    except FileNotFoundError as exc:
        raise ModuRawCorpusError(f"Raw corpus ZIP not found: {zip_path}") from exc
    except BadZipFile as exc:
        raise ModuRawCorpusError(f"Invalid raw corpus ZIP: {zip_path}") from exc
    except ijson.JSONError as exc:
        raise ModuRawCorpusError(f"Malformed JSON in {zip_path}: {exc}") from exc
    if yielded == 0:
        raise ModuRawCorpusError(f"No usable {source} text units found.")
