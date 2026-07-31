"""Load and validate the versioned word-similarity evaluation dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GROUPS = ("veryClose", "related", "unrelated", "surfaceTrap")
REQUIRED_METADATA = (
    "datasetName",
    "version",
    "language",
    "description",
    "groupDefinitions",
    "items",
)


class DatasetValidationError(ValueError):
    """Raised when an evaluation dataset cannot be used safely."""


@dataclass(frozen=True)
class EvaluationItem:
    """One answer and its grouped candidate words."""

    answer: str
    very_close: tuple[str, ...]
    related: tuple[str, ...]
    unrelated: tuple[str, ...]
    surface_trap: tuple[str, ...]

    def groups(self) -> dict[str, tuple[str, ...]]:
        """Return candidate groups using their JSON field names."""
        return {
            "veryClose": self.very_close,
            "related": self.related,
            "unrelated": self.unrelated,
            "surfaceTrap": self.surface_trap,
        }


@dataclass(frozen=True)
class EvaluationDataset:
    """Validated evaluation dataset and reproducibility metadata."""

    dataset_name: str
    version: str
    language: str
    description: str
    timestamp_field: str
    timestamp_value: str
    group_definitions: dict[str, str]
    items: tuple[EvaluationItem, ...]


def _non_blank_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{location} must be a non-empty string.")
    return value.strip()


def _word_list(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise DatasetValidationError(f"{location} must be a non-empty array of words.")
    return tuple(_non_blank_string(word, f"{location}[{index}]") for index, word in enumerate(value))


def _parse_item(raw: Any, index: int) -> EvaluationItem:
    location = f"items[{index}]"
    if not isinstance(raw, dict):
        raise DatasetValidationError(f"{location} must be an object.")

    missing = [field for field in ("answer", *GROUPS) if field not in raw]
    if missing:
        raise DatasetValidationError(f"{location} is missing required fields: {', '.join(missing)}.")

    answer = _non_blank_string(raw["answer"], f"{location}.answer")
    groups = {group: _word_list(raw[group], f"{location}.{group}") for group in GROUPS}

    seen: dict[str, str] = {answer: "answer"}
    for group, words in groups.items():
        for word in words:
            if word in seen:
                raise DatasetValidationError(
                    f"{location} contains duplicate word {word!r} in {seen[word]} and {group}."
                )
            seen[word] = group

    return EvaluationItem(
        answer=answer,
        very_close=groups["veryClose"],
        related=groups["related"],
        unrelated=groups["unrelated"],
        surface_trap=groups["surfaceTrap"],
    )


def parse_dataset(raw: Any) -> EvaluationDataset:
    """Validate an already-decoded dataset object."""
    if not isinstance(raw, dict):
        raise DatasetValidationError("Dataset root must be a JSON object.")

    missing = [field for field in REQUIRED_METADATA if field not in raw]
    if missing:
        raise DatasetValidationError(
            f"Dataset is missing required fields: {', '.join(missing)}."
        )
    if "createdAt" not in raw and "updatedAt" not in raw:
        raise DatasetValidationError("Dataset must include createdAt or updatedAt.")

    definitions = raw["groupDefinitions"]
    if not isinstance(definitions, dict):
        raise DatasetValidationError("groupDefinitions must be an object.")
    missing_definitions = [group for group in GROUPS if group not in definitions]
    if missing_definitions:
        raise DatasetValidationError(
            "groupDefinitions is missing: " + ", ".join(missing_definitions) + "."
        )
    parsed_definitions = {
        group: _non_blank_string(definitions[group], f"groupDefinitions.{group}")
        for group in GROUPS
    }

    raw_items = raw["items"]
    if not isinstance(raw_items, list) or not raw_items:
        raise DatasetValidationError("items must be a non-empty array.")
    items = tuple(_parse_item(item, index) for index, item in enumerate(raw_items))
    answers = [item.answer for item in items]
    duplicate_answers = sorted({answer for answer in answers if answers.count(answer) > 1})
    if duplicate_answers:
        raise DatasetValidationError(
            "Answer words must be unique; duplicates: " + ", ".join(duplicate_answers) + "."
        )

    timestamp_field = "updatedAt" if "updatedAt" in raw else "createdAt"
    return EvaluationDataset(
        dataset_name=_non_blank_string(raw["datasetName"], "datasetName"),
        version=_non_blank_string(raw["version"], "version"),
        language=_non_blank_string(raw["language"], "language"),
        description=_non_blank_string(raw["description"], "description"),
        timestamp_field=timestamp_field,
        timestamp_value=_non_blank_string(raw[timestamp_field], timestamp_field),
        group_definitions=parsed_definitions,
        items=items,
    )


def load_dataset(path: Path) -> EvaluationDataset:
    """Read and validate a UTF-8 JSON dataset with actionable errors."""
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise DatasetValidationError(
            f"Dataset file not found: {path}. Pass an existing file with --dataset."
        ) from exc
    except OSError as exc:
        raise DatasetValidationError(f"Could not read dataset {path}: {exc}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}."
        ) from exc
    return parse_dataset(raw)
