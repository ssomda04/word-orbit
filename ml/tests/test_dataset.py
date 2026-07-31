"""Tests for strict evaluation-dataset validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from contextle_eval.dataset import DatasetValidationError, load_dataset, parse_dataset


def _valid_dataset() -> dict[str, object]:
    return {
        "datasetName": "test",
        "version": "1.0.0",
        "language": "ko",
        "description": "test dataset",
        "createdAt": "2026-07-30",
        "groupDefinitions": {
            "veryClose": "close",
            "related": "related",
            "unrelated": "unrelated",
            "surfaceTrap": "surface trap",
        },
        "items": [
            {
                "answer": "정답",
                "veryClose": ["가까움"],
                "related": ["관련"],
                "unrelated": ["무관"],
                "surfaceTrap": ["표면함정"],
            }
        ],
    }


def test_loads_valid_utf8_dataset(tmp_path: Path) -> None:
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(_valid_dataset(), ensure_ascii=False), encoding="utf-8")

    dataset = load_dataset(path)

    assert dataset.dataset_name == "test"
    assert dataset.items[0].answer == "정답"
    assert dataset.items[0].very_close == ("가까움",)
    assert dataset.items[0].related == ("관련",)
    assert dataset.items[0].surface_trap == ("표면함정",)


def test_rejects_missing_required_field() -> None:
    raw = _valid_dataset()
    del raw["version"]

    with pytest.raises(DatasetValidationError, match="version"):
        parse_dataset(raw)


def test_rejects_blank_word() -> None:
    raw = _valid_dataset()
    raw["items"][0]["related"] = ["  "]

    with pytest.raises(DatasetValidationError, match="non-empty string"):
        parse_dataset(raw)


def test_rejects_duplicate_word_across_groups() -> None:
    raw = _valid_dataset()
    raw["items"][0]["surfaceTrap"] = ["가까움"]

    with pytest.raises(DatasetValidationError, match="duplicate word"):
        parse_dataset(raw)
