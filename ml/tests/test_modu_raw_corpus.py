"""Tests for bounded raw Modu corpus text-unit streaming."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from contextle_eval.modu_raw_corpus import iter_raw_text_units


def _write_archive(tmp_path: Path, entries: dict[str, dict[str, object]]) -> Path:
    path = tmp_path / "raw.zip"
    with ZipFile(path, "w") as archive:
        for name, payload in entries.items():
            archive.writestr(name, json.dumps(payload, ensure_ascii=False))
    return path


def _documents(key: str, values: list[dict[str, object]]) -> dict[str, object]:
    return {"id": "root", "document": [{"id": "doc", key: values}]}


def test_newspaper_parser_is_bounded_and_deterministic(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path,
        {
            "nested/NIRW2.json": _documents(
                "paragraph",
                [{"id": "n2", "form": "둘"}],
            ),
            "nested/NIRW1.json": _documents(
                "paragraph",
                [{"id": "n1", "form": "하나"}, {"id": "n1b", "form": "추가"}],
            ),
        },
    )

    first = tuple(iter_raw_text_units(archive, source="newspaper", limit=2))
    second = tuple(iter_raw_text_units(archive, source="newspaper", limit=2))

    assert first == second
    assert [(item.text_id, item.form) for item in first] == [("n1", "하나"), ("n1b", "추가")]
    assert all(item.schema_path == "document[].paragraph[]" for item in first)


def test_dialogue_parser_uses_utterances(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path,
        {
            "SARW1.json": _documents(
                "utterance",
                [{"id": "u1", "form": "안녕하세요"}, {"id": "u2", "form": "반갑습니다"}],
            )
        },
    )

    units = tuple(iter_raw_text_units(archive, source="dialogue", limit=1))

    assert len(units) == 1
    assert units[0].text_id == "u1"
    assert units[0].schema_path == "document[].utterance[]"


def test_online_ebrw_and_esrw_sentence_first_with_paragraph_fallback(tmp_path: Path) -> None:
    archive = _write_archive(
        tmp_path,
        {
            "EBRW1.json": _documents(
                "paragraph", [{"id": "e1", "form": "블로그 문단"}]
            ),
            "ESRW1.json": _documents(
                "paragraph",
                [
                    {
                        "id": "p1",
                        "form": "문단 전체",
                        "sentence": [
                            {"id": "s1", "form": "첫 문장"},
                            {"id": "s2", "form": "둘째 문장"},
                        ],
                    },
                    {"id": "p2", "form": "fallback 문단", "sentence": []},
                ],
            ),
        },
    )

    ebrw = tuple(iter_raw_text_units(archive, source="online_ebrw", limit=2))
    esrw = tuple(iter_raw_text_units(archive, source="online_esrw", limit=3))

    assert [(item.text_id, item.form) for item in ebrw] == [("e1", "블로그 문단")]
    assert [(item.text_id, item.form) for item in esrw] == [
        ("s1", "첫 문장"),
        ("s2", "둘째 문장"),
        ("p2", "fallback 문단"),
    ]
    assert esrw[-1].schema_path == "document[].paragraph[]"
