"""Tests for provenance-preserving Modu frequency normalization."""

from __future__ import annotations

import json
from pathlib import Path

from contextle_eval.modu_normalization import (
    NormalizedFrequencyRow,
    RawFrequencyRow,
    build_normalization_report,
    detect_collisions,
    load_raw_frequency,
    normalize_frequencies,
    normalize_row,
    write_collision_audit,
    write_normalized_frequency,
    write_report,
)

GAME_WORDS = frozenset(
    {"사과", "빨리", "있다", "보다", "맞다", "크다", "쓰다", "사다", "하다", "되다"}
)
ANSWER_WORDS = frozenset({"사과", "있다", "보다", "하다"})


def _raw(subtype: str, form: str, pos: str, count: int = 1) -> RawFrequencyRow:
    assert subtype in {"NXMP", "SXMP"}
    return RawFrequencyRow(subtype, form, pos, count)  # type: ignore[arg-type]


def test_contract_statuses_and_membership_evidence() -> None:
    cases = [
        (_raw("NXMP", "사과", "NNG"), "사과", "matched"),
        (_raw("NXMP", "미등록", "NNG"), "미등록", "unmatched"),
        (_raw("NXMP", "빨리", "MAG"), "빨리", "matched"),
        (_raw("NXMP", "있", "VA"), "있다", "matched"),
        (_raw("NXMP", "보", "VV"), "보다", "matched"),
        (_raw("NXMP", "읽", "VV"), "읽다", "review"),
        (_raw("NXMP", "서울", "NNP"), "서울", "review"),
        (_raw("SXMP", "어머", "IC"), "어머", "review"),
    ]
    for raw, canonical, status in cases:
        normalized = normalize_row(raw, GAME_WORDS, ANSWER_WORDS)
        assert normalized.canonical_form == canonical
        assert normalized.status == status
    assert normalize_row(_raw("NXMP", "사과", "NNG"), GAME_WORDS, frozenset()).status == (
        "matched"
    )


def test_auxiliary_suffix_function_and_unknown_pos_are_unmatched() -> None:
    for pos in ("VX", "XSV", "XSA", "VCP", "VCN", "NNB", "NP", "SL"):
        normalized = normalize_row(_raw("NXMP", "하", pos), GAME_WORDS, ANSWER_WORDS)
        assert normalized.status == "unmatched"
        assert normalized.canonical_form == ""


def test_frequency_loader_applies_nfkc_strip(tmp_path: Path) -> None:
    frequency = tmp_path / "frequency.csv"
    frequency.write_text(
        "source_subtype,morpheme,pos,count\nNXMP, Ａ , NNG ,2\n",
        encoding="utf-8",
    )

    assert load_raw_frequency(frequency) == (_raw("NXMP", "A", "NNG", 2),)


def test_subtype_provenance_collisions_and_count_conservation() -> None:
    raw_rows = (
        _raw("NXMP", "있", "VA", 10),
        _raw("SXMP", "있", "VV", 5),
        _raw("NXMP", "보", "VV", 7),
        _raw("SXMP", "보다", "MAG", 3),
        _raw("NXMP", "하", "VX", 4),
    )
    normalized = normalize_frequencies(raw_rows, GAME_WORDS, ANSWER_WORDS)
    collisions = detect_collisions(normalized)
    report = build_normalization_report(normalized, collisions)

    assert [row.source_subtype for row in normalized] == [
        "NXMP",
        "SXMP",
        "NXMP",
        "SXMP",
        "NXMP",
    ]
    assert any(item.collision_type == "cross_pos" and item.key == "있" for item in collisions)
    assert any(item.collision_type == "canonical" and item.key == "있다" for item in collisions)
    assert any(
        item.collision_type == "source_subtype" and item.key == "있다"
        for item in collisions
    )
    assert any(
        item.collision_type == "required_case" and item.key == "있다"
        for item in collisions
    )
    assert report["overall"]["total_token_count"] == 29
    assert report["overall"]["matched"]["token_count"] == 25
    assert report["overall"]["unmatched"]["token_count"] == 4
    assert report["overall"]["count_conservation"] is True
    conserved = sum(
        report["overall"][status]["token_count"]
        for status in ("matched", "review", "unmatched")
    )
    assert conserved == 29


def test_deterministic_outputs_keep_collision_sources_separate(tmp_path: Path) -> None:
    rows = (
        NormalizedFrequencyRow("NXMP", "있", "VA", "있다", "matched", 10, True, True),
        NormalizedFrequencyRow("SXMP", "있", "VV", "있다", "matched", 5, True, True),
    )
    collisions = detect_collisions(rows)
    report = build_normalization_report(rows, collisions)
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    for directory in (first_directory, second_directory):
        write_normalized_frequency(directory / "normalized.csv", rows)
        write_collision_audit(directory / "collisions.csv", collisions)
        write_report(directory / "report.json", report)

    assert (first_directory / "normalized.csv").read_bytes() == (
        second_directory / "normalized.csv"
    ).read_bytes()
    assert (first_directory / "collisions.csv").read_bytes() == (
        second_directory / "collisions.csv"
    ).read_bytes()
    assert (first_directory / "report.json").read_bytes() == (
        second_directory / "report.json"
    ).read_bytes()
    assert json.loads((first_directory / "report.json").read_text(encoding="utf-8"))[
        "contract"
    ]["canonical_collisions_merged"] is False
