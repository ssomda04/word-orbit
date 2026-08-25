"""Tests for raw Modu MP morpheme-form/POS frequency aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from contextle_eval.modu_corpus import (
    MorphemeRecord,
    ParsedSentence,
    SourceSubtype,
    ValidationIssue,
)
from contextle_eval.modu_frequency import (
    CSV_FIELDS,
    ModuFrequencyError,
    aggregate_morpheme_frequencies,
    build_report,
    write_outputs,
)


def _record(
    index: int,
    *,
    subtype: SourceSubtype = "NXMP",
    morpheme: str = "단어",
    pos: str = "NNG",
) -> MorphemeRecord:
    return MorphemeRecord(
        corpus_id=f"{subtype}.test",
        document_id=f"{subtype}.test.1",
        sentence_id=f"{subtype}.test.1.1",
        mp_id=index + 1,
        word_id=index + 1,
        word_form=morpheme,
        morpheme=morpheme,
        pos=pos,
        position=1,
        source_subtype=subtype,
    )


def _sentence(
    subtype: SourceSubtype,
    records: list[MorphemeRecord],
    issues: list[ValidationIssue] | None = None,
) -> ParsedSentence:
    return ParsedSentence(
        corpus_id=f"{subtype}.test",
        document_id=f"{subtype}.test.1",
        sentence_id=f"{subtype}.test.1.1",
        source_subtype=subtype,
        sentence_form="문장",
        original_form="문장",
        word_count=len(records),
        records=tuple(records),
        issues=tuple(issues or ()),
    )


def _issue(code: str, record_index: int, mp_id: int | None = None) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        sentence_id="NXMP.test.1.1",
        mp_id=mp_id if mp_id is not None else record_index + 1,
        record_index=record_index,
        detail=code,
    )


def test_aggregates_raw_forms_with_nfkc_and_keeps_subtypes_separate() -> None:
    sentences = [
        _sentence("NXMP", [_record(0, morpheme=" Ａ "), _record(1, morpheme="A")]),
        _sentence("SXMP", [_record(0, subtype="SXMP", morpheme="Ａ")]),
    ]

    result = aggregate_morpheme_frequencies(sentences)

    assert [(row.source_subtype, row.morpheme, row.pos, row.count) for row in result.rows] == [
        ("NXMP", "A", "NNG", 2),
        ("SXMP", "A", "NNG", 1),
    ]
    assert result.subtypes["NXMP"].total_sentences == 1
    assert result.subtypes["NXMP"].counted_mp == 2
    assert result.subtypes["SXMP"].counted_mp == 1


def test_applies_explicit_issue_policy_without_duplicate_double_count() -> None:
    records = [
        _record(0, morpheme="유지"),
        _record(1, morpheme=""),
        _record(2, morpheme="품사없음", pos=""),
        _record(3, morpheme="연결없음"),
        _record(4, morpheme="고아"),
        _record(5, morpheme="중복"),
        _record(6, morpheme="중복"),
    ]
    issues = [
        _issue("position_order", 0),
        _issue("empty_morpheme", 1),
        _issue("empty_pos", 2),
        _issue("missing_word_id", 3),
        _issue("orphan_word_id", 4),
        _issue("duplicate_mp_id", 6, mp_id=6),
    ]

    result = aggregate_morpheme_frequencies([_sentence("NXMP", records, issues)])
    stats = result.subtypes["NXMP"]

    assert [(row.morpheme, row.count) for row in result.rows] == [("유지", 1), ("중복", 1)]
    assert stats.total_mp_records == 7
    assert stats.counted_mp == 2
    assert stats.excluded_mp == 5
    assert stats.issue_counts == {
        "duplicate_mp_id": 1,
        "empty_morpheme": 1,
        "empty_pos": 1,
        "missing_word_id": 1,
        "orphan_word_id": 1,
        "position_order": 1,
    }


def test_rejects_unknown_issue_instead_of_silently_dropping() -> None:
    sentence = _sentence("NXMP", [_record(0)], [_issue("future_issue", 0)])

    with pytest.raises(ModuFrequencyError, match="Unsupported validation issue"):
        aggregate_morpheme_frequencies([sentence])


def test_output_order_and_schema_are_deterministic(tmp_path: Path) -> None:
    sentences = [
        _sentence("SXMP", [_record(0, subtype="SXMP", morpheme="나", pos="VV")]),
        _sentence(
            "NXMP",
            [
                _record(0, morpheme="가", pos="NNG"),
                _record(1, morpheme="나", pos="VV"),
                _record(2, morpheme="가", pos="NNG"),
            ],
        ),
    ]
    first = aggregate_morpheme_frequencies(sentences)
    second = aggregate_morpheme_frequencies(sentences)
    assert first == second

    first_csv = tmp_path / "first.csv"
    first_report = tmp_path / "first.json"
    second_csv = tmp_path / "second.csv"
    second_report = tmp_path / "second.json"
    write_outputs(first, first_csv, first_report)
    write_outputs(second, second_csv, second_report)

    assert first_csv.read_bytes() == second_csv.read_bytes()
    assert first_report.read_bytes() == second_report.read_bytes()
    with first_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0]) == CSV_FIELDS
    assert rows == [
        {"source_subtype": "NXMP", "morpheme": "가", "pos": "NNG", "count": "2"},
        {"source_subtype": "NXMP", "morpheme": "나", "pos": "VV", "count": "1"},
        {"source_subtype": "SXMP", "morpheme": "나", "pos": "VV", "count": "1"},
    ]
    report = json.loads(first_report.read_text(encoding="utf-8"))
    assert report == build_report(first)
    assert report["subtypes"]["NXMP"]["unique_morpheme_pos"] == 2
