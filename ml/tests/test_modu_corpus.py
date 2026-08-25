"""Tests for bounded, streaming NXMP parsing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from contextle_eval.modu_corpus import (
    ModuCorpusError,
    iter_mp_sentences,
    iter_nxmp_sentences,
    iter_sxmp_sentences,
    select_nxmp_entry,
    select_sxmp_entry,
)


def _sentence(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": "NXMP.test.1.1",
        "form": "출장길이다",
        "original_form": "출장길이다",
        "word": [
            {"id": 1, "form": "출장길"},
            {"id": 2, "form": "이다"},
        ],
        "MP": [
            {"id": 1, "form": "출장길", "label": "NNG", "word_id": 1, "position": 1},
            {"id": 2, "form": "이", "label": "VCP", "word_id": 2, "position": 1},
            {"id": 3, "form": "다", "label": "EF", "word_id": 2, "position": 2},
        ],
    }
    value.update(overrides)
    return value


def _write_zip(
    tmp_path: Path,
    sentences: list[dict[str, object]],
    *,
    entry: str = "NXMP.json",
    source_subtype: str = "NXMP",
) -> Path:
    archive_path = tmp_path / "corpus.zip"
    payload = {
        "id": f"{source_subtype}.test",
        "metadata": {"annotation_level": ["형태 분석"]},
        "document": [{"id": f"{source_subtype}.test.1", "sentence": sentences}],
    }
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(entry, json.dumps(payload, ensure_ascii=False))
        other_subtype = "SXMP" if source_subtype == "NXMP" else "NXMP"
        archive.writestr(f"{other_subtype}.json", "{}")
    return archive_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_parses_nxmp_and_links_sentence_word_and_mp(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path, [_sentence()])

    parsed = list(iter_nxmp_sentences(archive))

    assert len(parsed) == 1
    assert parsed[0].word_count == 2
    assert parsed[0].issues == ()
    assert parsed[0].records[0].morpheme == "출장길"
    assert parsed[0].records[0].pos == "NNG"
    assert parsed[0].records[0].word_form == "출장길"
    assert parsed[0].records[2].word_form == "이다"
    assert parsed[0].records[2].source_subtype == "NXMP"


def test_selects_only_nxmp_json_inside_zip(tmp_path: Path) -> None:
    archive_path = _write_zip(tmp_path, [_sentence()], entry="nested/NXMP2500.json")
    with ZipFile(archive_path) as archive:
        assert select_nxmp_entry(archive) == "nested/NXMP2500.json"


def test_sxmp_uses_same_record_schema_and_distinct_subtype(tmp_path: Path) -> None:
    sentence = _sentence(id="SXMP.test.1.1")
    archive = _write_zip(
        tmp_path,
        [sentence],
        entry="nested/SXMP2500.json",
        source_subtype="SXMP",
    )

    with ZipFile(archive) as opened:
        assert select_sxmp_entry(opened) == "nested/SXMP2500.json"
    parsed = list(iter_sxmp_sentences(archive))
    generic = list(iter_mp_sentences(archive, source_subtype="SXMP"))

    assert parsed == generic
    assert parsed[0].corpus_id == "SXMP.test"
    assert parsed[0].records[0].source_subtype == "SXMP"
    assert parsed[0].records[0].morpheme == "출장길"
    assert parsed[0].records[0].pos == "NNG"


def test_reports_missing_and_orphan_word_ids(tmp_path: Path) -> None:
    sentence = _sentence(
        MP=[
            {"id": 1, "form": "가", "label": "NNG", "position": 1},
            {"id": 2, "form": "나", "label": "NNG", "word_id": 99, "position": 1},
        ]
    )
    parsed = next(iter_nxmp_sentences(_write_zip(tmp_path, [sentence])))

    assert [issue.code for issue in parsed.issues] == ["missing_word_id", "orphan_word_id"]
    assert parsed.records[0].word_id is None
    assert parsed.records[1].word_form is None


def test_reports_position_order_and_duplicate_mp_id(tmp_path: Path) -> None:
    sentence = _sentence(
        MP=[
            {"id": 1, "form": "출", "label": "NNG", "word_id": 1, "position": 2},
            {"id": 1, "form": "장", "label": "NNG", "word_id": 1, "position": 1},
        ]
    )
    parsed = next(iter_nxmp_sentences(_write_zip(tmp_path, [sentence])))

    assert [issue.code for issue in parsed.issues] == [
        "position_order",
        "duplicate_mp_id",
        "position_order",
    ]


def test_reports_non_positive_and_missing_positions(tmp_path: Path) -> None:
    sentence = _sentence(
        MP=[
            {"id": 1, "form": "출", "label": "NNG", "word_id": 1, "position": 0},
            {"id": 2, "form": "장", "label": "NNG", "word_id": 1},
        ]
    )
    parsed = next(iter_nxmp_sentences(_write_zip(tmp_path, [sentence])))

    assert [issue.code for issue in parsed.issues] == [
        "position_order",
        "position_order",
    ]


@pytest.mark.parametrize("missing", ["id", "form", "word", "MP"])
def test_rejects_missing_required_sentence_field(tmp_path: Path, missing: str) -> None:
    sentence = _sentence()
    del sentence[missing]

    with pytest.raises(ModuCorpusError, match=missing):
        list(iter_nxmp_sentences(_write_zip(tmp_path, [sentence])))


def test_rejects_malformed_json(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("NXMP.json", '{"id":"broken","document":[')

    with pytest.raises(ModuCorpusError, match="Malformed NXMP JSON"):
        list(iter_nxmp_sentences(archive_path))


def test_rejects_sentence_without_document_id(tmp_path: Path) -> None:
    archive_path = tmp_path / "missing-document-id.zip"
    payload = {"id": "NXMP.test", "document": [{"sentence": [_sentence()]}]}
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("NXMP.json", json.dumps(payload, ensure_ascii=False))

    with pytest.raises(ModuCorpusError, match="document id"):
        list(iter_nxmp_sentences(archive_path))


def test_rejects_empty_sentence_stream_and_mismatched_subtype(tmp_path: Path) -> None:
    empty = _write_zip(tmp_path, [])
    with pytest.raises(ModuCorpusError, match="contains no sentences"):
        list(iter_nxmp_sentences(empty))

    mismatch_path = tmp_path / "mismatch.zip"
    payload = {
        "id": "NXMP.test",
        "document": [{"id": "NXMP.test.1", "sentence": [_sentence()]}],
    }
    with ZipFile(mismatch_path, "w") as archive:
        archive.writestr("SXMP.json", json.dumps(payload, ensure_ascii=False))
    with pytest.raises(ModuCorpusError, match="mismatched root id"):
        list(iter_sxmp_sentences(mismatch_path))


def test_output_is_deterministic_and_archive_is_unchanged(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path, [_sentence(), _sentence(id="NXMP.test.1.2")])
    before_hash = _sha256(archive)
    before_stat = archive.stat()

    first = list(iter_nxmp_sentences(archive, limit_sentences=1))
    second = list(iter_nxmp_sentences(archive, limit_sentences=1))

    assert first == second
    assert _sha256(archive) == before_hash
    assert archive.stat().st_size == before_stat.st_size
    assert archive.stat().st_mtime_ns == before_stat.st_mtime_ns
