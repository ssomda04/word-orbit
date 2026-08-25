"""Tests for reviewable answer-candidate generation."""

from __future__ import annotations

import bz2
import csv
from pathlib import Path

from contextle_eval.answer_candidates import (
    build_answer_candidates,
    extract_quality_metadata,
)

MEDIAWIKI_OPEN = '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">'


def _page(title: str, text: str) -> str:
    escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f"<page><title>{title}</title><ns>0</ns>"
        f"<revision><text>{escaped_text}</text></revision></page>"
    )


def _write_dump(path: Path, pages: list[str]) -> None:
    xml = MEDIAWIKI_OPEN + "".join(pages) + "</mediawiki>"
    path.write_bytes(bz2.compress(xml.encode("utf-8")))


def test_builds_candidates_and_excludes_only_automatic_failures(tmp_path: Path) -> None:
    dump_path = tmp_path / "sample.xml.bz2"
    vocabulary_path = tmp_path / "game_words.txt"
    output_path = tmp_path / "answer_candidates.csv"
    vocabulary_path.write_text(
        "학생\n서울\n의료기술\n아주아주길고긴낱말\n옛낱말\n사투리말\n옛관직\n드문말\n"
        "접사말\n단어!\n없는말\n",
        encoding="utf-8",
    )
    _write_dump(
        dump_path,
        [
            _page("학생", "== 한국어 ==\n=== 명사 ===\n뜻"),
            _page("서울", "== 한국어 ==\n=== 고유 명사 ===\n뜻"),
            _page("의료기술", "== 한국어 ==\n=== 명사 ===\n{{라벨|ko|의학}}"),
            _page("아주아주길고긴낱말", "== 한국어 ==\n=== 명사 ===\n뜻"),
            _page("옛낱말", "== 한국어 ==\n=== 명사 ===\n{{lb|ko|고어}}"),
            _page("사투리말", "== 한국어 ==\n=== 명사 ===\n[[분류:한국어 제주 방언]]"),
            _page("옛관직", "== 한국어 ==\n=== 명사 ===\n{{tlb|ko|역사}}"),
            _page("드문말", "== 한국어 ==\n=== 명사 ===\n{{라벨|ko|드물게}}"),
            _page("접사말", "== 한국어 ==\n=== 접사 ===\n뜻"),
            _page("단어!", "== 한국어 ==\n=== 명사 ===\n뜻"),
        ],
    )

    stats = build_answer_candidates(dump_path, vocabulary_path, output_path)
    with output_path.open(encoding="utf-8", newline="") as csv_file:
        rows = {row["word"]: row for row in csv.DictReader(csv_file)}

    assert set(rows) == {
        "학생",
        "의료기술",
        "아주아주길고긴낱말",
        "옛낱말",
        "사투리말",
        "옛관직",
        "드문말",
    }
    assert rows["학생"]["review_required"] == "false"
    assert rows["의료기술"]["review_reason"] == "technical_term"
    assert rows["의료기술"]["domain_labels"] == "의학"
    assert rows["아주아주길고긴낱말"]["review_reason"] == "long_word"
    assert rows["옛낱말"]["review_reason"] == "archaic"
    assert rows["사투리말"]["review_reason"] == "dialect"
    assert rows["옛관직"]["review_reason"] == "historical_term"
    assert rows["드문말"]["review_reason"] == "explicit_rare_label"
    assert stats.vocabulary_words == 11
    assert stats.candidate_words == 7
    assert stats.excluded == {
        "contains_special_character": 1,
        "missing_wiktionary_metadata": 1,
        "not_general_lexical_pos": 1,
        "proper_noun": 1,
    }


def test_multiple_review_reasons_and_pos_are_deterministic(tmp_path: Path) -> None:
    dump_path = tmp_path / "sample.xml.bz2"
    vocabulary_path = tmp_path / "game_words.txt"
    output_path = tmp_path / "answer_candidates.csv"
    vocabulary_path.write_text("길고긴역사용어입니다\n", encoding="utf-8")
    _write_dump(
        dump_path,
        [
            _page(
                "길고긴역사용어입니다",
                "== 한국어 ==\n=== 명사 ===\n=== 감탄사 ===\n{{라벨|ko|역사|군사}}",
            )
        ],
    )

    stats = build_answer_candidates(dump_path, vocabulary_path, output_path)
    with output_path.open(encoding="utf-8", newline="") as csv_file:
        row = next(csv.DictReader(csv_file))

    assert row["pos"] == "감탄사|명사"
    assert row["review_reason"] == "long_word|technical_term|historical_term"
    assert stats.review_required_words == 1
    assert stats.review_reasons == {
        "historical_term": 1,
        "long_word": 1,
        "technical_term": 1,
    }


def test_explicit_korean_dialect_alternative_form_is_reviewed() -> None:
    quality = extract_quality_metadata(("{{alternative form of|ko|하나|from=방언}}",))

    assert quality.is_dialect is True
