"""Sample-dump tests for Korean Wiktionary headword extraction."""

from __future__ import annotations

import bz2
import hashlib
import importlib.util
import unicodedata
from pathlib import Path

import pytest
from contextle_eval.wiktionary_words import (
    WiktionaryExtractionError,
    extract_dump,
    format_statistics,
)

MEDIAWIKI_OPEN = '<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">'


def _page(
    title: str,
    text: str,
    *,
    namespace: int = 0,
    redirect: bool = False,
) -> str:
    redirect_xml = '<redirect title="대상" />' if redirect else ""
    escaped_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        f"<page><title>{title}</title><ns>{namespace}</ns>{redirect_xml}"
        f"<revision><text>{escaped_text}</text></revision></page>"
    )


def _write_dump(path: Path, pages: list[str]) -> None:
    xml = MEDIAWIKI_OPEN + "".join(pages) + "</mediawiki>"
    path.write_bytes(bz2.compress(xml.encode("utf-8")))


def _sample_pages() -> list[str]:
    decomposed_word = unicodedata.normalize("NFD", "가다")
    return [
        _page("학생", "== 한국어 ==\n=== 명사 ===\n뜻"),
        _page("예쁘다", "==한국어==\n=== 형용사 ===\n뜻"),
        _page("빨리", "==  한국어  ==\n=== 부사 1 ===\n뜻"),
        _page(decomposed_word, "== 한국어 ==\n=== 동사 ===\n뜻"),
        _page("가다", "== 한국어 ==\n=== 동사 ===\n중복"),
        _page("먹었다", "== 한국어 ==\n=== 동사 활용형 ===\n활용형"),
        _page("예쁜", "== 한국어 ==\n=== 형용사 ===\n활용형"),
        _page("긴 표현", "== 한국어 ==\n=== 명사 ===\n구"),
        _page("단어2", "== 한국어 ==\n=== 명사 ===\n숫자"),
        _page("단어!", "== 한국어 ==\n=== 명사 ===\n기호"),
        _page("word", "== 한국어 ==\n=== 명사 ===\n비한글"),
        _page("외국어", "== 영어 ==\n=== 명사 ===\n== 한국어 ==\n=== 접사 ==="),
        _page("프랑스", "== 프랑스어 ==\n=== 명사 ==="),
        _page("토론", "== 한국어 ==\n=== 명사 ===", namespace=1),
        _page("넘김", "#넘겨주기 [[대상]]", redirect=True),
    ]


def test_extracts_normalized_unique_korean_lemmas(tmp_path: Path) -> None:
    dump_path = tmp_path / "sample.xml.bz2"
    output_path = tmp_path / "game_words.txt"
    _write_dump(dump_path, _sample_pages())

    stats = extract_dump(dump_path, output_path)

    payload = output_path.read_bytes()
    decoded = payload.decode("utf-8", errors="strict")
    assert decoded.splitlines() == [
        "가다",
        "빨리",
        "예쁘다",
        "학생",
    ]
    assert b"\r\n" not in payload
    assert not payload.startswith(b"\xef\xbb\xbf")
    assert payload.endswith(b"\n")
    assert all(
        word == unicodedata.normalize("NFKC", word).strip()
        for word in decoded.splitlines()
    )
    assert hashlib.sha256(payload).hexdigest() == (
        "87c88e7d74c081597edf8456d3819353f791235ff23d11fa5ed0ad2e098b031d"
    )
    assert stats.pages_seen == 15
    assert stats.unique_words == 4
    assert stats.normalized_titles == 1
    assert stats.excluded["duplicate"] == 1
    assert stats.excluded["non_lemma_predicate"] == 2
    assert stats.excluded["contains_internal_whitespace"] == 1
    assert stats.excluded["contains_number"] == 1
    assert stats.excluded["contains_special_character"] == 1
    assert stats.excluded["contains_non_hangul_character"] == 1
    assert stats.excluded["no_allowed_pos"] == 1
    assert stats.excluded["no_korean_section"] == 1
    assert stats.excluded["non_main_namespace"] == 1
    assert stats.excluded["redirect"] == 1


@pytest.mark.parametrize(
    ("part_of_speech", "word"),
    [
        ("명사", "사람"),
        ("대명사", "우리"),
        ("수사", "하나"),
        ("동사", "먹다"),
        ("형용사", "맑다"),
        ("관형사", "새"),
        ("부사", "매우"),
        ("감탄사", "아하"),
    ],
)
def test_all_major_parts_of_speech_are_allowed(
    tmp_path: Path, part_of_speech: str, word: str
) -> None:
    dump_path = tmp_path / "sample.xml.bz2"
    output_path = tmp_path / "words.txt"
    _write_dump(dump_path, [_page(word, f"== 한국어 ==\n=== {part_of_speech} ===")])

    extract_dump(dump_path, output_path)

    assert output_path.read_text(encoding="utf-8") == f"{word}\n"


def test_other_language_pos_does_not_qualify_page(tmp_path: Path) -> None:
    dump_path = tmp_path / "sample.xml.bz2"
    output_path = tmp_path / "words.txt"
    _write_dump(
        dump_path,
        [_page("테스트", "== 영어 ==\n=== 명사 ===\n== 한국어 ==\n=== 접사 ===")],
    )

    stats = extract_dump(dump_path, output_path)

    assert output_path.read_text(encoding="utf-8") == ""
    assert stats.excluded["no_allowed_pos"] == 1


def test_nested_pos_heading_is_detected(tmp_path: Path) -> None:
    dump_path = tmp_path / "sample.xml.bz2"
    output_path = tmp_path / "words.txt"
    _write_dump(
        dump_path,
        [_page("비", "== 한국어 ==\n=== 어원 1 ===\n==== 명사 2 ====\n뜻")],
    )

    extract_dump(dump_path, output_path)

    assert output_path.read_text(encoding="utf-8") == "비\n"


def test_particle_heading_does_not_qualify_game_word(tmp_path: Path) -> None:
    dump_path = tmp_path / "sample.xml.bz2"
    output_path = tmp_path / "words.txt"
    _write_dump(dump_path, [_page("부터", "== 한국어 ==\n=== 조사 ===")])

    stats = extract_dump(dump_path, output_path)

    assert output_path.read_text(encoding="utf-8") == ""
    assert stats.excluded["no_allowed_pos"] == 1


def test_statistics_include_every_exclusion_reason(tmp_path: Path) -> None:
    dump_path = tmp_path / "sample.xml.bz2"
    output_path = tmp_path / "words.txt"
    _write_dump(dump_path, [_page("학생", "== 한국어 ==\n=== 명사 ===")])

    rendered = format_statistics(extract_dump(dump_path, output_path))

    assert "Unique headwords: 1" in rendered
    assert "contains_number: 0" in rendered
    assert "non_lemma_predicate: 0" in rendered


def test_missing_dump_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(WiktionaryExtractionError, match="dump file not found"):
        extract_dump(tmp_path / "missing.xml.bz2", tmp_path / "words.txt")


def test_cli_writes_output_and_prints_statistics(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dump_path = tmp_path / "sample.xml.bz2"
    output_path = tmp_path / "words.txt"
    _write_dump(dump_path, [_page("학생", "== 한국어 ==\n=== 명사 ===")])
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "extract_wiktionary_words.py"
    spec = importlib.util.spec_from_file_location("extract_wiktionary_words", script_path)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    exit_code = script.main(["--dump-path", str(dump_path), "--output", str(output_path)])

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "학생\n"
    assert "Excluded by reason:" in capsys.readouterr().out
