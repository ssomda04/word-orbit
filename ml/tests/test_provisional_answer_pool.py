"""Tests for development-only provisional answer-pool selection."""

from __future__ import annotations

from pathlib import Path

import pytest
from contextle_eval.provisional_answer_pool import (
    ProvisionalAnswerPoolError,
    select_provisional_entries,
    validate_reference_membership,
    write_provisional_pool,
)


def _row(
    word: str,
    *,
    pos: str = "명사",
    percentile: str = "70",
    found: str = "true",
    status: str = "normal",
    manual: str = "false",
    high_count: str = "false",
) -> dict[str, str]:
    return {
        "word": word,
        "pos": pos,
        "frequency_found": found,
        "frequency_percentile": percentile,
        "frequency_calibration_status": status,
        "manual_review_required": manual,
        "high_count_risk": high_count,
    }


def test_selects_p70_content_words_and_sorts_deterministically() -> None:
    rows = [
        _row("먹다", pos="동사", percentile="71"),
        _row("사람", percentile="70"),
        _row("낮다", pos="형용사", percentile="69.999"),
        _row("아", pos="감탄사", percentile="100"),
        _row("그", pos="대명사|명사", percentile="100"),
        _row("빨리", pos="부사", found="false", percentile=""),
    ]

    entries = select_provisional_entries(rows)

    assert [entry.word for entry in entries] == ["먹다", "사람"]


@pytest.mark.parametrize(
    ("word", "overrides"),
    [
        ("혼합어", {"status": "mixed_pos"}),
        ("교차어", {"status": "cross_pos"}),
        ("검토어", {"manual": "true"}),
        ("고빈도어", {"high_count": "true"}),
        ("하다", {}),
        ("되다", {}),
        ("있다", {}),
        ("보다", {}),
        ("않다", {}),
        ("감사하다", {}),
        ("진정하다", {}),
        ("달다", {}),
    ],
)
def test_excludes_every_risk_class(word: str, overrides: dict[str, str]) -> None:
    assert select_provisional_entries([_row(word, **overrides)]) == []


def test_rejects_duplicates_and_missing_reference_words(tmp_path: Path) -> None:
    with pytest.raises(ProvisionalAnswerPoolError, match="duplicate"):
        select_provisional_entries([_row("사람"), _row("사람", percentile="80")])

    entries = select_provisional_entries([_row("사람")])
    with pytest.raises(ProvisionalAnswerPoolError, match="missing from game_words"):
        validate_reference_membership(
            entries, game_words=["학교"], answer_candidate_words=["사람"]
        )

    output = tmp_path / "pool.txt"
    validate_reference_membership(entries, game_words=["사람"], answer_candidate_words=["사람"])
    write_provisional_pool(output, entries)
    assert output.read_bytes() == "사람\n".encode()
