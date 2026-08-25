"""Tests for corpus-frequency joins and answer-candidate analysis."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from contextle_eval.frequency_analysis import (
    FrequencyAnalysisError,
    join_answer_candidate_frequency,
    load_frequency_csv,
)

CANDIDATE_FIELDS = (
    "word",
    "pos",
    "length",
    "is_proper_noun",
    "review_required",
    "review_reason",
)


def _write_csv(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_candidates(path: Path) -> None:
    _write_csv(
        path,
        CANDIDATE_FIELDS,
        [
            {
                "word": "  Ａ ",
                "pos": "명사",
                "length": "1",
                "is_proper_noun": "false",
                "review_required": "false",
                "review_reason": "",
            },
            {
                "word": "나",
                "pos": "대명사",
                "length": "1",
                "is_proper_noun": "false",
                "review_required": "true",
                "review_reason": "technical_term",
            },
            {
                "word": "다",
                "pos": "명사",
                "length": "1",
                "is_proper_noun": "false",
                "review_required": "true",
                "review_reason": "long_word",
            },
            {
                "word": "없음",
                "pos": "명사",
                "length": "2",
                "is_proper_noun": "false",
                "review_required": "false",
                "review_reason": "",
            },
        ],
    )


def test_nfkc_join_aggregation_ranking_unmatched_and_metadata_preservation(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidates.csv"
    frequency_path = tmp_path / "frequency.csv"
    output_path = tmp_path / "joined.csv"
    _write_candidates(candidate_path)
    _write_csv(
        frequency_path,
        ("word", "count", "document_frequency", "frequency_rank", "source"),
        [
            {
                "word": " A ",
                "count": "2",
                "document_frequency": "1",
                "frequency_rank": "8",
                "source": "corpus-a",
            },
            {
                "word": "Ａ",
                "count": "3",
                "document_frequency": "2",
                "frequency_rank": "5",
                "source": "corpus-b",
            },
            {
                "word": "나",
                "count": "5",
                "document_frequency": "4",
                "frequency_rank": "1",
                "source": "corpus-a",
            },
            {
                "word": "다",
                "count": "1",
                "document_frequency": "",
                "frequency_rank": "",
                "source": "",
            },
            {
                "word": "  ",
                "count": "99",
                "document_frequency": "",
                "frequency_rank": "",
                "source": "",
            },
        ],
    )

    analysis = join_answer_candidate_frequency(
        candidate_path, frequency_path, output_path, seed=7, sample_size=2
    )
    with output_path.open(encoding="utf-8", newline="") as csv_file:
        rows = {row["word"]: row for row in csv.DictReader(csv_file)}

    assert rows["A"]["frequency"] == "5"
    assert rows["A"]["frequency_rank"] == "1"
    assert rows["나"]["frequency_rank"] == "1"
    assert rows["다"]["frequency_rank"] == "3"
    assert rows["A"]["document_frequency"] == "3"
    assert rows["A"]["source"] == "corpus-a|corpus-b"
    assert rows["A"]["source_frequency_rank"] == "5"
    assert rows["A"]["pos"] == "명사"
    assert rows["나"]["review_reason"] == "technical_term"
    assert rows["없음"]["frequency_found"] == "false"
    assert rows["없음"]["frequency"] == ""
    assert rows["없음"]["frequency_rank"] == ""
    assert analysis.candidate_count == 4
    assert analysis.matched_count == 3
    assert analysis.unmatched_count == 1
    assert analysis.coverage_ratio == pytest.approx(0.75)
    assert analysis.frequency_min == 1
    assert analysis.frequency_max == 5
    assert analysis.frequency_median == 5
    assert analysis.input_stats.duplicate_rows == 1
    assert analysis.input_stats.skipped_blank_words == 1
    assert [row["word"] for row in analysis.top_sample] == ["A", "나"]


def test_duplicate_policy_error_rejects_normalized_duplicates(tmp_path: Path) -> None:
    frequency_path = tmp_path / "frequency.csv"
    _write_csv(
        frequency_path,
        ("word", "frequency"),
        [{"word": "Ａ", "frequency": "1.5"}, {"word": "A", "frequency": "2.5"}],
    )

    records, stats = load_frequency_csv(frequency_path)
    assert records["A"].frequency == 4
    assert stats.duplicate_rows == 1

    with pytest.raises(FrequencyAnalysisError, match="Duplicate normalized"):
        load_frequency_csv(frequency_path, duplicate_policy="error")


@pytest.mark.parametrize("bad_value", ["NaN", "Infinity", "-1"])
def test_rejects_non_finite_or_negative_frequency(
    tmp_path: Path, bad_value: str
) -> None:
    frequency_path = tmp_path / "frequency.csv"
    _write_csv(
        frequency_path,
        ("word", "frequency"),
        [{"word": "단어", "frequency": bad_value}],
    )

    with pytest.raises(FrequencyAnalysisError, match="finite non-negative"):
        load_frequency_csv(frequency_path)


def test_empty_frequency_input_is_rejected(tmp_path: Path) -> None:
    frequency_path = tmp_path / "frequency.csv"
    _write_csv(frequency_path, ("word", "count"), [{"word": "  ", "count": "1"}])

    with pytest.raises(FrequencyAnalysisError, match="no usable"):
        load_frequency_csv(frequency_path)


def test_both_value_columns_require_explicit_selection(tmp_path: Path) -> None:
    frequency_path = tmp_path / "frequency.csv"
    _write_csv(
        frequency_path,
        ("word", "count", "frequency"),
        [{"word": "단어", "count": "3", "frequency": "0.25"}],
    )

    with pytest.raises(FrequencyAnalysisError, match="both"):
        load_frequency_csv(frequency_path)

    records, stats = load_frequency_csv(frequency_path, value_column="count")
    assert records["단어"].frequency == 3
    assert stats.value_column == "count"
