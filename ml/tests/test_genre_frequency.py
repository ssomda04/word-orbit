"""Tests for schema-independent genre frequency normalization."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
from contextle_eval.genre_frequency import (
    PRODUCTION_AGGREGATED_POS,
    CsvRecordAdapter,
    GenreFrequencyError,
    GenreFrequencyRecord,
    compare_genre_frequencies,
    load_genre_frequency_csv,
    load_production_genre_frequency,
)


def _fixture() -> tuple[GenreFrequencyRecord, ...]:
    totals = {
        "newspaper": 100_000_000,
        "dialogue": 1_000_000,
        "online": 10_000_000,
    }
    counts = {
        "newspaper": {
            "신문특화": 100_000,
            "대화특화": 10,
            "온라인특화": 10,
            "균형": 1_000,
            "동률A": 50,
            "동률B": 50,
            "한장르누락": 25,
            "관측제로": 0,
        },
        "dialogue": {
            "신문특화": 1,
            "대화특화": 10_000,
            "온라인특화": 1,
            "균형": 10,
            "동률A": 5,
            "동률B": 5,
            "한장르누락": 2,
            "관측제로": 0,
        },
        "online": {
            "신문특화": 10,
            "대화특화": 10,
            "온라인특화": 100_000,
            "균형": 100,
            "동률A": 5,
            "동률B": 5,
            "온라인누락": 20,
            "관측제로": 0,
        },
    }
    return tuple(
        GenreFrequencyRecord(genre, word, "NNG", count, totals[genre])  # type: ignore[arg-type]
        for genre, word_counts in counts.items()
        for word, count in word_counts.items()
    )


def _by_word(records: tuple[GenreFrequencyRecord, ...] | list[GenreFrequencyRecord]):
    return {row.canonical_word: row for row in compare_genre_frequencies(records)}


def test_synthetic_fixture_exposes_raw_bias_and_relative_size_correction() -> None:
    rows = _by_word(_fixture())
    balanced = rows["균형"]
    assert balanced.newspaper is not None and balanced.dialogue is not None
    assert balanced.online is not None
    assert balanced.newspaper.raw_count == 100 * balanced.dialogue.raw_count
    assert balanced.newspaper.relative_frequency_per_million == 10
    assert balanced.dialogue.relative_frequency_per_million == 10
    assert balanced.online.relative_frequency_per_million == 10
    assert balanced.newspaper.log1p_relative_frequency == math.log1p(10)


def test_percentile_ties_missing_zero_and_coverage_are_explicit() -> None:
    rows = _by_word(_fixture())
    assert rows["동률A"].newspaper is not None
    assert rows["동률B"].newspaper is not None
    assert rows["동률A"].newspaper.empirical_percentile == rows["동률B"].newspaper.empirical_percentile
    assert rows["온라인누락"].newspaper is None
    assert rows["온라인누락"].dialogue is None
    assert rows["온라인누락"].genre_coverage == 1
    assert rows["한장르누락"].newspaper is not None
    assert rows["한장르누락"].dialogue is not None
    assert rows["한장르누락"].online is None
    assert rows["한장르누락"].genre_coverage == 2
    assert rows["관측제로"].newspaper is not None
    assert rows["관측제로"].newspaper.raw_count == 0
    assert rows["관측제로"].genre_coverage == 3
    absent = next(
        row
        for row in compare_genre_frequencies(_fixture(), [("완전미관측", "NNG")])
        if row.canonical_word == "완전미관측"
    )
    assert absent.genre_coverage == 0
    assert absent.mean_percentile is None


def test_genre_specialists_and_percentile_summaries_are_available() -> None:
    rows = _by_word(_fixture())
    assert rows["신문특화"].newspaper.raw_count == 100_000  # type: ignore[union-attr]
    assert rows["대화특화"].dialogue.raw_count == 10_000  # type: ignore[union-attr]
    assert rows["온라인특화"].online.raw_count == 100_000  # type: ignore[union-attr]
    balanced = rows["균형"]
    assert balanced.genre_coverage == 3
    assert balanced.mean_percentile is not None
    assert balanced.median_percentile is not None
    assert balanced.max_percentile is not None


def test_output_is_deterministic_and_input_order_independent() -> None:
    records = _fixture()
    expected = compare_genre_frequencies(records)
    assert compare_genre_frequencies(reversed(records)) == expected
    assert compare_genre_frequencies(records) == expected


def test_csv_adapter_decouples_source_column_names_and_allows_extras(tmp_path: Path) -> None:
    path = tmp_path / "future.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("lemma", "tag", "tokens", "corpus_size", "provenance"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "lemma": " 단어 ",
                "tag": "NNG",
                "tokens": 0,
                "corpus_size": 100,
                "provenance": "synthetic",
            }
        )
    records = load_genre_frequency_csv(
        path,
        CsvRecordAdapter(
            canonical_word_field="lemma",
            pos_field="tag",
            raw_count_field="tokens",
            genre_total_lexical_count_field="corpus_size",
            fixed_genre="online",
        ),
    )
    assert records == (GenreFrequencyRecord("online", "단어", "NNG", 0, 100),)


def test_production_adapter_reads_total_from_report_and_marks_pos_aggregated(
    tmp_path: Path,
) -> None:
    frequency_path = tmp_path / "newspaper_frequency.csv"
    frequency_path.write_text(
        "canonical_form,count\n균형,10\n신문특화,90\n", encoding="utf-8"
    )
    report_path = tmp_path / "newspaper_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "newspaper",
                "frequency": {
                    "assignments": 100,
                    "unique_canonical_forms": 2,
                    "derivational_base_assignments": 0,
                    "derivational_candidate_assignments": 0,
                },
            }
        ),
        encoding="utf-8",
    )

    records = load_production_genre_frequency(
        frequency_path, report_path, "newspaper"
    )

    assert records == (
        GenreFrequencyRecord("newspaper", "균형", PRODUCTION_AGGREGATED_POS, 10, 100),
        GenreFrequencyRecord(
            "newspaper", "신문특화", PRODUCTION_AGGREGATED_POS, 90, 100
        ),
    )


def test_production_adapter_rejects_csv_report_count_mismatch(tmp_path: Path) -> None:
    frequency_path = tmp_path / "dialogue_frequency.csv"
    frequency_path.write_text("canonical_form,count\n단어,9\n", encoding="utf-8")
    report_path = tmp_path / "dialogue_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "source": "dialogue",
                "frequency": {"assignments": 10},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(GenreFrequencyError, match="do not equal"):
        load_production_genre_frequency(frequency_path, report_path, "dialogue")
