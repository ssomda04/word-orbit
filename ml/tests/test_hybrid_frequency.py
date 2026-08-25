"""Tests for POS-aware raw/lemma hybrid frequency selection."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from contextle_eval.hybrid_frequency import (
    HybridFrequencyError,
    LemmaBucket,
    build_hybrid_report,
    join_hybrid_frequency,
    load_lemma_frequency,
    load_raw_frequency,
    policy_source,
)

CANDIDATE_FIELDS = ("word", "pos", "review_required", "review_reason")


def _write_csv(
    path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, str]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_policy_gives_predicate_membership_precedence() -> None:
    assert policy_source("동사") == "lemma"
    assert policy_source("명사|형용사") == "lemma"
    assert policy_source("명사|부사") == "raw"


def test_selects_exactly_one_source_without_addition_or_fallback(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    output = tmp_path / "hybrid.csv"
    _write_csv(
        candidates,
        CANDIDATE_FIELDS,
        [
            {"word": "먹다", "pos": "동사", "review_required": "false", "review_reason": ""},
            {"word": "사람", "pos": "명사", "review_required": "false", "review_reason": ""},
            {"word": "웃다", "pos": "동사", "review_required": "false", "review_reason": ""},
            {"word": "희귀명사", "pos": "명사", "review_required": "true", "review_reason": "technical_term"},
        ],
    )
    raw = {"먹다": 5, "사람": 30, "웃다": 9}
    lemma = {
        ("먹다", "동사"): LemmaBucket("먹다", "동사", 100, 4),
        ("사람", "명사"): LemmaBucket("사람", "명사", 40, 1),
        ("희귀명사", "명사"): LemmaBucket("희귀명사", "명사", 7, 1),
    }

    rows = join_hybrid_frequency(candidates, raw, lemma, output)

    assert rows[0]["selected_frequency"] == "100"
    assert rows[0]["selected_frequency_source"] == "lemma"
    assert rows[0]["raw_frequency"] == "5"
    assert rows[0]["lemma_frequency"] == "100"
    assert rows[1]["selected_frequency"] == "30"
    assert rows[1]["selected_frequency_source"] == "raw"
    assert rows[2]["raw_frequency"] == "9"
    assert rows[2]["selected_frequency"] == ""
    assert rows[2]["selected_frequency_source"] == "none"
    assert rows[2]["frequency_found"] == "false"
    assert rows[3]["lemma_frequency"] == "7"
    assert rows[3]["selected_frequency"] == ""
    assert rows[3]["selected_frequency_source"] == "none"
    assert all("rank" not in field for field in rows[0])


def test_mixed_and_cross_pos_risks_are_explicit(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    output = tmp_path / "hybrid.csv"
    _write_csv(
        candidates,
        CANDIDATE_FIELDS,
        [
            {"word": "달다", "pos": "동사|명사|형용사", "review_required": "false", "review_reason": ""}
        ],
    )
    lemma = {
        ("달다", "동사"): LemmaBucket("달다", "동사", 40, 3),
        ("달다", "형용사"): LemmaBucket("달다", "형용사", 60, 4),
    }

    row = join_hybrid_frequency(candidates, {"달다": 5}, lemma, output)[0]

    assert row["selected_frequency"] == "100"
    assert row["selected_frequency_source"] == "lemma"
    assert row["lemma_frequency_pos"] == "동사|형용사"
    assert row["lemma_source_surface_count"] == "7"
    assert row["hybrid_risk_flags"] == (
        "mixed_candidate_pos_predicate_precedence|cross_pos_lemma_buckets|"
        "special_lemma_review"
    )


def test_report_compares_raw_lemma_and_hybrid_coverage(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    output = tmp_path / "hybrid.csv"
    _write_csv(
        candidates,
        CANDIDATE_FIELDS,
        [
            {"word": "먹다", "pos": "동사", "review_required": "false", "review_reason": ""},
            {"word": "사람", "pos": "명사", "review_required": "false", "review_reason": ""},
            {"word": "없다", "pos": "형용사", "review_required": "true", "review_reason": "archaic"},
        ],
    )
    raw = {"먹다": 5, "사람": 30}
    lemma = {
        ("먹다", "동사"): LemmaBucket("먹다", "동사", 100, 4),
        ("없다", "형용사"): LemmaBucket("없다", "형용사", 80, 3),
    }
    rows = join_hybrid_frequency(candidates, raw, lemma, output)

    report = build_hybrid_report(rows, raw, lemma, {}, seed=7)

    overall = report["overall_coverage"]
    assert overall["raw_matched"] == 2
    assert overall["lemma_matched"] == 2
    assert overall["hybrid_matched"] == 3
    assert overall["hybrid_coverage_ratio"] == 1.0
    assert report["review_required_coverage"]["true"]["hybrid_matched"] == 1
    assert len(report["top_100"]) == 3
    assert len(report["unmatched_100"]) == 0


def test_loaders_validate_schema_status_and_values(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.csv"
    lemma_path = tmp_path / "lemma.csv"
    _write_csv(raw_path, ("word", "count"), [{"word": "사람", "count": "3"}])
    _write_csv(
        lemma_path,
        ("lemma", "pos", "count", "source_surface_count", "analysis_status"),
        [
            {
                "lemma": "먹다",
                "pos": "동사",
                "count": "7",
                "source_surface_count": "2",
                "analysis_status": "exact",
            }
        ],
    )
    assert load_raw_frequency(raw_path) == {"사람": 3}
    assert load_lemma_frequency(lemma_path)["먹다", "동사"].count == 7

    _write_csv(raw_path, ("word", "count"), [{"word": "사람", "count": "-1"}])
    with pytest.raises(HybridFrequencyError, match="non-negative"):
        load_raw_frequency(raw_path)

    _write_csv(
        lemma_path,
        ("lemma", "pos", "count", "source_surface_count", "analysis_status"),
        [
            {
                "lemma": "먹다",
                "pos": "동사",
                "count": "7",
                "source_surface_count": "2",
                "analysis_status": "ambiguous",
            }
        ],
    )
    with pytest.raises(HybridFrequencyError, match="not an exact aggregate"):
        load_lemma_frequency(lemma_path)
