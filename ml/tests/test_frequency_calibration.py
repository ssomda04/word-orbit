"""Tests for POS/source percentile calibration and cutoff simulation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from contextle_eval.frequency_calibration import (
    FrequencyCalibrationError,
    build_calibration_report,
    calibrate_candidates,
    load_hybrid_candidates,
    percentile_by_frequency,
    simulate_cutoffs,
    write_calibrated_csv,
)


def _row(
    word: str,
    pos: str,
    selected: str,
    source: str,
    *,
    raw: str = "",
    lemma: str = "",
    lemma_pos: str = "",
    surface_count: str = "",
    flags: str = "",
    review_required: str = "false",
    review_reason: str = "",
) -> dict[str, str]:
    found = "true" if selected else "false"
    policy = "lemma" if "동사" in pos or "형용사" in pos else "raw"
    return {
        "word": word,
        "pos": pos,
        "selected_frequency": selected,
        "selected_frequency_source": source if selected else "none",
        "frequency_policy_source": policy,
        "frequency_found": found,
        "raw_frequency": raw,
        "lemma_frequency": lemma,
        "lemma_frequency_pos": lemma_pos,
        "lemma_source_surface_count": surface_count,
        "hybrid_risk_flags": flags,
        "review_required": review_required,
        "review_reason": review_reason,
    }


def test_percentile_is_monotonic_and_ties_are_equal() -> None:
    mapping = percentile_by_frequency([10, 20, 20, 100])

    assert mapping[10] == pytest.approx(25)
    assert mapping[20] == pytest.approx(75)
    assert mapping[100] == pytest.approx(100)
    assert mapping[10] <= mapping[20] <= mapping[100]


def test_calibration_is_bucket_local_and_never_adds_raw_and_lemma() -> None:
    input_rows = [
        _row("명사하", "명사", "10", "raw", raw="10", lemma="999"),
        _row("명사상", "명사", "100", "raw", raw="100", lemma="999"),
        _row("동사하", "동사", "100", "lemma", raw="999", lemma="100", lemma_pos="동사"),
        _row("동사상", "동사", "1000", "lemma", raw="999", lemma="1000", lemma_pos="동사"),
    ]

    rows = calibrate_candidates(input_rows)
    by_word = {row["word"]: row for row in rows}

    assert by_word["명사상"]["frequency_bucket"] == "raw:noun"
    assert by_word["동사하"]["frequency_bucket"] == "lemma:verb"
    assert float(by_word["명사상"]["frequency_percentile"]) == pytest.approx(100)
    assert float(by_word["동사하"]["frequency_percentile"]) == pytest.approx(50)
    assert by_word["명사상"]["selected_frequency"] == "100"
    assert by_word["동사하"]["selected_frequency"] == "100"


def test_unmatched_has_null_percentile_and_is_not_treated_as_zero() -> None:
    row = calibrate_candidates([_row("미확인", "명사", "", "none")])[0]

    assert row["frequency_percentile"] == ""
    assert row["frequency_calibration_status"] == "unmatched"
    assert row["selected_frequency"] == ""


def test_mixed_cross_and_special_words_require_manual_review() -> None:
    rows = calibrate_candidates(
        [
            _row(
                "보다",
                "동사|부사|조동사",
                "100",
                "lemma",
                lemma="100",
                lemma_pos="동사",
                surface_count="30",
                flags="mixed_candidate_pos_predicate_precedence|special_lemma_review",
            ),
            _row(
                "있다",
                "동사|형용사",
                "200",
                "lemma",
                lemma="200",
                lemma_pos="동사|형용사",
                surface_count="40",
                flags="cross_pos_lemma_buckets|special_lemma_review",
            ),
        ]
    )
    by_word = {row["word"]: row for row in rows}

    assert by_word["보다"]["frequency_calibration_status"] == "mixed_pos"
    assert by_word["보다"]["manual_review_required"] == "true"
    assert by_word["있다"]["frequency_calibration_status"] == "cross_pos"
    assert by_word["있다"]["frequency_bucket"] == "lemma:cross"
    assert by_word["있다"]["manual_review_required"] == "true"


def test_simulation_has_include_and_exclude_risk_scenarios() -> None:
    rows = calibrate_candidates(
        [
            _row("사람", "명사", "100", "raw", raw="100"),
            _row("먹다", "동사", "100", "lemma", lemma="100", lemma_pos="동사"),
            _row(
                "보다",
                "동사|부사",
                "100",
                "lemma",
                lemma="100",
                lemma_pos="동사",
                flags="mixed_candidate_pos_predicate_precedence",
            ),
            _row("아", "감탄사", "100", "raw", raw="100"),
        ]
    )

    simulations = simulate_cutoffs(rows)

    assert simulations["all_pos_include_risk"]["P90"]["retained_matched_count"] == 4
    assert simulations["all_pos_exclude_mixed_cross"]["P90"]["retained_matched_count"] == 3
    assert simulations["content_words_include_risk"]["P90"]["retained_matched_count"] == 3
    assert simulations["content_words_exclude_mixed_cross"]["P90"]["retained_matched_count"] == 2


def test_report_and_samples_are_deterministic() -> None:
    rows = calibrate_candidates(
        [
            _row("사람", "명사", "10", "raw", raw="10"),
            _row("먹다", "동사", "20", "lemma", lemma="20", lemma_pos="동사"),
            _row("좋다", "형용사", "30", "lemma", lemma="30", lemma_pos="형용사"),
            _row("빨리", "부사", "40", "raw", raw="40", review_required="true"),
        ]
    )

    first = build_calibration_report(rows, seed=20260823)
    second = build_calibration_report(rows, seed=20260823)

    assert first == second
    assert first["candidate_count"] == 4
    assert first["matched_count"] == 4
    assert first["unmatched_policy"]["treated_as_zero"] is False


def test_io_preserves_input_rows_and_manual_flag(tmp_path: Path) -> None:
    input_path = tmp_path / "hybrid.csv"
    output_path = tmp_path / "scored.csv"
    rows = [
        _row(
            "하다",
            "동사",
            "100",
            "lemma",
            lemma="100",
            lemma_pos="동사",
            surface_count="30",
            flags="special_lemma_review",
        )
    ]
    fields = tuple(rows[0])
    with input_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    loaded_fields, loaded = load_hybrid_candidates(input_path)
    calibrated = calibrate_candidates(loaded)
    write_calibrated_csv(output_path, loaded_fields, calibrated)

    with output_path.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == len(rows) == 1
    assert written[0]["manual_review_required"] == "true"
    assert written[0]["selected_frequency"] == "100"


def test_loader_rejects_inconsistent_found_flag(tmp_path: Path) -> None:
    row = _row("사람", "명사", "10", "raw", raw="10")
    row["frequency_found"] = "false"
    with pytest.raises(FrequencyCalibrationError, match="inconsistent"):
        calibrate_candidates([row])
