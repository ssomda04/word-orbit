"""Tests for policy-neutral final answer-pool candidate audit preparation."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from contextle_eval.final_pool_evaluation import (
    AnswerCandidateMetadata,
    FinalPoolCandidate,
    FinalPoolEvaluationError,
    FrequencyEvidence,
    GenreComparisonEvidence,
    evaluate_candidates,
    load_final_pool_candidates,
    load_genre_comparison_evidence,
    write_candidate_audit,
)


def _metadata(*, proper: bool = False, review: bool = False) -> AnswerCandidateMetadata:
    return AnswerCandidateMetadata(
        is_proper_noun=proper,
        is_general_lexical_pos=True,
        is_archaic=False,
        is_dialect=False,
        is_historical=False,
        is_technical=False,
        wiktionary_labels=(),
        domain_labels=(),
        review_required=review,
        review_reasons=("technical_term",) if review else (),
    )


def _frequency(
    *, found: bool = True, status: str = "normal", manual: bool = False
) -> FrequencyEvidence:
    return FrequencyEvidence(
        found=found,
        selected_frequency=100 if found else None,
        source="raw" if found else "none",
        percentile=80.0 if found else None,
        calibration_status=status,
        high_count_risk=status == "high_count_risk",
        manual_review_required=manual,
        risk_flags=(),
    )


def _genre(coverage: int) -> GenreComparisonEvidence:
    genres = ("newspaper", "dialogue", "online")[:coverage]
    return GenreComparisonEvidence(
        pos="AGGREGATED",
        genre_coverage=coverage,
        observed_genres=genres,
        mean_percentile=0.8 if coverage else None,
        median_percentile=0.8 if coverage else None,
        max_percentile=0.9 if coverage else None,
    )


def test_synthetic_fixture_preserves_evidence_and_routes_established_risks() -> None:
    candidates = (
        FinalPoolCandidate("균형명사", "명사", _metadata(), _frequency(), _genre(3)),
        FinalPoolCandidate("단일장르", "명사", _metadata(), _frequency(), _genre(1)),
        FinalPoolCandidate("장르미관측", "명사", _metadata(), _frequency(), _genre(0)),
        FinalPoolCandidate("서울", "고유 명사", _metadata(proper=True), _frequency()),
        FinalPoolCandidate("혼합어", "명사|동사", _metadata(), _frequency(status="mixed_pos")),
        FinalPoolCandidate("교차어", "동사|형용사", _metadata(), _frequency(status="cross_pos")),
        FinalPoolCandidate("검토어", "명사", _metadata(review=True), _frequency()),
        FinalPoolCandidate(
            "기존약한어",
            "명사",
            _metadata(),
            _frequency(),
            _genre(0),
            in_provisional_pool_baseline=True,
        ),
        FinalPoolCandidate("충분어", "명사", _metadata(), _frequency(), _genre(2)),
    )

    rows = {row.candidate.canonical_word: row for row in evaluate_candidates(candidates)}

    assert rows["균형명사"].status == "eligible"
    assert rows["균형명사"].candidate.genre.genre_coverage == 3  # type: ignore[union-attr]
    assert rows["단일장르"].status == "eligible"
    assert rows["단일장르"].candidate.genre.genre_coverage == 1  # type: ignore[union-attr]
    assert rows["장르미관측"].candidate.genre.genre_coverage == 0  # type: ignore[union-attr]
    assert rows["장르미관측"].status == "review_required"
    assert rows["서울"].status == "excluded"
    assert rows["서울"].reasons == ("proper_noun",)
    assert rows["혼합어"].reasons == ("mixed_pos",)
    assert rows["교차어"].reasons == ("cross_pos",)
    assert rows["검토어"].reasons == ("manual_review",)
    assert rows["기존약한어"].status == "review_required"
    assert rows["기존약한어"].reasons == ("insufficient_frequency_evidence",)
    assert "provisional_pool_baseline" in rows["기존약한어"].available_evidence
    assert rows["충분어"].status == "eligible"


def test_risk_reason_vocabulary_is_explicit_and_deterministic() -> None:
    candidate = FinalPoolCandidate(
        "위험어",
        "동사",
        _metadata(),
        FrequencyEvidence(
            found=True,
            selected_frequency=1_000,
            source="lemma",
            percentile=99.5,
            calibration_status="high_count_risk",
            high_count_risk=True,
            manual_review_required=True,
            risk_flags=("derivational_review",),
        ),
    )

    evaluation = evaluate_candidates([candidate])[0]

    assert evaluation.status == "review_required"
    assert evaluation.reasons == (
        "derivational_review",
        "high_count_risk",
        "manual_review",
    )
    assert evaluation.manual_review_required is True


def test_genre_csv_adapter_and_audit_output_are_schema_decoupled(tmp_path: Path) -> None:
    genre_path = tmp_path / "genre.csv"
    fields = (
        "canonical_word",
        "pos",
        "newspaper_raw",
        "dialogue_raw",
        "online_raw",
        "genre_coverage",
        "mean_percentile",
        "median_percentile",
        "max_percentile",
    )
    with genre_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "canonical_word": "단어",
                "pos": "AGGREGATED",
                "newspaper_raw": 10,
                "dialogue_raw": "",
                "online_raw": 5,
                "genre_coverage": 2,
                "mean_percentile": 0.7,
                "median_percentile": 0.7,
                "max_percentile": 0.8,
            }
        )
    genre = load_genre_comparison_evidence(genre_path)
    evaluations = evaluate_candidates(
        [
            FinalPoolCandidate(
                "단어",
                "명사",
                _metadata(),
                _frequency(),
                genre[("단어", "AGGREGATED")],
                genre_match_type="aggregated",
            )
        ]
    )
    output = tmp_path / "audit.csv"

    write_candidate_audit(output, evaluations)

    with output.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["candidate_status"] == "eligible"
    assert row["available_evidence"] == (
        "answer_candidate_metadata|frequency_evidence|genre_comparison_evidence"
    )
    assert row["genre_coverage"] == "2"
    assert row["genre_evidence_pos"] == "AGGREGATED"
    assert row["genre_match_type"] == "aggregated"
    assert row["manual_review_required"] == "false"


def test_production_aggregated_genre_row_joins_candidate_pos_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    genre_path = tmp_path / "production_genre_comparison.csv"
    genre_fields = (
        "canonical_word",
        "pos",
        "newspaper_raw",
        "newspaper_relative",
        "newspaper_log",
        "newspaper_percentile",
        "dialogue_raw",
        "dialogue_relative",
        "dialogue_log",
        "dialogue_percentile",
        "online_raw",
        "online_relative",
        "online_log",
        "online_percentile",
        "genre_coverage",
        "mean_percentile",
        "median_percentile",
        "max_percentile",
    )
    with genre_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=genre_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "canonical_word": "사람",
                "pos": "AGGREGATED",
                "newspaper_raw": 100,
                "newspaper_relative": 10,
                "newspaper_log": 2.397895,
                "newspaper_percentile": 0.8,
                "dialogue_raw": 20,
                "dialogue_relative": 10,
                "dialogue_log": 2.397895,
                "dialogue_percentile": 0.7,
                "online_raw": 50,
                "online_relative": 10,
                "online_log": 2.397895,
                "online_percentile": 0.9,
                "genre_coverage": 3,
                "mean_percentile": 0.8,
                "median_percentile": 0.8,
                "max_percentile": 0.9,
            }
        )
    evidence = load_genre_comparison_evidence(genre_path)
    candidate_path = tmp_path / "candidate.csv"
    candidate_row = _enriched_row("사람", "명사")
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(candidate_row), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(candidate_row)

    candidate = load_final_pool_candidates(
        candidate_path, genre_evidence=evidence
    )[0]

    assert candidate.pos == "명사"
    assert candidate.genre is not None
    assert candidate.genre.pos == "AGGREGATED"
    assert candidate.genre.genre_coverage == 3
    assert candidate.genre_match_type == "aggregated"
    audit_row = evaluate_candidates([candidate])[0].as_audit_row()
    assert audit_row["genre_evidence_pos"] == "AGGREGATED"
    assert audit_row["genre_match_type"] == "aggregated"


def test_pos_specific_genre_row_takes_precedence_over_aggregated_fallback(
    tmp_path: Path,
) -> None:
    candidate_path = tmp_path / "candidate.csv"
    candidate_row = _enriched_row("보다", "동사")
    with candidate_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=tuple(candidate_row), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(candidate_row)
    evidence = {
        ("보다", "AGGREGATED"): _genre(1),
        ("보다", "동사"): GenreComparisonEvidence("동사", 2, ("newspaper", "dialogue")),
    }

    candidate = load_final_pool_candidates(
        candidate_path, genre_evidence=evidence
    )[0]

    assert candidate.genre is not None
    assert candidate.genre.pos == "동사"
    assert candidate.genre.genre_coverage == 2
    assert candidate.genre_match_type == "exact"
    output = tmp_path / "audit.csv"
    write_candidate_audit(output, evaluate_candidates([candidate]))
    with output.open(encoding="utf-8", newline="") as handle:
        audit_row = next(csv.DictReader(handle))
    assert audit_row["genre_evidence_pos"] == "동사"
    assert audit_row["genre_match_type"] == "exact"


def test_enriched_candidate_loader_preserves_metadata_and_baseline(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    row = _enriched_row(" 단어 ", "명사")
    row.update(
        {
            "is_technical": "true",
            "wiktionary_labels": "일반|검토",
            "domain_labels": "기술",
            "review_required": "true",
            "review_reason": "technical_term",
            "hybrid_risk_flags": "",
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)

    candidate = load_final_pool_candidates(
        path, provisional_words=("단어",)
    )[0]

    assert candidate.canonical_word == "단어"
    assert candidate.metadata.is_technical is True
    assert candidate.metadata.review_reasons == ("technical_term",)
    assert candidate.frequency.selected_frequency == 42
    assert candidate.frequency.percentile == 75.5
    assert candidate.genre is None
    assert candidate.genre_match_type == "none"
    assert candidate.in_provisional_pool_baseline is True
    evaluation = evaluate_candidates([candidate])[0]
    assert evaluation.reasons == ("manual_review",)
    output = tmp_path / "audit.csv"
    write_candidate_audit(output, [evaluation])
    with output.open(encoding="utf-8", newline="") as handle:
        audit_row = next(csv.DictReader(handle))
    assert audit_row["source_review_reasons"] == "technical_term"
    assert audit_row["genre_evidence_pos"] == ""
    assert audit_row["genre_match_type"] == "none"


@pytest.mark.parametrize("percentile", ["-0.1", "100.1"])
def test_candidate_loader_rejects_frequency_percentile_outside_0_to_100(
    tmp_path: Path, percentile: str
) -> None:
    path = tmp_path / "candidates.csv"
    row = _enriched_row("단어", "명사")
    row["frequency_percentile"] = percentile
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(FinalPoolEvaluationError, match="outside 0.0..100.0"):
        load_final_pool_candidates(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("newspaper_percentile", "-0.1"),
        ("newspaper_percentile", "1.1"),
        ("mean_percentile", "1.1"),
        ("median_percentile", "-0.1"),
        ("max_percentile", "1.1"),
    ],
)
def test_genre_loader_rejects_percentiles_outside_0_to_1(
    tmp_path: Path, field: str, value: str
) -> None:
    path = tmp_path / "genre.csv"
    row = {
        "canonical_word": "단어",
        "pos": "명사",
        "newspaper_raw": "10",
        "newspaper_percentile": "0.7",
        "dialogue_raw": "",
        "online_raw": "",
        "genre_coverage": "1",
        "mean_percentile": "0.7",
        "median_percentile": "0.7",
        "max_percentile": "0.7",
    }
    row[field] = value
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(FinalPoolEvaluationError, match="outside 0.0..1.0"):
        load_genre_comparison_evidence(path)


def _enriched_row(word: str, pos: str) -> dict[str, str]:
    return {
        "word": word,
        "pos": pos,
        "is_proper_noun": "false",
        "is_general_lexical_pos": "true",
        "is_archaic": "false",
        "is_dialect": "false",
        "is_historical": "false",
        "is_technical": "false",
        "wiktionary_labels": "",
        "domain_labels": "",
        "review_required": "false",
        "review_reason": "",
        "selected_frequency": "42",
        "selected_frequency_source": "raw",
        "frequency_found": "true",
        "frequency_percentile": "75.5",
        "frequency_calibration_status": "normal",
        "manual_review_required": "false",
        "high_count_risk": "false",
        "hybrid_risk_flags": "",
    }
