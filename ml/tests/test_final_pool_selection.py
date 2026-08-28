"""Tests for the approved final answer-pool selection boundary."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest
from contextle_eval.final_pool_evaluation import (
    AnswerCandidateMetadata,
    FinalPoolCandidate,
    FrequencyEvidence,
    GenreComparisonEvidence,
    evaluate_candidates,
)
from contextle_eval.final_pool_selection import (
    FinalPoolSelectionError,
    evidence_gap_reviews,
    select_final_pool,
    write_final_pool_outputs,
)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_final_answer_pool.py"
SPEC = importlib.util.spec_from_file_location("build_final_answer_pool", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
build_final_answer_pool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_final_answer_pool)


def test_cli_defaults_evidence_gap_to_sibling_csv_file() -> None:
    args = build_final_answer_pool.parse_args([])

    assert args.evidence_gap_output == Path(r"C:\data\evidence_gap_review.csv")
    assert args.evidence_gap_output != Path(r"C:\data\evidence_gap_review\.csv")


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


def _frequency() -> FrequencyEvidence:
    return FrequencyEvidence(
        found=True,
        selected_frequency=100,
        source="raw",
        percentile=80.0,
        calibration_status="normal",
        high_count_risk=False,
        manual_review_required=False,
        risk_flags=(),
    )


def _candidate(
    word: str,
    *,
    coverage: int | None = 2,
    mean: float = 0.20,
    median: float = 0.20,
    metadata: AnswerCandidateMetadata | None = None,
    provisional: bool = False,
    pos: str = "명사",
) -> FinalPoolCandidate:
    genre = (
        None
        if coverage is None
        else GenreComparisonEvidence(
            pos="AGGREGATED",
            genre_coverage=coverage,
            observed_genres=("newspaper", "online")[:coverage],
            mean_percentile=mean,
            median_percentile=median,
            max_percentile=0.0,
        )
    )
    return FinalPoolCandidate(
        canonical_word=word,
        pos=pos,
        metadata=metadata or _metadata(),
        frequency=_frequency(),
        genre=genre,
        in_provisional_pool_baseline=provisional,
        genre_match_type="none" if genre is None else "aggregated",
    )


@pytest.mark.parametrize(
    ("candidate", "expected_reason"),
    [
        (_candidate("단일장르", coverage=1), "genre_coverage_below_2"),
        (_candidate("낮은평균", mean=0.199999), "mean_percentile_below_0.20"),
        (_candidate("낮은중앙", median=0.199999), "median_percentile_below_0.20"),
    ],
)
def test_balanced_gate_rejects_each_failed_boundary(
    candidate: FinalPoolCandidate, expected_reason: str
) -> None:
    selection = select_final_pool(evaluate_candidates([candidate]))[0]

    assert selection.genre_policy_pass is False
    assert selection.final_selected is False
    assert expected_reason in selection.final_selection_reasons


def test_balanced_gate_is_inclusive_at_exactly_point_two_and_ignores_max() -> None:
    selection = select_final_pool(
        evaluate_candidates([_candidate("경계값", mean=0.20, median=0.20)])
    )[0]

    assert selection.genre_policy_pass is True
    assert selection.final_selected is True
    assert selection.final_selection_reasons == ("selected",)


@pytest.mark.parametrize(
    "metadata",
    [_metadata(review=True), _metadata(proper=True)],
)
def test_existing_review_or_exclusion_never_enters_final_pool(
    metadata: AnswerCandidateMetadata,
) -> None:
    selection = select_final_pool(
        evaluate_candidates([_candidate("메타데이터탈락", metadata=metadata)])
    )[0]

    assert selection.genre_policy_pass is True
    assert selection.final_selected is False
    assert selection.final_selection_reasons[0].startswith("existing_evaluator_")


def test_no_evidence_is_rejected_and_provisional_is_kept_in_review_lane() -> None:
    selection = select_final_pool(
        evaluate_candidates(
            [_candidate("가능하다", coverage=None, provisional=True)]
        )
    )[0]

    assert selection.genre_policy_pass is False
    assert selection.final_selected is False
    assert selection.final_selection_reasons == ("no_genre_evidence",)
    assert evidence_gap_reviews([selection]) == (selection,)


def test_outputs_are_deterministic_complete_and_preserve_evidence_gap(
    tmp_path: Path,
) -> None:
    selections = select_final_pool(
        evaluate_candidates(
            [
                _candidate("하늘"),
                _candidate("가능하다", coverage=None, provisional=True),
                _candidate("바다"),
            ]
        )
    )
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        write_final_pool_outputs(
            pool_path=root / "pool.txt",
            audit_path=root / "audit.csv",
            evidence_gap_path=root / "gap.csv",
            selections=selections,
        )

    assert (first / "pool.txt").read_text(encoding="utf-8") == "바다\n하늘\n"
    for name in ("pool.txt", "audit.csv", "gap.csv"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    with (first / "audit.csv").open(encoding="utf-8", newline="") as handle:
        audit = list(csv.DictReader(handle))
    with (first / "gap.csv").open(encoding="utf-8", newline="") as handle:
        gaps = list(csv.DictReader(handle))
    assert len(audit) == 3
    assert [row["word"] for row in gaps] == ["가능하다"]
    assert gaps[0]["final_selection_reason"] == "no_genre_evidence"
    assert gaps[0]["frequency_found"] == "true"


def test_duplicate_canonical_words_are_rejected_before_output() -> None:
    candidates = [
        _candidate("중복", pos="명사"),
        _candidate("중복", pos="부사"),
    ]

    with pytest.raises(FinalPoolSelectionError, match="duplicate canonical words"):
        select_final_pool(evaluate_candidates(candidates))
