"""Apply the approved genre policy after policy-neutral candidate evaluation."""

from __future__ import annotations

import csv
import os
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from contextle_eval.final_pool_evaluation import CandidateEvaluation, FinalPoolCandidate

MIN_GENRE_COVERAGE = 2
MIN_MEAN_PERCENTILE = 0.20
MIN_MEDIAN_PERCENTILE = 0.20

FINAL_AUDIT_FIELDS = (
    "word",
    "candidate_pos",
    "existing_evaluator_status",
    "existing_evaluator_reasons",
    "source_review_reasons",
    "provisional_membership",
    "genre_evidence_present",
    "genre_evidence_pos",
    "genre_match_type",
    "genre_coverage",
    "mean_percentile",
    "median_percentile",
    "max_percentile",
    "genre_policy_pass",
    "final_selected",
    "final_selection_reason",
)

EVIDENCE_GAP_FIELDS = (
    *FINAL_AUDIT_FIELDS,
    "frequency_found",
    "selected_frequency",
    "selected_frequency_source",
    "frequency_percentile",
    "frequency_calibration_status",
    "frequency_manual_review_required",
    "high_count_risk",
    "hybrid_risk_flags",
)


class FinalPoolSelectionError(ValueError):
    """Raised when final selection or output integrity is invalid."""


def _format_bool(value: bool) -> str:
    return "true" if value else "false"


@dataclass(frozen=True, slots=True)
class FinalPoolSelection:
    """One evaluated candidate plus the approved final-selection decision."""

    evaluation: CandidateEvaluation
    genre_policy_pass: bool
    final_selected: bool
    final_selection_reasons: tuple[str, ...]

    def as_audit_row(self) -> dict[str, str | int]:
        candidate = self.evaluation.candidate
        genre = candidate.genre
        return {
            "word": candidate.canonical_word,
            "candidate_pos": candidate.pos,
            "existing_evaluator_status": self.evaluation.status,
            "existing_evaluator_reasons": "|".join(self.evaluation.reasons),
            "source_review_reasons": "|".join(candidate.metadata.review_reasons),
            "provisional_membership": _format_bool(
                candidate.in_provisional_pool_baseline
            ),
            "genre_evidence_present": _format_bool(genre is not None),
            "genre_evidence_pos": "" if genre is None else genre.pos,
            "genre_match_type": candidate.genre_match_type,
            "genre_coverage": "" if genre is None else genre.genre_coverage,
            "mean_percentile": "" if genre is None else _format_float(genre.mean_percentile),
            "median_percentile": (
                "" if genre is None else _format_float(genre.median_percentile)
            ),
            "max_percentile": "" if genre is None else _format_float(genre.max_percentile),
            "genre_policy_pass": _format_bool(self.genre_policy_pass),
            "final_selected": _format_bool(self.final_selected),
            "final_selection_reason": "|".join(self.final_selection_reasons),
        }

    def as_evidence_gap_row(self) -> dict[str, str | int]:
        candidate = self.evaluation.candidate
        frequency = candidate.frequency
        return {
            **self.as_audit_row(),
            "frequency_found": _format_bool(frequency.found),
            "selected_frequency": (
                "" if frequency.selected_frequency is None else frequency.selected_frequency
            ),
            "selected_frequency_source": frequency.source,
            "frequency_percentile": _format_float(frequency.percentile),
            "frequency_calibration_status": frequency.calibration_status,
            "frequency_manual_review_required": _format_bool(
                frequency.manual_review_required
            ),
            "high_count_risk": _format_bool(frequency.high_count_risk),
            "hybrid_risk_flags": "|".join(frequency.risk_flags),
        }


def _format_float(value: float | None) -> str:
    return "" if value is None else repr(value)


def has_usable_genre_evidence(candidate: FinalPoolCandidate) -> bool:
    """Genre evidence is usable only when a row exists and observed at least one genre.

    A missing row and a row that observed zero genres are the same audit state: no
    observed genre evidence. Both must route to the evidence-gap lane rather than
    being read as a merely weak but present signal.
    """
    genre = candidate.genre
    return genre is not None and genre.genre_coverage > 0


def apply_approved_policy(evaluation: CandidateEvaluation) -> FinalPoolSelection:
    """Combine the existing audit status with the approved balanced genre gate."""
    candidate = evaluation.candidate
    genre = candidate.genre
    reasons: list[str] = []
    if evaluation.status != "eligible":
        reasons.append(f"existing_evaluator_{evaluation.status}")
    if not has_usable_genre_evidence(candidate):
        reasons.append("no_genre_evidence")
        genre_policy_pass = False
    else:
        if genre.genre_coverage < MIN_GENRE_COVERAGE:
            reasons.append("genre_coverage_below_2")
        if genre.mean_percentile is None:
            reasons.append("mean_percentile_missing")
        elif genre.mean_percentile < MIN_MEAN_PERCENTILE:
            reasons.append("mean_percentile_below_0.20")
        if genre.median_percentile is None:
            reasons.append("median_percentile_missing")
        elif genre.median_percentile < MIN_MEDIAN_PERCENTILE:
            reasons.append("median_percentile_below_0.20")
        genre_policy_pass = not any(
            reason.startswith(("genre_", "mean_", "median_")) for reason in reasons
        )
    final_selected = evaluation.status == "eligible" and genre_policy_pass
    return FinalPoolSelection(
        evaluation=evaluation,
        genre_policy_pass=genre_policy_pass,
        final_selected=final_selected,
        final_selection_reasons=("selected",) if final_selected else tuple(reasons),
    )


def select_final_pool(
    evaluations: Sequence[CandidateEvaluation],
) -> tuple[FinalPoolSelection, ...]:
    """Apply policy in deterministic order and reject word collisions."""
    selections = tuple(
        apply_approved_policy(evaluation)
        for evaluation in sorted(
            evaluations,
            key=lambda item: (
                item.candidate.canonical_word,
                item.candidate.pos,
            ),
        )
    )
    words = [selection.evaluation.candidate.canonical_word for selection in selections]
    if len(words) != len(set(words)):
        raise FinalPoolSelectionError("Candidate input contains duplicate canonical words.")
    for word in words:
        if not word or word != unicodedata.normalize("NFKC", word) or word != word.strip():
            raise FinalPoolSelectionError(
                f"Candidate word is not non-blank, stripped Unicode NFKC: {word!r}."
            )
        if "\n" in word or "\r" in word:
            raise FinalPoolSelectionError(f"Candidate word contains a line break: {word!r}.")
    return selections


def evidence_gap_reviews(
    selections: Sequence[FinalPoolSelection],
) -> tuple[FinalPoolSelection, ...]:
    """Return provisional no-evidence candidates without auto-approving them."""
    return tuple(
        selection
        for selection in selections
        if not has_usable_genre_evidence(selection.evaluation.candidate)
        and selection.evaluation.candidate.in_provisional_pool_baseline
        and not selection.final_selected
    )


def write_final_pool_outputs(
    *,
    pool_path: Path,
    audit_path: Path,
    evidence_gap_path: Path,
    selections: Sequence[FinalPoolSelection],
) -> None:
    """Write deterministic final, full-audit, and evidence-gap artifacts."""
    ordered = tuple(
        sorted(
            selections,
            key=lambda item: (
                item.evaluation.candidate.canonical_word,
                item.evaluation.candidate.pos,
            ),
        )
    )
    words = [
        selection.evaluation.candidate.canonical_word
        for selection in ordered
        if selection.final_selected
    ]
    if len(words) != len(set(words)):
        raise FinalPoolSelectionError("Final answer pool contains duplicate words.")
    if words != sorted(words):
        raise FinalPoolSelectionError("Final answer pool is not deterministically sorted.")
    gaps = evidence_gap_reviews(ordered)
    _atomic_write_text(pool_path, "".join(f"{word}\n" for word in words))
    _atomic_write_csv(
        audit_path,
        FINAL_AUDIT_FIELDS,
        [selection.as_audit_row() for selection in ordered],
    )
    _atomic_write_csv(
        evidence_gap_path,
        EVIDENCE_GAP_FIELDS,
        [selection.as_evidence_gap_row() for selection in gaps],
    )


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8", newline="")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise FinalPoolSelectionError(f"Could not write {path}: {exc}") from exc


def _atomic_write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, str | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise FinalPoolSelectionError(f"Could not write {path}: {exc}") from exc
