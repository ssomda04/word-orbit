#!/usr/bin/env python3
"""Build the approved final answer pool and complete selection audits."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.final_pool_evaluation import (
    FinalPoolEvaluationError,
    evaluate_candidates,
    load_final_pool_candidates,
    load_genre_comparison_evidence,
)
from contextle_eval.final_pool_selection import (
    FinalPoolSelectionError,
    evidence_gap_reviews,
    select_final_pool,
    write_final_pool_outputs,
)

DEFAULT_CANDIDATES = ML_ROOT / "data" / "answer_candidates_with_frequency_score.csv"
DEFAULT_PROVISIONAL = ML_ROOT / "data" / "answer_pool_provisional.txt"
DEFAULT_GENRE_COMPARISON = Path(r"C:\data\genre_comparison.csv")
DEFAULT_POOL_OUTPUT = Path(r"C:\data\final_answer_pool.txt")
DEFAULT_AUDIT_OUTPUT = Path(r"C:\data\final_answer_pool_audit.csv")
DEFAULT_EVIDENCE_GAP_OUTPUT = Path(r"C:\data\evidence_gap_review.csv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--provisional", type=Path, default=DEFAULT_PROVISIONAL)
    parser.add_argument("--genre-comparison", type=Path, default=DEFAULT_GENRE_COMPARISON)
    parser.add_argument("--pool-output", type=Path, default=DEFAULT_POOL_OUTPUT)
    parser.add_argument("--audit-output", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument(
        "--evidence-gap-output", type=Path, default=DEFAULT_EVIDENCE_GAP_OUTPUT
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        provisional_words = args.provisional.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        genre_evidence = load_genre_comparison_evidence(args.genre_comparison)
        candidates = load_final_pool_candidates(
            args.candidates,
            genre_evidence=genre_evidence,
            provisional_words=provisional_words,
        )
        evaluations = evaluate_candidates(candidates)
        selections = select_final_pool(evaluations)
        write_final_pool_outputs(
            pool_path=args.pool_output,
            audit_path=args.audit_output,
            evidence_gap_path=args.evidence_gap_output,
            selections=selections,
        )
    except (OSError, FinalPoolEvaluationError, FinalPoolSelectionError) as exc:
        print(f"Final answer-pool generation failed: {exc}", file=sys.stderr)
        return 2

    reason_counts = Counter(
        selection.final_selection_reasons for selection in selections
    )
    summary = {
        "candidates": len(selections),
        "genre_evidence": sum(
            selection.evaluation.candidate.genre is not None
            for selection in selections
        ),
        "genre_policy_pass": sum(
            selection.genre_policy_pass for selection in selections
        ),
        "final_selected": sum(selection.final_selected for selection in selections),
        "evidence_gap_review": len(evidence_gap_reviews(selections)),
        "selection_reasons": {
            "|".join(reasons): count
            for reasons, count in sorted(reason_counts.items())
        },
        "outputs": {
            "pool": str(args.pool_output),
            "audit": str(args.audit_output),
            "evidence_gap": str(args.evidence_gap_output),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
