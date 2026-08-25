#!/usr/bin/env python3
"""Join corpus frequency data to answer candidates and print analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.frequency_analysis import (
    DEFAULT_SAMPLE_SEED,
    DEFAULT_SAMPLE_SIZE,
    FrequencyAnalysisError,
    format_frequency_analysis,
    join_answer_candidate_frequency,
)

DEFAULT_CANDIDATES = ML_ROOT / "data" / "answer_candidates.csv"
DEFAULT_OUTPUT = ML_ROOT / "data" / "answer_candidates_with_frequency.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Join real corpus frequency data to answer candidates and analyze coverage."
    )
    parser.add_argument("--frequency", required=True, type=Path)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--value-column", choices=("count", "frequency"))
    parser.add_argument("--duplicate-policy", choices=("sum", "error"), default="sum")
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        analysis = join_answer_candidate_frequency(
            args.candidates,
            args.frequency,
            args.output,
            value_column=args.value_column,
            duplicate_policy=args.duplicate_policy,
            seed=args.seed,
            sample_size=args.sample_size,
        )
    except FrequencyAnalysisError as exc:
        print(f"Frequency analysis failed: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote frequency-joined candidates to {args.output}")
    print(format_frequency_analysis(analysis))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
