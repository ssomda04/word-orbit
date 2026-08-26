#!/usr/bin/env python3
"""Analyze Modu MP base-form strategies without selecting a final answer pool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.modu_baseform_analysis import (  # noqa: E402
    ModuBaseformAnalysisError,
    build_analysis_report,
    file_metadata,
    load_answer_candidates,
    load_frequency_entries,
    load_vocabulary,
    write_report,
)

DEFAULT_FREQUENCY = ML_ROOT / "data" / "modu_frequency" / "mp_frequency.csv"
DEFAULT_VOCABULARY = ML_ROOT / "data" / "game_words.txt"
DEFAULT_ANSWER_CANDIDATES = ML_ROOT / "data" / "answer_candidates.csv"
DEFAULT_OUTPUT = ML_ROOT / "data" / "modu_frequency" / "baseform_analysis.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare vocabulary-backed base-form strategies for Modu MP counts."
    )
    parser.add_argument("--frequency", type=Path, default=DEFAULT_FREQUENCY)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--answer-candidates", type=Path, default=DEFAULT_ANSWER_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        entries = load_frequency_entries(args.frequency)
        vocabulary = load_vocabulary(args.vocabulary)
        answer_candidates = load_answer_candidates(args.answer_candidates)
        report = build_analysis_report(
            entries,
            vocabulary,
            answer_candidates,
            input_metadata={
                "frequency": file_metadata(args.frequency),
                "game_vocabulary": file_metadata(args.vocabulary),
                "answer_candidates": file_metadata(args.answer_candidates),
            },
        )
        write_report(args.output, report)
    except (ModuBaseformAnalysisError, OSError) as exc:
        print(f"Modu base-form analysis failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote analysis report to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
