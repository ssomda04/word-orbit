#!/usr/bin/env python3
"""Build a human-review CSV of answer candidates from Korean Wiktionary."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.answer_candidates import (
    DEFAULT_LONG_WORD_LENGTH,
    AnswerCandidateError,
    build_answer_candidates,
    format_candidate_statistics,
)

DEFAULT_VOCABULARY = ML_ROOT / "data" / "game_words.txt"
DEFAULT_OUTPUT = ML_ROOT / "data" / "answer_candidates.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reviewable answer candidates from a Korean Wiktionary dump."
    )
    parser.add_argument("--dump-path", required=True, type=Path)
    parser.add_argument("--vocabulary", default=DEFAULT_VOCABULARY, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument(
        "--long-word-length",
        default=DEFAULT_LONG_WORD_LENGTH,
        type=int,
        help=f"Mark words of at least this length for review (default: {DEFAULT_LONG_WORD_LENGTH}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stats = build_answer_candidates(
            args.dump_path,
            args.vocabulary,
            args.output,
            long_word_length=args.long_word_length,
        )
    except AnswerCandidateError as exc:
        print(f"Candidate generation failed: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote answer candidates to {args.output}")
    print(format_candidate_statistics(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
