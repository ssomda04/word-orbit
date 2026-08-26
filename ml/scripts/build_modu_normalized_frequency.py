#!/usr/bin/env python3
"""Apply the Modu normalization contract and write collision audits."""

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
    file_metadata,
    load_answer_candidates,
    load_vocabulary,
)
from contextle_eval.modu_normalization import (  # noqa: E402
    ModuNormalizationError,
    build_normalization_report,
    detect_collisions,
    load_raw_frequency,
    normalize_frequencies,
    write_collision_audit,
    write_normalized_frequency,
    write_report,
)

DEFAULT_DIRECTORY = ML_ROOT / "data" / "modu_frequency"
DEFAULT_FREQUENCY = DEFAULT_DIRECTORY / "mp_frequency.csv"
DEFAULT_VOCABULARY = ML_ROOT / "data" / "game_words.txt"
DEFAULT_ANSWER_CANDIDATES = ML_ROOT / "data" / "answer_candidates.csv"
DEFAULT_NORMALIZED_OUTPUT = DEFAULT_DIRECTORY / "normalized_frequency.csv"
DEFAULT_REPORT_OUTPUT = DEFAULT_DIRECTORY / "normalization_report.json"
DEFAULT_COLLISION_OUTPUT = DEFAULT_DIRECTORY / "collision_audit.csv"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize complete Modu MP frequencies without merging collisions."
    )
    parser.add_argument("--frequency", type=Path, default=DEFAULT_FREQUENCY)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--answer-candidates", type=Path, default=DEFAULT_ANSWER_CANDIDATES)
    parser.add_argument("--normalized-output", type=Path, default=DEFAULT_NORMALIZED_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    parser.add_argument("--collision-output", type=Path, default=DEFAULT_COLLISION_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        raw_rows = load_raw_frequency(args.frequency)
        game_words = load_vocabulary(args.vocabulary)
        answer_words = load_answer_candidates(args.answer_candidates)
        normalized = normalize_frequencies(raw_rows, game_words, answer_words)
        collisions = detect_collisions(normalized)
        report = build_normalization_report(
            normalized,
            collisions,
            input_metadata={
                "frequency": file_metadata(args.frequency),
                "game_vocabulary": file_metadata(args.vocabulary),
                "answer_candidates": file_metadata(args.answer_candidates),
            },
        )
        write_normalized_frequency(args.normalized_output, normalized)
        write_collision_audit(args.collision_output, collisions)
        write_report(args.report_output, report)
    except (ModuNormalizationError, OSError) as exc:
        print(f"Modu normalization failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote normalized frequency to {args.normalized_output}")
    print(f"Wrote collision audit to {args.collision_output}")
    print(f"Wrote normalization report to {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
