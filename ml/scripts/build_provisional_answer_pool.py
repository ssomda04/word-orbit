#!/usr/bin/env python3
"""Create a development-only provisional answer pool from saved calibration.

Development-only provisional answer pool derived from FrequencyWords calibration.
Do not treat as the final answer pool.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.frequency_calibration import canonical_pos_parts
from contextle_eval.provisional_answer_pool import (
    DEFAULT_PERCENTILE_CUTOFF,
    ProvisionalAnswerPoolError,
    load_calibrated_candidates,
    select_provisional_entries,
    validate_reference_membership,
    write_provisional_pool,
)

DEFAULT_INPUT = ML_ROOT / "data" / "answer_candidates_with_frequency_score.csv"
DEFAULT_GAME_WORDS = ML_ROOT / "data" / "game_words.txt"
DEFAULT_ANSWER_CANDIDATES = ML_ROOT / "data" / "answer_candidates.csv"
DEFAULT_OUTPUT = ML_ROOT / "data" / "answer_pool_provisional.txt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only provisional answer pool derived from FrequencyWords "
            "calibration. Do not treat as the final answer pool."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--game-words", type=Path, default=DEFAULT_GAME_WORDS)
    parser.add_argument("--answer-candidates", type=Path, default=DEFAULT_ANSWER_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--percentile-cutoff", type=float, default=DEFAULT_PERCENTILE_CUTOFF)
    return parser.parse_args(argv)


def _load_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8-sig").splitlines()


def _load_candidate_words(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "word" not in reader.fieldnames:
            raise ProvisionalAnswerPoolError(f"Answer candidates CSV lacks word column: {path}")
        return [row["word"] for row in reader]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = load_calibrated_candidates(args.input)
        entries = select_provisional_entries(rows, percentile_cutoff=args.percentile_cutoff)
        validate_reference_membership(
            entries,
            game_words=_load_lines(args.game_words),
            answer_candidate_words=_load_candidate_words(args.answer_candidates),
        )
        write_provisional_pool(args.output, entries)
    except (OSError, ProvisionalAnswerPoolError) as exc:
        print(f"Provisional answer-pool generation failed: {exc}", file=sys.stderr)
        return 2

    pos_counts = Counter(
        pos
        for entry in entries
        for pos in canonical_pos_parts(entry.pos)
        if pos in {"noun", "verb", "adjective", "adverb"}
    )
    print(f"Wrote {len(entries)} development-only provisional words to {args.output}")
    print(f"Percentile cutoff: P{args.percentile_cutoff:g}")
    print(f"POS counts: {dict(sorted(pos_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

