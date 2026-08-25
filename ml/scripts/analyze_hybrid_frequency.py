#!/usr/bin/env python3
"""Build POS-aware hybrid frequencies and print a deterministic audit report."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.hybrid_frequency import (
    DEFAULT_SAMPLE_SEED,
    HybridFrequencyError,
    build_hybrid_report,
    format_hybrid_report,
    join_hybrid_frequency,
    load_lemma_frequency,
    load_raw_frequency,
    load_special_sources,
)

DEFAULT_RAW = ML_ROOT / "data" / "frequency" / "korean_frequency.csv"
DEFAULT_LEMMA = ML_ROOT / "data" / "frequency" / "korean_frequency_lemma.csv"
DEFAULT_AUDIT = ML_ROOT / "data" / "frequency" / "korean_frequency_lemma_analysis.csv"
DEFAULT_CANDIDATES = ML_ROOT / "data" / "answer_candidates.csv"
DEFAULT_OUTPUT = ML_ROOT / "data" / "answer_candidates_with_hybrid_frequency.csv"
DEFAULT_REPORT = ML_ROOT / "data" / "frequency" / "korean_frequency_hybrid_report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select lemma frequency for predicate candidates and raw surface "
            "frequency for all other candidates without fallback or addition."
        )
    )
    parser.add_argument("--raw-frequency", default=DEFAULT_RAW, type=Path)
    parser.add_argument("--lemma-frequency", default=DEFAULT_LEMMA, type=Path)
    parser.add_argument("--lemma-audit", default=DEFAULT_AUDIT, type=Path)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--report-output", default=DEFAULT_REPORT, type=Path)
    parser.add_argument("--seed", default=DEFAULT_SAMPLE_SEED, type=int)
    return parser.parse_args(argv)


def _write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(f"{content}\n", encoding="utf-8")
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        raw = load_raw_frequency(args.raw_frequency)
        lemma = load_lemma_frequency(args.lemma_frequency)
        special_sources = load_special_sources(args.lemma_audit)
        joined = join_hybrid_frequency(args.candidates, raw, lemma, args.output)
        report = build_hybrid_report(joined, raw, lemma, special_sources, seed=args.seed)
        rendered = format_hybrid_report(report)
        _write_report(args.report_output, rendered)
    except (HybridFrequencyError, OSError) as exc:
        print(f"Hybrid frequency analysis failed: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote POS-aware hybrid candidates to {args.output}")
    print(f"Wrote hybrid analysis report to {args.report_output}")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
