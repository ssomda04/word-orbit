#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kiwipiepy==0.23.2",
# ]
# ///
"""Build conservative Kiwi lemma frequencies and compare candidate coverage."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.lemma_frequency import (
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_SAMPLE_SEED,
    DEFAULT_TOP_N,
    LemmaFrequencyError,
    analyze_frequencies,
    build_report,
    format_report,
    join_candidates,
    load_surface_frequency,
    write_analysis_csv,
    write_lemma_csv,
)

DEFAULT_FREQUENCY = ML_ROOT / "data" / "frequency" / "ko_50k.txt"
DEFAULT_CANDIDATES = ML_ROOT / "data" / "answer_candidates.csv"
DEFAULT_LEMMA_OUTPUT = ML_ROOT / "data" / "frequency" / "korean_frequency_lemma.csv"
DEFAULT_ANALYSIS_OUTPUT = (
    ML_ROOT / "data" / "frequency" / "korean_frequency_lemma_analysis.csv"
)
DEFAULT_CANDIDATE_OUTPUT = ML_ROOT / "data" / "answer_candidates_with_lemma_frequency.csv"
DEFAULT_REPORT_OUTPUT = ML_ROOT / "data" / "frequency" / "korean_frequency_lemma_report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze surface frequencies with Kiwi, aggregate only confident single "
            "lemma/POS assignments, and compare answer-candidate coverage."
        )
    )
    parser.add_argument("--frequency", default=DEFAULT_FREQUENCY, type=Path)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES, type=Path)
    parser.add_argument("--lemma-output", default=DEFAULT_LEMMA_OUTPUT, type=Path)
    parser.add_argument("--analysis-output", default=DEFAULT_ANALYSIS_OUTPUT, type=Path)
    parser.add_argument("--candidate-output", default=DEFAULT_CANDIDATE_OUTPUT, type=Path)
    parser.add_argument("--report-output", default=DEFAULT_REPORT_OUTPUT, type=Path)
    parser.add_argument("--top-n", default=DEFAULT_TOP_N, type=int)
    parser.add_argument(
        "--ambiguity-margin", default=DEFAULT_AMBIGUITY_MARGIN, type=float
    )
    parser.add_argument("--seed", default=DEFAULT_SAMPLE_SEED, type=int)
    parser.add_argument("--model-type", choices=("cong", "cong-global"), default="cong")
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
        import kiwipiepy
        from kiwipiepy import Kiwi

        rows = load_surface_frequency(args.frequency)
        kiwi = Kiwi(model_type=args.model_type)
        analyses, aggregates = analyze_frequencies(
            kiwi,
            rows,
            top_n=args.top_n,
            ambiguity_margin=args.ambiguity_margin,
        )
        write_analysis_csv(args.analysis_output, analyses)
        write_lemma_csv(args.lemma_output, aggregates)
        joined = join_candidates(args.candidates, aggregates, args.candidate_output)
        report = build_report(
            analyses,
            aggregates,
            joined,
            rows,
            kiwi_version=kiwipiepy.__version__,
            top_n=args.top_n,
            ambiguity_margin=args.ambiguity_margin,
            seed=args.seed,
        )
        rendered = format_report(report)
        _write_report(args.report_output, rendered)
    except (LemmaFrequencyError, OSError, ImportError) as exc:
        print(f"Lemma frequency analysis failed: {exc}", file=sys.stderr)
        return 2

    print(f"Kiwi version: {kiwipiepy.__version__}")
    print(f"Wrote surface audit to {args.analysis_output}")
    print(f"Wrote exact lemma aggregates to {args.lemma_output}")
    print(f"Wrote lemma-joined candidates to {args.candidate_output}")
    print(f"Wrote analysis report to {args.report_output}")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
