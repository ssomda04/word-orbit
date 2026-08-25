#!/usr/bin/env python3
"""Calibrate POS/source frequency percentiles and simulate answer cutoffs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.frequency_calibration import (
    CALIBRATION_FIELDS,
    DEFAULT_SEED,
    FrequencyCalibrationError,
    build_calibration_report,
    calibrate_candidates,
    format_calibration_report,
    load_hybrid_candidates,
    write_calibrated_csv,
)

DEFAULT_INPUT = ML_ROOT / "data" / "answer_candidates_with_hybrid_frequency.csv"
DEFAULT_OUTPUT = ML_ROOT / "data" / "answer_candidates_with_frequency_score.csv"
DEFAULT_REPORT = ML_ROOT / "data" / "frequency" / "cutoff_simulation_report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute source/POS bucket percentiles for matched hybrid frequencies "
            "and simulate P50/P60/P70/P80/P90 without filtering candidates."
        )
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--report-output", default=DEFAULT_REPORT, type=Path)
    parser.add_argument("--seed", default=DEFAULT_SEED, type=int)
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
        fieldnames, input_rows = load_hybrid_candidates(args.input)
        rows = calibrate_candidates(input_rows)
        write_calibrated_csv(args.output, fieldnames, rows)
        report = build_calibration_report(rows, seed=args.seed)
        rendered = format_calibration_report(report)
        _write_report(args.report_output, rendered)
    except (FrequencyCalibrationError, OSError) as exc:
        print(f"Frequency calibration failed: {exc}", file=sys.stderr)
        return 2

    print(f"Preserved {len(rows)} candidate rows")
    print(f"Added calibration columns: {', '.join(CALIBRATION_FIELDS)}")
    print(f"Wrote calibrated candidates to {args.output}")
    print(f"Wrote cutoff simulation report to {args.report_output}")
    print(f"Matched candidates: {report['matched_count']}")
    print(f"Unmatched candidates: {report['unmatched_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
