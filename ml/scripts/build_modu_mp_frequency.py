#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "ijson>=3.4,<4",
# ]
# ///
"""Build raw NXMP/SXMP morpheme-form/POS frequencies from a Modu ZIP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.modu_corpus import ModuCorpusError
from contextle_eval.modu_frequency import (
    ModuFrequencyError,
    aggregate_mp_zip,
    write_outputs,
)

DEFAULT_OUTPUT_DIRECTORY = ML_ROOT / "data" / "modu_frequency"
DEFAULT_CSV_OUTPUT = DEFAULT_OUTPUT_DIRECTORY / "mp_frequency.csv"
DEFAULT_REPORT_OUTPUT = DEFAULT_OUTPUT_DIRECTORY / "mp_frequency_report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate complete NXMP/SXMP raw morpheme/POS frequencies."
    )
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = aggregate_mp_zip(args.zip_path)
        report = write_outputs(result, args.csv_output, args.report_output)
    except (ModuCorpusError, ModuFrequencyError, OSError) as exc:
        print(f"Modu MP frequency generation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Wrote frequency CSV to {args.csv_output}")
    print(f"Wrote audit report to {args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
