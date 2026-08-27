#!/usr/bin/env python3
"""Verify completed Newspaper/Dialogue/Online raw-frequency outputs read-only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.post_run_verifier import (
    GENRES,
    format_human_summary,
    verify_outputs,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory containing the per-genre production outputs.",
    )
    parser.add_argument(
        "--genre",
        action="append",
        choices=GENRES,
        dest="genres",
        help="Verify one genre; repeat as needed. Default: all three.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for the machine-readable verification summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    genres = tuple(args.genres or GENRES)
    result = verify_outputs(args.output_dir, genres)  # type: ignore[arg-type]
    print(format_human_summary(result))
    if args.json_output is not None:
        payload = json.dumps(result.as_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        try:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(payload + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"Could not write JSON summary: {exc}", file=sys.stderr)
            return 2
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
