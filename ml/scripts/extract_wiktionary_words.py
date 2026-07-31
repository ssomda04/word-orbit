#!/usr/bin/env python3
"""Extract Korean game-word headwords from a local Korean Wiktionary dump."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.wiktionary_words import (  # noqa: E402
    WiktionaryExtractionError,
    extract_dump,
    format_statistics,
)

DEFAULT_OUTPUT = ML_ROOT / "data" / "game_words.txt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream a local kowiktionary pages-articles XML bz2 dump and write "
            "normalized Korean game-word headwords."
        )
    )
    parser.add_argument(
        "--dump-path",
        required=True,
        type=Path,
        help="Local kowiktionary-*-pages-articles.xml.bz2 file.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        type=Path,
        help=f"UTF-8 output path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        stats = extract_dump(args.dump_path, args.output)
    except WiktionaryExtractionError as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote Korean headwords to {args.output}")
    print(format_statistics(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
