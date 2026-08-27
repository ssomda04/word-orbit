#!/usr/bin/env python3
"""Compare production genre-frequency CSV and report pairs."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.genre_frequency import (
    GENRES,
    GenreFrequencyError,
    compare_genre_frequencies,
    load_production_genre_frequency,
)

OUTPUT_FIELDS = (
    "canonical_word",
    "pos",
    *(
        f"{genre}_{metric}"
        for genre in GENRES
        for metric in ("raw", "relative", "log", "percentile")
    ),
    "genre_coverage",
    "mean_percentile",
    "median_percentile",
    "max_percentile",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare genre frequency normalizations.")
    for genre in GENRES:
        parser.add_argument(f"--{genre}", type=Path, required=True)
        parser.add_argument(f"--{genre}-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records = tuple(
            record
            for genre in GENRES
            for record in load_production_genre_frequency(
                getattr(args, genre), getattr(args, f"{genre}_report"), genre
            )
        )
        comparisons = compare_genre_frequencies(records)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=OUTPUT_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            for comparison in comparisons:
                row: dict[str, object] = {
                    "canonical_word": comparison.canonical_word,
                    "pos": comparison.pos,
                    "genre_coverage": comparison.genre_coverage,
                    "mean_percentile": comparison.mean_percentile,
                    "median_percentile": comparison.median_percentile,
                    "max_percentile": comparison.max_percentile,
                }
                for genre in GENRES:
                    value = getattr(comparison, genre)
                    row.update(
                        {
                            f"{genre}_raw": None if value is None else value.raw_count,
                            f"{genre}_relative": (
                                None
                                if value is None
                                else value.relative_frequency_per_million
                            ),
                            f"{genre}_log": (
                                None
                                if value is None
                                else value.log1p_relative_frequency
                            ),
                            f"{genre}_percentile": (
                                None if value is None else value.empirical_percentile
                            ),
                        }
                    )
                writer.writerow(row)
    except (GenreFrequencyError, OSError) as exc:
        print(f"Genre frequency comparison failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
