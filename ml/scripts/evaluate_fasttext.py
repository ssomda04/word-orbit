#!/usr/bin/env python3
"""Evaluate a local Korean FastText model against the Contextle dataset."""

from __future__ import annotations

import argparse
import importlib.metadata
import random
import sys
import time
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.dataset import DatasetValidationError, load_dataset
from contextle_eval.fasttext_provider import FastTextLoadError, FastTextVectorProvider
from contextle_eval.metrics import evaluate
from contextle_eval.reporting import ReportWriteError, write_reports


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse portable path and reproducibility options."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a local Facebook FastText .bin model. "
            "The script never downloads a model."
        )
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New or empty versioned directory; existing results are never overwritten.",
    )
    parser.add_argument("--top-k", default=10, type=_positive_integer)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args(argv)


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    """Run evaluation and return a process exit code."""
    args = parse_args(argv)
    random.seed(args.seed)
    total_started = time.perf_counter()

    try:
        dataset = load_dataset(args.dataset)

        load_started = time.perf_counter()
        provider = FastTextVectorProvider.load(args.model_path)
        model_load_seconds = time.perf_counter() - load_started

        result = evaluate(dataset, provider, top_k=args.top_k, seed=args.seed)
        result.summary["timingSeconds"]["modelLoad"] = model_load_seconds
        result.summary["timingSeconds"]["totalRun"] = time.perf_counter() - total_started

        run_config = {
            "modelFileName": args.model_path.name,
            "datasetFileName": args.dataset.name,
            "datasetName": dataset.dataset_name,
            "datasetVersion": dataset.version,
            "outputDirectoryName": args.output_dir.name,
            "topK": args.top_k,
            "seed": args.seed,
            "pythonVersion": sys.version.split()[0],
            "libraryVersions": {
                "fasttext-wheel": _package_version("fasttext-wheel"),
            },
        }
        write_reports(args.output_dir, result, run_config)
    except (DatasetValidationError, FastTextLoadError, ReportWriteError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        return 2

    failure_count = len(result.summary["vectorFailures"])
    print(f"Wrote FastText evaluation reports to {args.output_dir}")
    if failure_count:
        print(
            f"Evaluation completed with {failure_count} vector failure(s); "
            "see summary.json.",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
