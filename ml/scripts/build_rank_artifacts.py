#!/usr/bin/env python3
"""Build at most 50 provisional answer rank artifacts from a local FastText model."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from contextle_eval.fasttext_provider import FastTextLoadError, FastTextVectorProvider
from contextle_eval.rank_artifact import (
    DEFAULT_SEED,
    RankArtifactError,
    VocabularyIndex,
    artifact_size_bytes,
    build_artifact,
    build_normalized_vector_matrix,
    save_artifact_npy,
    save_artifact_npz,
    select_benchmark_answers,
)

DEFAULT_VOCABULARY = ML_ROOT / "data" / "game_words.txt"
DEFAULT_POOL = ML_ROOT / "data" / "answer_pool_provisional.txt"
DEFAULT_CANDIDATES = ML_ROOT / "data" / "answer_candidates_with_frequency_score.csv"
DEFAULT_OUTPUT = ML_ROOT / "data" / "artifacts" / "provisional"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 10 or 50 provisional rank artifacts.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--answer-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, choices=(10, 50), default=10)
    parser.add_argument("--format", choices=("npy", "npz"), default="npy")
    parser.add_argument("--similarity-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def load_answer_rows(pool_path: Path, candidates_path: Path) -> list[tuple[str, str]]:
    pool = pool_path.read_text(encoding="utf-8-sig").splitlines()
    with candidates_path.open(encoding="utf-8-sig", newline="") as handle:
        by_word = {row["word"]: row["pos"] for row in csv.DictReader(handle)}
    missing = [word for word in pool if word not in by_word]
    if missing:
        raise RankArtifactError(f"Pool words are missing candidate POS metadata: {missing[:10]}")
    return [(word, by_word[word]) for word in pool]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(f"Output directory must be new or empty: {args.output_dir}", file=sys.stderr)
        return 2
    started = time.perf_counter()
    try:
        vocabulary = VocabularyIndex.from_file(args.vocabulary)
        answers = select_benchmark_answers(
            load_answer_rows(args.answer_pool, args.candidates), count=args.count, seed=args.seed
        )
        provider = FastTextVectorProvider.load(args.model_path)
        matrix = build_normalized_vector_matrix(vocabulary, provider)
        paths = []
        for answer in answers:
            artifact = build_artifact(
                answer,
                vocabulary,
                matrix,
                embedding_model=args.model_path.name,
                similarity_dtype=args.similarity_dtype,
                ranktable_compatibility_provider=provider,
            )
            save = save_artifact_npy if args.format == "npy" else save_artifact_npz
            paths.append(save(args.output_dir, artifact))
        manifest = {
            "seed": args.seed,
            "answers": answers,
            "count": len(answers),
            "format": args.format,
            "similarity_dtype": args.similarity_dtype,
            "vocabulary_sha256": vocabulary.sha256,
            "total_size_bytes": sum(artifact_size_bytes(path) for path in paths),
            "elapsed_seconds": time.perf_counter() - started,
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, FastTextLoadError, RankArtifactError, ValueError) as exc:
        print(f"Rank artifact generation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
