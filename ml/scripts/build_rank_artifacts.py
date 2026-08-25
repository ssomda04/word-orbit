#!/usr/bin/env python3
"""Build a production-oriented provisional answer artifact root."""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
    build_artifact,
    build_normalized_vector_matrix,
    select_benchmark_answers,
    write_artifact_root,
)
from contextle_eval.rank_table import normalize_word

DEFAULT_VOCABULARY = ML_ROOT / "data" / "game_words.txt"
DEFAULT_POOL = ML_ROOT / "data" / "answer_pool_provisional.txt"
DEFAULT_CANDIDATES = ML_ROOT / "data" / "answer_candidates_with_frequency_score.csv"
DEFAULT_OUTPUT = ML_ROOT / "data" / "artifacts" / "provisional"
DEFAULT_EMBEDDING_MODEL_NAME = "fasttext-cc-ko-300"
DEFAULT_ANSWERS_LIMIT = 10


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a provisional rank artifact root.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--answer-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument(
        "--artifact-root",
        "--output-dir",
        dest="artifact_root",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--answers-limit", type=int, choices=(10, 50))
    selection.add_argument("--all-answers", action="store_true")
    parser.add_argument("--embedding-model-name", default=DEFAULT_EMBEDDING_MODEL_NAME)
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


def select_answers(
    answer_rows: list[tuple[str, str]],
    *,
    answers_limit: int | None,
    all_answers: bool,
    seed: int,
) -> tuple[str, ...]:
    """Select bounded benchmark answers unless the full pool is explicit."""
    if answers_limit is not None and all_answers:
        raise RankArtifactError("--answers-limit and --all-answers cannot be used together.")
    if all_answers:
        answers = tuple(word for word, _ in answer_rows)
    else:
        limit = answers_limit or DEFAULT_ANSWERS_LIMIT
        answers = select_benchmark_answers(answer_rows, count=limit, seed=seed)
    normalized = tuple(normalize_word(answer) for answer in answers)
    if not normalized or any(not answer for answer in normalized):
        raise RankArtifactError("Answer selection must contain non-empty words.")
    if len(set(normalized)) != len(normalized):
        raise RankArtifactError("Answer selection contains duplicate normalized answers.")
    return normalized


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.artifact_root.exists() and any(args.artifact_root.iterdir()):
        print(f"Artifact root must be new or empty: {args.artifact_root}", file=sys.stderr)
        return 2
    try:
        vocabulary = VocabularyIndex.from_file(args.vocabulary)
        answers = select_answers(
            load_answer_rows(args.answer_pool, args.candidates),
            answers_limit=args.answers_limit,
            all_answers=args.all_answers,
            seed=args.seed,
        )
        provider = FastTextVectorProvider.load(args.model_path)
        matrix = build_normalized_vector_matrix(vocabulary, provider)
        artifacts = (
            build_artifact(
                answer,
                vocabulary,
                matrix,
                embedding_model=args.model_path.name,
                similarity_dtype="float32",
                ranktable_compatibility_provider=provider,
            )
            for answer in answers
        )
        manifest = write_artifact_root(
            args.artifact_root,
            vocabulary,
            artifacts,
            embedding_model_name=args.embedding_model_name,
            embedding_model_source=args.model_path.name,
        )
    except (OSError, FastTextLoadError, RankArtifactError, ValueError) as exc:
        print(f"Rank artifact generation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
