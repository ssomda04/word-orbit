#!/usr/bin/env python3
"""Benchmark 10 then 50 FastText rank artifacts without building the full pool."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

ML_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ML_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_rank_artifacts import load_answer_rows
from contextle_eval.fasttext_provider import FastTextLoadError, FastTextVectorProvider
from contextle_eval.rank_artifact import (
    DEFAULT_SEED,
    RankArtifactError,
    VocabularyIndex,
    artifact_from_calculated,
    artifact_size_bytes,
    build_normalized_vector_matrix,
    load_artifact_npy,
    load_artifact_npz,
    process_rss_bytes,
    ranks_from_similarities,
    refine_ranktable_near_ties,
    save_artifact_npy,
    save_artifact_npz,
    select_benchmark_answers,
    similarities_for_answer,
    timed_lookup,
)

DEFAULT_VOCABULARY = ML_ROOT / "data" / "game_words.txt"
DEFAULT_POOL = ML_ROOT / "data" / "answer_pool_provisional.txt"
DEFAULT_CANDIDATES = ML_ROOT / "data" / "answer_candidates_with_frequency_score.csv"
DEFAULT_OUTPUT = ML_ROOT / "data" / "artifacts" / "provisional"
CONFIGS = (("npy", "float32"), ("npy", "float16"), ("npz", "float32"), ("npz", "float16"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark exactly 10 then 50 answer artifacts.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--vocabulary", type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument("--answer-pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _load_timing(path: Path, vocabulary: VocabularyIndex, format_name: str) -> tuple[object, float]:
    started = time.perf_counter_ns()
    artifact = (
        load_artifact_npy(path, vocabulary)
        if format_name == "npy"
        else load_artifact_npz(path, vocabulary)
    )
    return artifact, (time.perf_counter_ns() - started) / 1e6


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        print(f"Output directory must be new or empty: {args.output_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        map_started = time.perf_counter()
        vocabulary = VocabularyIndex.from_file(args.vocabulary)
        vocabulary_map_seconds = time.perf_counter() - map_started
        rows = load_answer_rows(args.answer_pool, args.candidates)
        answers_10 = select_benchmark_answers(rows, count=10, seed=args.seed)
        answers_50 = select_benchmark_answers(rows, count=50, seed=args.seed)
        if answers_50[:10] != answers_10:
            raise RankArtifactError("The 10-answer sample must be a subset prefix of 50 answers.")

        rss_before_model = process_rss_bytes()
        model_started = time.perf_counter()
        provider = FastTextVectorProvider.load(args.model_path)
        model_load_seconds = time.perf_counter() - model_started
        rss_after_model = process_rss_bytes()
        matrix_started = time.perf_counter()
        matrix = build_normalized_vector_matrix(vocabulary, provider)
        matrix_seconds = time.perf_counter() - matrix_started

        config_results: dict[str, object] = {}
        reference_ranks: dict[str, np.ndarray] = {}
        paths_by_config: dict[str, list[Path]] = {}
        serialization_seconds = {f"{fmt}_{dtype}": 0.0 for fmt, dtype in CONFIGS}
        calculation_seconds = 0.0
        calculation_at_10 = 0.0
        ten_checkpoint: dict[str, float] = {}
        for answer_number, answer in enumerate(answers_50, start=1):
            answer_index = vocabulary.index_of(answer)
            if answer_index is None:
                raise RankArtifactError(f"Selected answer missing from vocabulary: {answer}")
            calculation_started = time.perf_counter()
            full = similarities_for_answer(answer_index, matrix)
            full = refine_ranktable_near_ties(full, vocabulary, answer_index, provider)
            ranks = ranks_from_similarities(full, vocabulary, answer_index)
            calculation_seconds += time.perf_counter() - calculation_started
            reference_ranks[answer] = ranks
            random_indices = np.asarray(
                random.Random(args.seed + answer_number).sample(range(len(vocabulary.words)), 100)
            )
            for format_name, dtype in CONFIGS:
                key = f"{format_name}_{dtype}"
                directory = args.output_dir / key
                config_started = time.perf_counter()
                artifact = artifact_from_calculated(
                    answer,
                    vocabulary,
                    full,
                    ranks,
                    embedding_model=args.model_path.name,
                    similarity_dtype=dtype,
                )
                if not np.array_equal(artifact.ranks, ranks):
                    raise RankArtifactError(f"Rank mismatch for {answer} in {key}")
                if any(
                    int(artifact.ranks[index]) != int(ranks[index])
                    or not math.isclose(
                        float(artifact.similarities[index]),
                        float(full[index]),
                        abs_tol=5e-4 if dtype == "float16" else 1e-6,
                    )
                    for index in random_indices
                ):
                    raise RankArtifactError(f"Random lookup mismatch for {answer} in {key}")
                save = save_artifact_npy if format_name == "npy" else save_artifact_npz
                paths_by_config.setdefault(key, []).append(save(directory, artifact))
                serialization_seconds[key] += time.perf_counter() - config_started
            if answer_number == 10:
                ten_checkpoint = dict(serialization_seconds)
                calculation_at_10 = calculation_seconds

        representative = answers_10[0]
        rep_index = vocabulary.index_of(representative)
        assert rep_index is not None
        rep_full = similarities_for_answer(rep_index, matrix)
        top_30_indices = np.argsort(reference_ranks[representative])[:30]
        top_30 = [
            {
                "word": vocabulary.words[int(index)],
                "rank": int(reference_ranks[representative][index]),
                "similarity": float(rep_full[index]),
            }
            for index in top_30_indices
        ]

        rng = np.random.default_rng(args.seed)
        runtime: dict[str, object] = {}
        quantization: dict[str, object] = {}
        for format_name, dtype in CONFIGS:
            key = f"{format_name}_{dtype}"
            paths = paths_by_config[key]
            artifact, cold_load_ms = _load_timing(paths[0], vocabulary, format_name)
            indices_1k = rng.integers(0, len(vocabulary.words), size=1_000)
            indices_10k = rng.integers(0, len(vocabulary.words), size=10_000)
            runtime[key] = {
                "cold_load_ms": cold_load_ms,
                "lookup_1": timed_lookup(artifact, np.asarray([indices_1k[0]]))["one_lookup_ns"],
                "lookup_1000_ms": timed_lookup(artifact, indices_1k)["batch_lookup_ns"] / 1e6,
                "lookup_10000_ms": timed_lookup(artifact, indices_10k)["batch_lookup_ns"] / 1e6,
                "array_bytes": int(artifact.similarities.nbytes + artifact.ranks.nbytes),
            }
            total_size = sum(artifact_size_bytes(path) for path in paths)
            config_results[key] = {
                "ten_size_bytes": sum(artifact_size_bytes(path) for path in paths[:10]),
                "fifty_size_bytes": total_size,
                "mean_answer_size_bytes": total_size / 50,
                "ten_calculation_seconds": calculation_at_10,
                "fifty_calculation_seconds": calculation_seconds,
                "ten_serialization_seconds": ten_checkpoint[key],
                "fifty_serialization_seconds": serialization_seconds[key],
                "ten_generation_seconds": calculation_at_10 + ten_checkpoint[key],
                "fifty_generation_seconds": calculation_seconds + serialization_seconds[key],
            }
            if dtype == "float16":
                stored = np.asarray(artifact.similarities, dtype=np.float64)
                error = np.abs(stored - rep_full)
                reranked = ranks_from_similarities(stored, vocabulary, rep_index)
                quantization[key] = {
                    "max_abs_similarity_error": float(error.max()),
                    "mean_abs_similarity_error": float(error.mean()),
                    "rank_changed_count": int(np.count_nonzero(reranked != reference_ranks[representative])),
                    "top_10_changed_count": len(
                        set(np.argsort(reranked)[:10]) ^ set(np.argsort(reference_ranks[representative])[:10])
                    ),
                    "top_100_changed_count": len(
                        set(np.argsort(reranked)[:100]) ^ set(np.argsort(reference_ranks[representative])[:100])
                    ),
                }

        report = {
            "seed": args.seed,
            "vocabulary_size": len(vocabulary.words),
            "vocabulary_sha256": vocabulary.sha256,
            "answers_10": answers_10,
            "answers_50": answers_50,
            "vocabulary_map_seconds": vocabulary_map_seconds,
            "model_load_seconds": model_load_seconds,
            "vector_matrix_seconds": matrix_seconds,
            "rss_before_model_bytes": rss_before_model,
            "rss_after_model_and_matrix_bytes": process_rss_bytes(),
            "rss_model_load_delta_bytes": (
                rss_after_model - rss_before_model
                if rss_before_model is not None and rss_after_model is not None
                else None
            ),
            "normalized_vector_matrix_bytes": int(matrix.nbytes),
            "generation": config_results,
            "runtime": runtime,
            "float16_quantization_representative": quantization,
            "representative_answer": representative,
            "representative_top_30": top_30,
            "full_pool_estimate_multiplier": 1949 / 50,
            "output_directory_size_bytes": _directory_size(args.output_dir),
        }
        (args.output_dir / "benchmark_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, FastTextLoadError, RankArtifactError, ValueError) as exc:
        print(f"Rank artifact benchmark failed: {exc}", file=sys.stderr)
        return 2
    finally:
        # No model or temporary output cleanup is destructive; generated artifacts remain ignored.
        pass
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
