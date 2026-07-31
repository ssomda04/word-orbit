"""Pure metric calculations for the word-similarity harness."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Protocol

from contextle_eval.dataset import GROUPS, EvaluationDataset

PAIRWISE_RELATIONS = (
    ("veryClose", "unrelated"),
    ("related", "unrelated"),
    ("veryClose", "related"),
    ("veryClose", "surfaceTrap"),
    ("related", "surfaceTrap"),
)


class VectorProvider(Protocol):
    """Minimal interface used by the evaluator and test doubles."""

    def vector(self, word: str) -> Sequence[float]:
        """Return a vector for ``word``."""
        ...

    def vocabulary_status(self, word: str) -> bool | None:
        """Return direct vocabulary membership when the provider can determine it."""
        ...


@dataclass(frozen=True)
class EvaluationResult:
    """Structured rows and aggregate metrics ready for reporting."""

    summary: dict[str, object]
    detailed_rows: list[dict[str, object]]
    per_answer_rows: list[dict[str, object]]
    top_k_rows: list[dict[str, object]]


def cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    """Return raw cosine similarity, clamped only for floating-point overshoot."""
    if not first or not second:
        raise ValueError("Cosine similarity requires two non-empty vectors.")
    if len(first) != len(second):
        raise ValueError(
            f"Vector dimensions differ: first={len(first)}, second={len(second)}."
        )
    if any(not math.isfinite(float(value)) for value in (*first, *second)):
        raise ValueError("Cosine similarity requires finite vector values.")

    first_norm = math.sqrt(sum(float(value) ** 2 for value in first))
    second_norm = math.sqrt(sum(float(value) ** 2 for value in second))
    if first_norm == 0.0 or second_norm == 0.0:
        raise ValueError("Cosine similarity is undefined for a zero vector.")
    similarity = sum(float(a) * float(b) for a, b in zip(first, second, strict=True))
    similarity /= first_norm * second_norm
    return max(-1.0, min(1.0, similarity))


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "standardDeviation": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": fmean(values),
        "standardDeviation": pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def pairwise_accuracy(higher: Sequence[float], lower: Sequence[float]) -> tuple[int, int, float]:
    """Measure the fraction of strict ``higher > lower`` pair comparisons."""
    total = len(higher) * len(lower)
    correct = sum(left > right for left in higher for right in lower)
    return correct, total, correct / total if total else 0.0


def top_k_metrics(
    ranked: Sequence[tuple[str, str, float]], top_k: int, relevant_count: int
) -> dict[str, float | int]:
    """Calculate diagnostic candidate-pool ranking metrics."""
    selected = ranked[:top_k]
    very_close_count = sum(group == "veryClose" for _, group, _ in selected)
    related_count = sum(group == "related" for _, group, _ in selected)
    surface_trap_count = sum(group == "surfaceTrap" for _, group, _ in selected)
    denominator = min(top_k, len(ranked))
    return {
        "precisionAtK": very_close_count / denominator if denominator else 0.0,
        "recallAtK": very_close_count / relevant_count if relevant_count else 0.0,
        "veryCloseInTopK": very_close_count,
        "relatedInTopK": related_count,
        "surfaceTrapInTopK": surface_trap_count,
    }


def _unique_words(dataset: EvaluationDataset) -> list[str]:
    words: dict[str, None] = {}
    for item in dataset.items:
        words.setdefault(item.answer, None)
        for candidates in item.groups().values():
            for word in candidates:
                words.setdefault(word, None)
    return list(words)


def _load_vectors(
    provider: VectorProvider, words: Sequence[str]
) -> tuple[dict[str, Sequence[float]], list[dict[str, str]], float]:
    vectors: dict[str, Sequence[float]] = {}
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    for word in words:
        try:
            vector = provider.vector(word)
            cosine_similarity(vector, vector)
            vectors[word] = vector
        except (TypeError, ValueError, RuntimeError) as exc:
            failures.append({"word": word, "error": str(exc)})
    elapsed = time.perf_counter() - started
    return vectors, failures, elapsed


def evaluate(
    dataset: EvaluationDataset,
    provider: VectorProvider,
    top_k: int,
    seed: int,
) -> EvaluationResult:
    """Evaluate one provider over the complete dataset."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    evaluation_started = time.perf_counter()
    words = _unique_words(dataset)
    vectors, failures, vector_batch_seconds = _load_vectors(provider, words)

    cache_started = time.perf_counter()
    for word in words:
        vectors.get(word)
    cache_seconds = time.perf_counter() - cache_started

    vocabulary_counts = {
        "totalWords": len(words),
        "directVocabulary": 0,
        "subword": 0,
        "failed": len(failures),
        "unknownStatus": 0,
    }
    for word in words:
        if word not in vectors:
            continue
        status = provider.vocabulary_status(word)
        if status is True:
            vocabulary_counts["directVocabulary"] += 1
        elif status is False:
            vocabulary_counts["subword"] += 1
        else:
            vocabulary_counts["unknownStatus"] += 1

    detailed_rows: list[dict[str, object]] = []
    per_answer_rows: list[dict[str, object]] = []
    top_k_rows: list[dict[str, object]] = []
    aggregate_groups: dict[str, list[float]] = {group: [] for group in GROUPS}
    aggregate_pairwise: dict[str, list[int]] = {
        f"{higher}>{lower}": [0, 0] for higher, lower in PAIRWISE_RELATIONS
    }

    for item in dataset.items:
        group_scores: dict[str, list[float]] = {group: [] for group in GROUPS}
        ranked: list[tuple[str, str, float]] = []
        answer_vector = vectors.get(item.answer)
        if answer_vector is not None:
            for group, candidates in item.groups().items():
                for candidate in candidates:
                    candidate_vector = vectors.get(candidate)
                    if candidate_vector is None:
                        continue
                    score = cosine_similarity(answer_vector, candidate_vector)
                    group_scores[group].append(score)
                    aggregate_groups[group].append(score)
                    ranked.append((candidate, group, score))
                    detailed_rows.append(
                        {
                            "answer": item.answer,
                            "candidate": candidate,
                            "group": group,
                            "similarity": score,
                        }
                    )

        ranked.sort(key=lambda row: (-row[2], row[0]))
        top_metrics = top_k_metrics(ranked, top_k, len(item.very_close))
        for rank, (candidate, group, score) in enumerate(ranked, start=1):
            top_k_rows.append(
                {
                    "answer": item.answer,
                    "rank": rank,
                    "candidate": candidate,
                    "group": group,
                    "similarity": score,
                    "inTopK": rank <= top_k,
                    "veryCloseInTopK": top_metrics["veryCloseInTopK"],
                    "relatedInTopK": top_metrics["relatedInTopK"],
                    "surfaceTrapInTopK": top_metrics["surfaceTrapInTopK"],
                }
            )

        row: dict[str, object] = {
            "answer": item.answer,
            "precisionAtK": top_metrics["precisionAtK"],
            "recallAtK": top_metrics["recallAtK"],
            "veryCloseInTopK": top_metrics["veryCloseInTopK"],
            "relatedInTopK": top_metrics["relatedInTopK"],
            "surfaceTrapInTopK": top_metrics["surfaceTrapInTopK"],
        }
        for group in GROUPS:
            stats = _stats(group_scores[group])
            for metric, value in stats.items():
                row[f"{group}_{metric}"] = value
        for higher, lower in PAIRWISE_RELATIONS:
            correct, total, accuracy = pairwise_accuracy(
                group_scores[higher], group_scores[lower]
            )
            key = f"{higher}>{lower}"
            aggregate_pairwise[key][0] += correct
            aggregate_pairwise[key][1] += total
            row[f"{higher}_gt_{lower}_accuracy"] = accuracy
            row[f"{higher}_gt_{lower}_correct"] = correct
            row[f"{higher}_gt_{lower}_total"] = total
        per_answer_rows.append(row)

    overall_pairwise = {}
    for key, (correct, total) in aggregate_pairwise.items():
        overall_pairwise[key] = {
            "correct": correct,
            "total": total,
            "accuracy": correct / total if total else 0.0,
        }

    valid_word_count = len(vectors)
    summary: dict[str, object] = {
        "dataset": {
            "name": dataset.dataset_name,
            "version": dataset.version,
            "language": dataset.language,
            "answerCount": len(dataset.items),
        },
        "seed": seed,
        "requestedTopK": top_k,
        "candidatePoolScope": (
            "Each answer is ranked only against candidates listed for that answer; "
            "this is not a full game-vocabulary rank."
        ),
        "groupMetrics": {group: _stats(values) for group, values in aggregate_groups.items()},
        "pairwiseAccuracy": overall_pairwise,
        "topK": {
            "meanPrecisionAtK": fmean(
                float(row["precisionAtK"]) for row in per_answer_rows
            ),
            "meanRecallAtK": fmean(float(row["recallAtK"]) for row in per_answer_rows),
            "veryCloseInTopK": sum(
                int(row["veryCloseInTopK"]) for row in per_answer_rows
            ),
            "relatedInTopK": sum(
                int(row["relatedInTopK"]) for row in per_answer_rows
            ),
            "surfaceTrapInTopK": sum(
                int(row["surfaceTrapInTopK"]) for row in per_answer_rows
            ),
        },
        "timingSeconds": {
            "vectorBatch": vector_batch_seconds,
            "meanVectorLookupApprox": (
                vector_batch_seconds / valid_word_count if valid_word_count else None
            ),
            "cachedDictionaryBatch": cache_seconds,
            "meanCachedDictionaryLookup": cache_seconds / len(words) if words else None,
            "evaluationExcludingModelLoad": time.perf_counter() - evaluation_started,
        },
        "vocabularyHandling": vocabulary_counts,
        "vectorFailures": failures,
    }
    return EvaluationResult(summary, detailed_rows, per_answer_rows, top_k_rows)
