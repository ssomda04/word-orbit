"""Tests for model-independent metric and reporting logic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from contextle_eval.dataset import parse_dataset
from contextle_eval.metrics import (
    cosine_similarity,
    evaluate,
    pairwise_accuracy,
    top_k_metrics,
)
from contextle_eval.reporting import ReportWriteError, write_reports


class FakeVectorProvider:
    """Small deterministic provider that never loads a real model."""

    def __init__(self) -> None:
        self.vectors = {
            "정답": [1.0, 0.0],
            "가까움1": [1.0, 0.0],
            "가까움2": [0.9, 0.1],
            "관련": [0.5, 0.8660254],
            "무관": [-1.0, 0.0],
            "표면함정": [0.0, 1.0],
        }

    def vector(self, word: str) -> list[float]:
        return list(self.vectors[word])

    def vocabulary_status(self, word: str) -> bool | None:
        return word != "표면함정"


def _dataset():
    return parse_dataset(
        {
            "datasetName": "metrics-test",
            "version": "1.0.0",
            "language": "ko",
            "description": "metrics test",
            "updatedAt": "2026-07-30",
            "groupDefinitions": {
                "veryClose": "close",
                "related": "related",
                "unrelated": "unrelated",
                "surfaceTrap": "surface trap",
            },
            "items": [
                {
                    "answer": "정답",
                    "veryClose": ["가까움1", "가까움2"],
                    "related": ["관련"],
                    "unrelated": ["무관"],
                    "surfaceTrap": ["표면함정"],
                }
            ],
        }
    )


def test_cosine_similarity_stays_in_raw_range() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    assert -1.0 <= cosine_similarity([0.3, 0.7], [0.4, -0.2]) <= 1.0


def test_pairwise_accuracy_uses_all_strict_pairs() -> None:
    correct, total, accuracy = pairwise_accuracy([0.9, 0.8], [0.7, 0.95])

    assert (correct, total) == (2, 4)
    assert accuracy == pytest.approx(0.5)


def test_top_k_metrics() -> None:
    ranked = [
        ("가까움1", "veryClose", 1.0),
        ("관련", "related", 0.9),
        ("표면함정", "surfaceTrap", 0.85),
        ("가까움2", "veryClose", 0.8),
    ]

    metrics = top_k_metrics(ranked, top_k=2, relevant_count=2)

    assert metrics["precisionAtK"] == pytest.approx(0.5)
    assert metrics["recallAtK"] == pytest.approx(0.5)
    assert metrics["veryCloseInTopK"] == 1
    assert metrics["relatedInTopK"] == 1
    assert metrics["surfaceTrapInTopK"] == 0


def test_evaluation_calculates_all_new_pairwise_relations() -> None:
    result = evaluate(_dataset(), FakeVectorProvider(), top_k=3, seed=42)

    assert result.summary["requestedTopK"] == 3
    assert set(result.summary["pairwiseAccuracy"]) == {
        "veryClose>unrelated",
        "related>unrelated",
        "veryClose>related",
        "veryClose>surfaceTrap",
        "related>surfaceTrap",
    }
    assert result.summary["topK"]["veryCloseInTopK"] == 2
    assert result.summary["topK"]["relatedInTopK"] == 1
    assert result.summary["topK"]["surfaceTrapInTopK"] == 0
    assert result.top_k_rows[0]["veryCloseInTopK"] == 2
    assert result.top_k_rows[0]["relatedInTopK"] == 1
    assert result.top_k_rows[0]["surfaceTrapInTopK"] == 0


def test_evaluation_is_reproducible_for_same_inputs() -> None:
    first = evaluate(_dataset(), FakeVectorProvider(), top_k=2, seed=42)
    second = evaluate(_dataset(), FakeVectorProvider(), top_k=2, seed=42)

    assert first.detailed_rows == second.detailed_rows
    assert first.per_answer_rows == second.per_answer_rows
    assert first.top_k_rows == second.top_k_rows
    assert first.summary["groupMetrics"] == second.summary["groupMetrics"]
    assert first.summary["pairwiseAccuracy"] == second.summary["pairwiseAccuracy"]


def test_writes_complete_report_bundle(tmp_path: Path) -> None:
    result = evaluate(_dataset(), FakeVectorProvider(), top_k=2, seed=42)
    config = {
        "modelFileName": "fake.bin",
        "seed": 42,
        "datasetVersion": "1.0.0",
    }

    write_reports(tmp_path / "reports", result, config)

    expected = {
        "summary.json",
        "detailed_results.csv",
        "per_answer_metrics.csv",
        "top_k_results.csv",
        "run_config.json",
    }
    assert {path.name for path in (tmp_path / "reports").iterdir()} == expected
    saved_config = json.loads((tmp_path / "reports" / "run_config.json").read_text("utf-8"))
    assert saved_config["modelFileName"] == "fake.bin"
    assert str(tmp_path) not in json.dumps(saved_config)


def test_refuses_to_overwrite_existing_results(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports-v0.1"
    output_dir.mkdir()
    previous = output_dir / "summary.json"
    previous.write_text('{"version": "0.1"}', encoding="utf-8")
    result = evaluate(_dataset(), FakeVectorProvider(), top_k=2, seed=42)

    with pytest.raises(ReportWriteError, match="not empty"):
        write_reports(output_dir, result, {"datasetVersion": "0.2.0"})

    assert previous.read_text(encoding="utf-8") == '{"version": "0.1"}'
