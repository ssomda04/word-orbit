"""CLI-selection tests for production-oriented rank artifact generation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from contextle_eval.rank_artifact import RankArtifactError

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_rank_artifacts.py"
SPEC = importlib.util.spec_from_file_location("build_rank_artifacts", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
build_rank_artifacts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_rank_artifacts)


def test_cli_defaults_to_bounded_mode() -> None:
    args = build_rank_artifacts.parse_args(["--model-path", "model.bin"])

    assert args.answers_limit is None
    assert args.all_answers is False


@pytest.mark.parametrize("limit", [10, 50])
def test_cli_accepts_supported_answer_limits(limit: int) -> None:
    args = build_rank_artifacts.parse_args(
        ["--model-path", "model.bin", "--answers-limit", str(limit)]
    )

    assert args.answers_limit == limit
    assert args.all_answers is False


def test_cli_accepts_explicit_all_answers() -> None:
    args = build_rank_artifacts.parse_args(["--model-path", "model.bin", "--all-answers"])

    assert args.all_answers is True
    assert args.answers_limit is None


def test_cli_rejects_limit_with_all_answers() -> None:
    with pytest.raises(SystemExit):
        build_rank_artifacts.parse_args(
            [
                "--model-path",
                "model.bin",
                "--answers-limit",
                "10",
                "--all-answers",
            ]
        )


def test_all_answers_preserves_pool_order_and_rejects_duplicates() -> None:
    rows = [(" Ａ ", "noun"), ("가", "noun")]
    assert build_rank_artifacts.select_answers(
        rows, answers_limit=None, all_answers=True, seed=1
    ) == ("A", "가")

    with pytest.raises(RankArtifactError, match="duplicate"):
        build_rank_artifacts.select_answers(
            [("Ａ", "noun"), ("A", "noun")],
            answers_limit=None,
            all_answers=True,
            seed=1,
        )
