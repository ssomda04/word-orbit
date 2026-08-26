"""Tests for conservative Modu morpheme/POS base-form analysis."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from contextle_eval.modu_baseform_analysis import (
    ModuBaseformAnalysisError,
    build_analysis_report,
    conservative_baseform,
    load_answer_candidates,
    load_frequency_entries,
    load_vocabulary,
    write_report,
)


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    frequency = tmp_path / "frequency.csv"
    frequency.write_text(
        "source_subtype,morpheme,pos,count\n"
        "NXMP,하,VV,10\n"
        "SXMP,하,VX,5\n"
        "NXMP,있,VA,8\n"
        "SXMP,있,VX,7\n"
        "NXMP,사과,NNG,6\n"
        "NXMP,사과,NNP,2\n"
        "SXMP,빨리,MAG,4\n"
        "NXMP,하,XSV,3\n",
        encoding="utf-8",
    )
    vocabulary = tmp_path / "game_words.txt"
    vocabulary.write_text("하\n하다\n있다\n사과\n빨리\n", encoding="utf-8")
    candidates = tmp_path / "answer_candidates.csv"
    with candidates.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["word", "pos"])
        writer.writeheader()
        writer.writerows(
            [
                {"word": "하다", "pos": "동사"},
                {"word": "있다", "pos": "형용사"},
                {"word": "사과", "pos": "명사"},
            ]
        )
    return frequency, vocabulary, candidates


def test_conservative_contract_is_pos_gated_and_vocabulary_backed() -> None:
    vocabulary = frozenset({"사과", "하다", "있다"})

    assert conservative_baseform("사과", "NNG", vocabulary) == ("사과", "matched")
    assert conservative_baseform("하", "VV", vocabulary) == ("하다", "matched")
    assert conservative_baseform("있", "VA", vocabulary) == ("있다", "matched")
    assert conservative_baseform("먹", "VV", vocabulary) == ("", "review")
    assert conservative_baseform("하", "VX", vocabulary) == ("", "unmatched")
    assert conservative_baseform("하", "XSV", vocabulary) == ("", "unmatched")
    assert conservative_baseform("사과", "NNP", vocabulary) == ("", "review")
    assert conservative_baseform("어머", "IC", vocabulary) == ("", "review")


def test_report_measures_strategy_subtype_and_cross_pos_coverage(tmp_path: Path) -> None:
    frequency, vocabulary_path, candidates_path = _write_inputs(tmp_path)
    entries = load_frequency_entries(frequency)
    vocabulary = load_vocabulary(vocabulary_path)
    candidates = load_answer_candidates(candidates_path)

    report = build_analysis_report(entries, vocabulary, candidates)

    vv = report["pos_analysis"]["VV"]["coverage"]
    assert vv["raw_in_game_vocabulary"]["unique_rate"] == 1.0
    assert vv["plus_da_in_game_vocabulary"]["unique_rate"] == 1.0
    assert vv["plus_da_in_answer_candidates"]["token_hits"] == 10
    assert report["subtype_coverage"]["SXMP"]["VX"]["token_count"] == 12
    assert report["cross_pos"]["form_count"] == 3
    assert report["collision_risks"]["raw_and_plus_da_both_in_game_vocabulary_count"] == 3
    assert report["scope"]["no_cutoff_or_pool_selection"] is True


def test_rejects_duplicate_frequency_key(tmp_path: Path) -> None:
    path = tmp_path / "frequency.csv"
    path.write_text(
        "source_subtype,morpheme,pos,count\nNXMP,하,VV,1\nNXMP,하,VV,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ModuBaseformAnalysisError, match="duplicate"):
        load_frequency_entries(path)


def test_report_serialization_is_deterministic(tmp_path: Path) -> None:
    frequency, vocabulary_path, candidates_path = _write_inputs(tmp_path)
    report = build_analysis_report(
        load_frequency_entries(frequency),
        load_vocabulary(vocabulary_path),
        load_answer_candidates(candidates_path),
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_report(first, report)
    write_report(second, report)

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8")) == report
