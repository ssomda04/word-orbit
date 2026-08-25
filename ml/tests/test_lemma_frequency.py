"""Tests for conservative Kiwi lemma-frequency aggregation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pytest
from contextle_eval.lemma_frequency import (
    LemmaFrequencyError,
    analyze_frequencies,
    analyze_surface,
    classify_tokens,
    join_candidates,
    load_surface_frequency,
)


@dataclass(frozen=True)
class FakeToken:
    form: str
    tag: str


class FakeKiwi:
    def __init__(
        self, analyses: dict[str, list[tuple[list[FakeToken], float]]]
    ) -> None:
        self.analyses = analyses

    def analyze(
        self, text: str, top_n: int = 1
    ) -> list[tuple[list[FakeToken], float]]:
        return self.analyses.get(text, [])[:top_n]


def tokens(*items: tuple[str, str]) -> list[FakeToken]:
    return [FakeToken(form, tag) for form, tag in items]


def test_classifies_predicate_inflection_and_derived_predicate() -> None:
    eaten = classify_tokens(tokens(("먹", "VV"), ("었", "EP"), ("어", "EF")), "먹었어")
    happy = classify_tokens(
        tokens(("행복", "NNG"), ("하", "XSA"), ("ᆫ", "ETM")), "행복한"
    )

    assert (eaten.status, eaten.lemma, eaten.pos) == ("exact", "먹다", "동사")
    assert (happy.status, happy.lemma, happy.pos) == ("exact", "행복하다", "형용사")


def test_plain_noun_is_exact_but_particle_sequence_is_multi_morpheme() -> None:
    plain = classify_tokens(tokens(("사람", "NNG")), "사람")
    inflected = classify_tokens(tokens(("사람", "NNG"), ("이", "JKS")), "사람이")

    assert (plain.status, plain.lemma, plain.pos) == ("exact", "사람", "명사")
    assert inflected.status == "multi_morpheme"


def test_close_competing_assignments_are_ambiguous_and_receive_no_lemma() -> None:
    kiwi = FakeKiwi(
        {
            "작은": [
                (tokens(("작", "VA"), ("은", "ETM")), -10.0),
                (tokens(("작은", "NNG")), -11.0),
            ]
        }
    )

    result = analyze_surface(kiwi, "작은", 20, top_n=2, ambiguity_margin=3.0)

    assert result.analysis_status == "ambiguous"
    assert result.lemma == ""
    assert result.pos == ""


def test_exact_counts_are_assigned_once_and_non_exact_counts_are_preserved() -> None:
    kiwi = FakeKiwi(
        {
            "먹어": [(tokens(("먹", "VV"), ("어", "EF")), -1.0)],
            "먹고": [(tokens(("먹", "VV"), ("고", "EC")), -1.0)],
            "사람이": [(tokens(("사람", "NNG"), ("이", "JKS")), -1.0)],
            "???": [],
        }
    )

    analyses, aggregates = analyze_frequencies(
        kiwi,
        [("먹어", 7), ("먹고", 11), ("사람이", 13), ("???", 17)],
    )

    assert [(row.lemma, row.pos, row.count, row.source_surface_count) for row in aggregates] == [
        ("먹다", "동사", 18, 2)
    ]
    assert sum(row.count for row in analyses) == 48
    assert {row.surface: row.analysis_status for row in analyses} == {
        "먹어": "exact",
        "먹고": "exact",
        "사람이": "multi_morpheme",
        "???": "unanalyzed",
    }


def test_join_requires_compatible_candidate_pos_and_creates_no_rank(tmp_path: Path) -> None:
    kiwi = FakeKiwi(
        {
            "먹어": [(tokens(("먹", "VV"), ("어", "EF")), -1.0)],
            "먹": [(tokens(("먹", "NNG")), -1.0)],
        }
    )
    _, aggregates = analyze_frequencies(kiwi, [("먹어", 7), ("먹", 5)])
    candidates = tmp_path / "candidates.csv"
    output = tmp_path / "joined.csv"
    with candidates.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("word", "pos", "review_required", "review_reason"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(
            [
                {"word": "먹다", "pos": "동사", "review_required": "false", "review_reason": ""},
                {"word": "먹다", "pos": "명사", "review_required": "false", "review_reason": ""},
                {"word": "먹", "pos": "명사", "review_required": "false", "review_reason": ""},
            ]
        )

    rows = join_candidates(candidates, aggregates, output)

    assert rows[0]["lemma_frequency"] == "7"
    assert rows[0]["lemma_frequency_pos"] == "동사"
    assert rows[1]["lemma_frequency_found"] == "false"
    assert rows[2]["lemma_frequency"] == "5"
    assert all("rank" not in key for key in rows[0])


def test_surface_loader_validates_integer_and_duplicate_input(tmp_path: Path) -> None:
    valid = tmp_path / "valid.txt"
    valid.write_text("먹어 7\n사람 3\n", encoding="utf-8")
    assert load_surface_frequency(valid) == [("먹어", 7), ("사람", 3)]

    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text("Ａ 1\nA 2\n", encoding="utf-8")
    with pytest.raises(LemmaFrequencyError, match="duplicate normalized surface"):
        load_surface_frequency(duplicate)

    invalid = tmp_path / "invalid.txt"
    invalid.write_text("먹어 NaN\n", encoding="utf-8")
    with pytest.raises(LemmaFrequencyError, match="non-integer count"):
        load_surface_frequency(invalid)
