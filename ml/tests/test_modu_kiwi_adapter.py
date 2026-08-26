"""Tests for adapting Kiwi tokens to the shared Modu normalization policy."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from contextle_eval.modu_kiwi_adapter import (
    KiwiAdapterError,
    adapt_kiwi_tokens,
    lexical_frequency_records,
)

GAME_WORDS = frozenset(
    {
        "공부",
        "가능",
        "사람",
        "먹다",
        "듣다",
        "걷다",
        "좋다",
        "춥다",
        "있다",
        "없다",
        "같다",
        "하다",
        "되다",
    }
)
ANSWER_WORDS = frozenset({"먹다", "듣다", "걷다", "좋다"})


@dataclass(frozen=True)
class FakeToken:
    form: str
    tag: str
    start: int
    len: int
    word_position: int
    sent_position: int = 0


def test_adapter_reuses_contract_and_preserves_provenance() -> None:
    text = "사람이 먹었다"
    tokens = [
        FakeToken("사람", "NNG", 0, 2, 0),
        FakeToken("이", "JKS", 2, 1, 0),
        FakeToken("먹", "VV", 4, 1, 1),
        FakeToken("었", "EP", 5, 1, 1),
        FakeToken("다", "EF", 6, 1, 1),
    ]

    records = adapt_kiwi_tokens("fixture-1", text, tokens, GAME_WORDS, ANSWER_WORDS)

    assert records[0].canonical_form == "사람"
    assert records[0].status == "matched"
    assert records[2].canonical_form == "먹다"
    assert records[2].normalization_reason == "predicate_plus_da_vocabulary_match"
    assert records[2].source_text_id == "fixture-1"
    assert records[2].token_index == 2
    assert records[2].eojeol_index == 1
    assert records[2].eojeol_surface == "먹었다"


def test_irregular_tag_maps_to_mp_base_pos_deterministically() -> None:
    text = "들었다 걸었다"
    tokens = [
        FakeToken("듣", "VV-I", 0, 1, 0),
        FakeToken("었", "EP", 1, 1, 0),
        FakeToken("다", "EF", 2, 1, 0),
        FakeToken("걷", "VV-I", 4, 1, 1),
        FakeToken("었", "EP", 5, 1, 1),
        FakeToken("다", "EF", 6, 1, 1),
    ]

    first = adapt_kiwi_tokens("irregular", text, tokens, GAME_WORDS, ANSWER_WORDS)
    second = adapt_kiwi_tokens("irregular", text, tokens, GAME_WORDS, ANSWER_WORDS)

    assert first == second
    assert [(first[index].source_pos, first[index].canonical_form) for index in (0, 3)] == [
        ("VV", "듣다"),
        ("VV", "걷다"),
    ]


def test_eojeol_grouping_includes_sentence_position() -> None:
    text = "먹다. 듣다."
    tokens = [
        FakeToken("먹", "VV", 0, 1, 0, 0),
        FakeToken("다", "EF", 1, 1, 0, 0),
        FakeToken(".", "SF", 2, 1, 0, 0),
        FakeToken("듣", "VV-I", 4, 1, 0, 1),
        FakeToken("다", "EF", 5, 1, 0, 1),
        FakeToken(".", "SF", 6, 1, 0, 1),
    ]

    records = adapt_kiwi_tokens("sentences", text, tokens, GAME_WORDS, ANSWER_WORDS)

    assert records[0].eojeol_surface == "먹다."
    assert records[3].eojeol_surface == "듣다."
    assert records[0].sentence_index == 0
    assert records[3].sentence_index == 1


def test_xsv_xsa_context_is_audited_without_synthetic_frequency() -> None:
    text = "공부하다 가능하다"
    tokens = [
        FakeToken("공부", "NNG", 0, 2, 0),
        FakeToken("하", "XSV", 2, 1, 0),
        FakeToken("다", "EF", 3, 1, 0),
        FakeToken("가능", "NNG", 5, 2, 1),
        FakeToken("하", "XSA", 7, 1, 1),
        FakeToken("다", "EF", 8, 1, 1),
    ]

    records = adapt_kiwi_tokens("derivation", text, tokens, GAME_WORDS, ANSWER_WORDS)

    assert records[1].status == records[4].status == "unmatched"
    assert records[1].derivational_candidate == "공부하다"
    assert records[4].derivational_candidate == "가능하다"
    assert records[0].eojeol_has_derivational_suffix is True
    assert [record.canonical_form for record in lexical_frequency_records(records)] == [
        "공부",
        "가능",
    ]
    with pytest.raises(KiwiAdapterError, match="more than once"):
        lexical_frequency_records((*records, records[0]))


def test_actual_kiwi_content_auxiliary_and_derivational_output() -> None:
    kiwipiepy = pytest.importorskip("kiwipiepy")
    kiwi = kiwipiepy.Kiwi()
    text = "밥을 먹었고 날씨가 좋았고 추웠다. 공부하지 않다. 성공이 가능하다."

    records = adapt_kiwi_tokens(
        "actual-kiwi", text, kiwi.tokenize(text), GAME_WORDS, ANSWER_WORDS
    )
    observed = {(record.source_morpheme, record.source_pos) for record in records}

    assert {("먹", "VV"), ("좋", "VA"), ("춥", "VA"), ("않", "VX")} <= observed
    assert ("춥", "VA-I") in {
        (record.source_morpheme, record.kiwi_tag) for record in records
    }
    assert ("하", "XSV") in observed
    assert ("하", "XSA") in observed
    assert all(record.source_text_id == "actual-kiwi" for record in records)
