"""Tests for reusable full-vocabulary FastText rank tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from contextle_eval.fasttext_provider import FastTextVectorProvider
from contextle_eval.rank_table import (
    NonFiniteEmbeddingError,
    RankTableError,
    VocabularyLoadError,
    build_rank_table,
    build_rank_table_from_file,
    load_vocabulary,
    normalize_vocabulary,
)


class FakeVectorProvider:
    """Return configured vectors and record every model lookup."""

    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls: list[str] = []

    def vector(self, word: str) -> Sequence[float]:
        self.calls.append(word)
        return self.vectors[word]


def test_ranks_by_descending_similarity() -> None:
    provider = FakeVectorProvider(
        {
            "정답": [1.0, 0.0],
            "가까움": [0.8, 0.6],
            "직교": [0.0, 1.0],
            "반대": [-1.0, 0.0],
        }
    )

    table = build_rank_table("정답", ["반대", "직교", "가까움", "정답"], provider)

    assert [entry.word for entry in table.ranked_entries] == [
        "정답",
        "가까움",
        "직교",
        "반대",
    ]
    assert table.rank_of("가까움") == 2
    assert table.similarity_of("가까움") == pytest.approx(0.8)


def test_similarity_ties_use_lexical_order_but_answer_stays_first() -> None:
    provider = FakeVectorProvider(
        {
            "정답": [1.0, 0.0],
            "가": [1.0, 0.0],
            "나": [1.0, 0.0],
        }
    )

    table = build_rank_table("정답", ["나", "가"], provider)

    assert [entry.word for entry in table.ranked_entries] == ["정답", "가", "나"]
    assert table.rank_of("정답") == 1
    assert all(entry.similarity == pytest.approx(1.0) for entry in table.ranked_entries)


def test_missing_answer_is_automatically_added() -> None:
    provider = FakeVectorProvider({"정답": [1.0, 0.0], "후보": [0.0, 1.0]})

    table = build_rank_table(" 정답 ", ["후보"], provider)

    assert table.answer == "정답"
    assert table.rank_of("정답") == 1
    assert len(table) == 2


def test_normalization_deduplicates_and_removes_blanks() -> None:
    assert normalize_vocabulary(["  Ａ  ", "A", "", " \t ", "Ｂ", " B "]) == (
        "A",
        "B",
    )


def test_vocabulary_file_is_utf8_bom_safe_and_buildable(tmp_path: Path) -> None:
    vocabulary_path = tmp_path / "game_words.txt"
    vocabulary_path.write_text("\ufeff  Ａ  \nA\n\nＢ\n", encoding="utf-8")
    provider = FakeVectorProvider({"정답": [1.0, 0.0], "A": [0.5, 0.5], "B": [0.0, 1.0]})

    table = build_rank_table_from_file("정답", vocabulary_path, provider)

    assert load_vocabulary(vocabulary_path) == ("A", "B")
    assert set(table.entries) == {"정답", "A", "B"}
    assert table.rank_of("  Ａ ") == table.rank_of("A")


def test_answer_zero_vector_is_rank_one_and_all_similarities_are_zero() -> None:
    provider = FakeVectorProvider({"정답": [0.0, 0.0], "가": [1.0, 0.0], "나": [0.0, 0.0]})

    table = build_rank_table("정답", ["나", "가"], provider)

    assert [entry.word for entry in table.ranked_entries] == ["정답", "가", "나"]
    assert table.rank_of("정답") == 1
    assert all(entry.similarity == 0.0 for entry in table.ranked_entries)


def test_candidate_zero_vector_similarity_is_zero() -> None:
    provider = FakeVectorProvider({"정답": [1.0, 0.0], "영벡터": [0.0, 0.0], "반대": [-1.0, 0.0]})

    table = build_rank_table("정답", ["반대", "영벡터"], provider)

    assert table.similarity_of("영벡터") == 0.0
    assert table.rank_of("영벡터") == 2
    assert table.rank_of("반대") == 3


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_vector_has_clear_error(non_finite: float) -> None:
    provider = FakeVectorProvider({"정답": [1.0, 0.0], "오류": [non_finite, 0.0]})

    with pytest.raises(NonFiniteEmbeddingError, match="NaN or infinity"):
        build_rank_table("정답", ["오류"], provider)


def test_non_finite_norm_has_clear_error() -> None:
    provider = FakeVectorProvider({"정답": [1.7e308, 1.7e308], "후보": [1.0, 0.0]})

    with pytest.raises(NonFiniteEmbeddingError, match="norm"):
        build_rank_table("정답", ["후보"], provider)


def test_dimension_mismatch_has_clear_error() -> None:
    provider = FakeVectorProvider({"정답": [1.0, 0.0], "오류": [1.0]})

    with pytest.raises(RankTableError, match="dimensions differ"):
        build_rank_table("정답", ["오류"], provider)


def test_result_is_deterministic_across_vocabulary_order() -> None:
    vectors = {
        "정답": [1.0, 0.0],
        "가": [0.5, 0.5],
        "나": [0.5, 0.5],
        "다": [0.0, 1.0],
    }

    first = build_rank_table("정답", ["다", "나", "가", "나"], FakeVectorProvider(vectors))
    second = build_rank_table("정답", ["가", "나", "다"], FakeVectorProvider(vectors))

    assert first == second
    assert first.ranked_entries == second.ranked_entries


def test_cached_table_reuses_results_without_model_calls() -> None:
    provider = FakeVectorProvider({"정답": [1.0, 0.0], "가": [0.5, 0.5], "나": [0.0, 1.0]})
    table = build_rank_table("정답", ["가", "나", "가"], provider)
    calls_after_build = list(provider.calls)

    for _ in range(3):
        assert table.rank_of("가") == 2
        assert table.similarity_of("나") == pytest.approx(0.0)
        assert table.lookup("없는단어") is None

    assert provider.calls == calls_after_build == ["정답", "가", "나"]
    with pytest.raises(TypeError):
        table.entries["새단어"] = table.ranked_entries[0]  # type: ignore[index]


def test_fasttext_adapter_allows_subword_oov() -> None:
    class StubFastTextModel:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get_word_vector(self, word: str) -> list[float]:
            self.calls.append(word)
            return [1.0, 0.0] if word == "정답" else [0.25, 0.75]

        def get_word_id(self, word: str) -> int:
            raise AssertionError("rank building must not reject subword OOV words")

    model = StubFastTextModel()
    provider = FastTextVectorProvider(model)

    table = build_rank_table("정답", ["사전에없는단어"], provider)

    assert table.rank_of("사전에없는단어") == 2
    assert model.calls == ["정답", "사전에없는단어"]


def test_empty_answer_is_rejected() -> None:
    with pytest.raises(RankTableError, match="Answer must not be empty"):
        build_rank_table("  ", [], FakeVectorProvider({}))


def test_missing_vocabulary_file_has_clear_error(tmp_path: Path) -> None:
    with pytest.raises(VocabularyLoadError, match="Vocabulary file not found"):
        load_vocabulary(tmp_path / "missing.txt")
