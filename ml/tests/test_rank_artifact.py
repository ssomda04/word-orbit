"""Unit tests for compact rank artifacts; no real FastText model required."""

from __future__ import annotations

import json
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from contextle_eval.fasttext_provider import FastTextVectorProvider
from contextle_eval.rank_artifact import (
    RankArtifactError,
    VocabularyIndex,
    build_artifact,
    build_normalized_vector_matrix,
    load_artifact_npy,
    load_artifact_npz,
    save_artifact_npy,
    save_artifact_npz,
)
from contextle_eval.rank_table import build_rank_table


class FakeProvider:
    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self.vectors = vectors

    def vector(self, word: str) -> Sequence[float]:
        return self.vectors[word]


def _fixture() -> tuple[VocabularyIndex, FakeProvider]:
    vocabulary = VocabularyIndex.create(["나", "정답", "가", "직교", "반대"])
    provider = FakeProvider(
        {
            "정답": [1.0, 0.0],
            "가": [1.0, 0.0],
            "나": [1.0, 0.0],
            "직교": [0.0, 1.0],
            "반대": [-1.0, 0.0],
        }
    )
    return vocabulary, provider


def test_vocabulary_index_and_hash_are_order_deterministic() -> None:
    first = VocabularyIndex.create([" Ａ ", "B", "A", "가"])
    second = VocabularyIndex.create(["A", "B", "가"])

    assert first.words == second.words == ("A", "B", "가")
    assert first.sha256 == second.sha256
    assert first.index_of(" Ａ ") == 0
    assert first.index_of("missing") is None


@pytest.mark.parametrize("format_name", ["npy", "npz"])
def test_artifact_round_trip_and_ranktable_equivalence(tmp_path, format_name: str) -> None:
    vocabulary, provider = _fixture()
    matrix = build_normalized_vector_matrix(vocabulary, provider)
    artifact = build_artifact("정답", vocabulary, matrix, embedding_model="fake")
    expected = build_rank_table("정답", vocabulary.words, provider)

    if format_name == "npy":
        path = save_artifact_npy(tmp_path, artifact)
        loaded = load_artifact_npy(path, vocabulary)
    else:
        path = save_artifact_npz(tmp_path, artifact)
        loaded = load_artifact_npz(path, vocabulary)

    assert loaded.lookup("정답", vocabulary) == pytest.approx((1.0, 1))
    assert loaded.lookup("missing", vocabulary) is None
    assert sorted(int(rank) for rank in loaded.ranks) == list(range(1, 6))
    for word in vocabulary.words:
        similarity, rank = loaded.lookup(word, vocabulary) or (None, None)
        assert rank == expected.rank_of(word)
        assert similarity == pytest.approx(expected.similarity_of(word), abs=1e-6)


def test_vocabulary_hash_mismatch_is_rejected(tmp_path) -> None:
    vocabulary, provider = _fixture()
    matrix = build_normalized_vector_matrix(vocabulary, provider)
    path = save_artifact_npy(
        tmp_path, build_artifact("정답", vocabulary, matrix, embedding_model="fake")
    )
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["vocabulary_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(RankArtifactError, match="hash"):
        load_artifact_npy(path, vocabulary)


def test_float16_keeps_authoritative_rank_and_answer_rank_one() -> None:
    vocabulary, provider = _fixture()
    matrix = build_normalized_vector_matrix(vocabulary, provider)
    artifact = build_artifact(
        "정답", vocabulary, matrix, embedding_model="fake", similarity_dtype="float16"
    )

    assert artifact.similarities.dtype == np.float16
    assert artifact.ranks[vocabulary.index_of("정답")] == 1
    assert len(np.unique(artifact.ranks)) == len(vocabulary.words)


def test_real_fasttext_artifact_matches_ranktable() -> None:
    """Optional full-vocabulary integration check against the existing implementation."""
    model_path = os.getenv("FASTTEXT_MODEL_PATH")
    vocabulary_path = os.getenv("FASTTEXT_VOCABULARY_PATH")
    if not model_path or not vocabulary_path:
        pytest.skip("FASTTEXT_MODEL_PATH and FASTTEXT_VOCABULARY_PATH are required")
    vocabulary = VocabularyIndex.from_file(Path(vocabulary_path))
    provider = FastTextVectorProvider.load(Path(model_path))
    matrix = build_normalized_vector_matrix(vocabulary, provider)
    answer = "주변"
    artifact = build_artifact(
        answer,
        vocabulary,
        matrix,
        embedding_model=Path(model_path).name,
        ranktable_compatibility_provider=provider,
    )
    table = build_rank_table(answer, vocabulary.words, provider)

    expected_ranks = np.empty(len(vocabulary.words), dtype=artifact.ranks.dtype)
    for entry in table.ranked_entries:
        index = vocabulary.index_of(entry.word)
        assert index is not None
        expected_ranks[index] = entry.rank
    mismatches = np.flatnonzero(artifact.ranks != expected_ranks)
    assert not len(mismatches), (
        f"rank mismatch count={len(mismatches)}, "
        f"max delta={max(abs(int(artifact.ranks[i]) - int(expected_ranks[i])) for i in mismatches)}, "
        f"sample={[(vocabulary.words[int(i)], int(artifact.ranks[i]), int(expected_ranks[i])) for i in mismatches[:10]]}"
    )
    for index in random.Random(20260823).sample(range(len(vocabulary.words)), 100):
        entry = table.lookup(vocabulary.words[index])
        assert entry is not None
        assert int(artifact.ranks[index]) == entry.rank
        assert float(artifact.similarities[index]) == pytest.approx(entry.similarity, abs=1e-6)
