"""Unit tests for compact rank artifacts; no real FastText model required."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
from contextle_eval.fasttext_provider import FastTextVectorProvider
from contextle_eval.rank_artifact import (
    RankArtifact,
    RankArtifactError,
    VocabularyIndex,
    artifact_id_for_answer,
    artifact_relative_directory,
    build_artifact,
    build_normalized_vector_matrix,
    load_artifact_npy,
    load_artifact_npz,
    load_artifact_root_answer,
    load_artifact_root_manifest,
    rank_dtype_for_size,
    save_artifact_npy,
    save_artifact_npz,
    validate_artifact,
    validate_artifact_root,
    write_artifact_root,
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


def _write_root(tmp_path: Path, *, name: str = "root") -> tuple[Path, VocabularyIndex]:
    vocabulary, provider = _fixture()
    matrix = build_normalized_vector_matrix(vocabulary, provider)
    artifacts = [
        build_artifact(answer, vocabulary, matrix, embedding_model="fake")
        for answer in ("정답", "가")
    ]
    root = tmp_path / name
    write_artifact_root(
        root,
        vocabulary,
        artifacts,
        embedding_model_name="fasttext-cc-ko-300",
        embedding_model_source="cc.ko.300.bin",
    )
    return root, vocabulary


def test_artifact_root_contract_and_normal_numpy_load(tmp_path: Path) -> None:
    root, vocabulary = _write_root(tmp_path)
    manifest = validate_artifact_root(root)
    vocabulary_payload = (root / "vocabulary.txt").read_bytes()

    assert vocabulary_payload.decode("utf-8").splitlines() == list(vocabulary.words)
    assert manifest["vocabulary"] == {
        "path": "vocabulary.txt",
        "size": len(vocabulary.words),
        "sha256": hashlib.sha256(vocabulary_payload).hexdigest(),
    }
    assert manifest["similarity_dtype"] == "float32"
    assert manifest["rank_dtype"] == "uint16"
    loaded = load_artifact_root_answer(root, " 정답 ")
    assert type(loaded.similarities) is np.ndarray
    assert type(loaded.ranks) is np.ndarray
    assert loaded.lookup("정답", vocabulary) == pytest.approx((1.0, 1))


def test_artifact_paths_are_deterministic_hash_only(tmp_path: Path) -> None:
    root, _ = _write_root(tmp_path)
    manifest, _ = load_artifact_root_manifest(root)
    expected_id = hashlib.sha256("정답".encode()).hexdigest()
    answer = manifest["answers"]["정답"]

    assert artifact_id_for_answer("  정답  ") == expected_id
    assert artifact_relative_directory("정답") == (
        Path("artifacts") / expected_id[:2] / expected_id
    )
    assert answer["artifact_id"] == expected_id
    assert answer["similarity_path"] == (
        f"artifacts/{expected_id[:2]}/{expected_id}/similarity.npy"
    )
    assert "정답" not in answer["similarity_path"]
    artifact_directory = root / "artifacts" / expected_id[:2] / expected_id
    assert sorted(path.name for path in artifact_directory.iterdir()) == [
        "rank.npy",
        "similarity.npy",
    ]


def test_manifest_generation_is_deterministic(tmp_path: Path) -> None:
    first, _ = _write_root(tmp_path, name="first")
    second, _ = _write_root(tmp_path, name="second")

    assert (first / "manifest.json").read_bytes() == (
        second / "manifest.json"
    ).read_bytes()
    assert (first / "vocabulary.txt").read_bytes() == (second / "vocabulary.txt").read_bytes()


def test_rank_dtype_scales_beyond_uint16() -> None:
    assert rank_dtype_for_size(65_535) == np.dtype(np.uint16)
    synthetic_vocabulary = VocabularyIndex.create(
        [f"word-{index}" for index in range(65_536)]
    )
    assert rank_dtype_for_size(len(synthetic_vocabulary.words)) == np.dtype(np.uint32)
    assert rank_dtype_for_size(np.iinfo(np.uint32).max + 1) == np.dtype(np.uint64)


def test_duplicate_answer_is_rejected(tmp_path: Path) -> None:
    vocabulary, provider = _fixture()
    matrix = build_normalized_vector_matrix(vocabulary, provider)
    artifact = build_artifact("정답", vocabulary, matrix, embedding_model="fake")

    with pytest.raises(RankArtifactError, match="Duplicate normalized answer"):
        write_artifact_root(
            tmp_path / "root",
            vocabulary,
            [artifact, artifact],
            embedding_model_name="fake",
            embedding_model_source="fake.bin",
        )


@pytest.mark.parametrize("content", [None, "{not-json", "[]"])
def test_missing_or_corrupt_manifest_is_rejected(tmp_path: Path, content: str | None) -> None:
    root = tmp_path / "root"
    root.mkdir()
    if content is not None:
        (root / "manifest.json").write_text(content, encoding="utf-8")

    with pytest.raises(RankArtifactError, match="manifest|Manifest"):
        load_artifact_root_manifest(root)


def test_empty_manifest_answers_are_rejected(tmp_path: Path) -> None:
    root, _ = _write_root(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["answers"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RankArtifactError, match="non-empty"):
        load_artifact_root_manifest(root)


@pytest.mark.parametrize("invalid_similarity", [1.0001, -1.0001])
def test_similarity_outside_cosine_range_is_rejected(invalid_similarity: float) -> None:
    vocabulary, provider = _fixture()
    artifact = build_artifact(
        "정답",
        vocabulary,
        build_normalized_vector_matrix(vocabulary, provider),
        embedding_model="fake",
    )
    similarities = artifact.similarities.copy()
    similarities[0] = invalid_similarity

    with pytest.raises(RankArtifactError, match="within"):
        validate_artifact(
            RankArtifact(artifact.metadata, similarities, artifact.ranks), vocabulary
        )


def test_similarity_cosine_boundaries_are_accepted() -> None:
    vocabulary, provider = _fixture()
    artifact = build_artifact(
        "정답",
        vocabulary,
        build_normalized_vector_matrix(vocabulary, provider),
        embedding_model="fake",
    )
    similarities = artifact.similarities.copy()
    similarities[0] = -1.0
    similarities[1] = 1.0

    validate_artifact(
        RankArtifact(artifact.metadata, similarities, artifact.ranks), vocabulary
    )


@pytest.mark.parametrize("answer_similarity", [0.0, 1.0 - 5e-7])
def test_answer_self_similarity_contract(answer_similarity: float) -> None:
    vocabulary, provider = _fixture()
    artifact = build_artifact(
        "정답",
        vocabulary,
        build_normalized_vector_matrix(vocabulary, provider),
        embedding_model="fake",
    )
    similarities = artifact.similarities.copy()
    similarities[int(artifact.metadata["answer_vocab_index"])] = answer_similarity
    candidate = RankArtifact(artifact.metadata, similarities, artifact.ranks)

    if answer_similarity == 0.0:
        with pytest.raises(RankArtifactError, match="similarity must be 1.0"):
            validate_artifact(candidate, vocabulary)
    else:
        validate_artifact(candidate, vocabulary)


def test_missing_answer_error_does_not_reveal_requested_answer(tmp_path: Path) -> None:
    root, _ = _write_root(tmp_path)
    hidden_answer = "secret<answer>\n"

    with pytest.raises(RankArtifactError) as exc_info:
        load_artifact_root_answer(root, hidden_answer)

    message = str(exc_info.value)
    assert hidden_answer not in message
    assert repr(hidden_answer) not in message
    assert "secret<answer>" not in message


@pytest.mark.parametrize(
    "payload",
    [
        b"\xef\xbb\xbf",
        b"\xff",
        b"\r\n",
        b" leading\n",
        b"trailing \n",
    ],
    ids=["bom", "malformed-utf8", "crlf", "leading-space", "trailing-space"],
)
def test_noncanonical_vocabulary_bytes_are_rejected(
    tmp_path: Path, payload: bytes
) -> None:
    root, _ = _write_root(tmp_path)
    canonical = (root / "vocabulary.txt").read_bytes()
    if payload == b"\xef\xbb\xbf":
        invalid = payload + canonical
    elif payload == b"\r\n":
        invalid = canonical.replace(b"\n", payload)
    elif payload.startswith(b" ") or payload.endswith(b" \n"):
        invalid = payload + canonical
    else:
        invalid = payload
    (root / "vocabulary.txt").write_bytes(invalid)

    with pytest.raises(RankArtifactError):
        load_artifact_root_manifest(root)


def test_writer_vocabulary_bytes_pass_strict_reader(tmp_path: Path) -> None:
    root, expected = _write_root(tmp_path)

    _, loaded = load_artifact_root_manifest(root)

    assert loaded == expected


def test_corrupt_vocabulary_hash_array_length_and_dtype_are_rejected(
    tmp_path: Path,
) -> None:
    root, _ = _write_root(tmp_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["vocabulary"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RankArtifactError, match="hash"):
        load_artifact_root_manifest(root)

    root, _ = _write_root(tmp_path, name="wrong-length")
    manifest, _ = load_artifact_root_manifest(root)
    path = root / manifest["answers"]["정답"]["rank_path"]
    np.save(path, np.asarray([1], dtype=np.uint16), allow_pickle=False)
    with pytest.raises(RankArtifactError, match="size"):
        load_artifact_root_answer(root, "정답")

    root, _ = _write_root(tmp_path, name="wrong-dtype")
    manifest, _ = load_artifact_root_manifest(root)
    path = root / manifest["answers"]["정답"]["rank_path"]
    np.save(path, np.arange(1, 6, dtype=np.uint32), allow_pickle=False)
    with pytest.raises(RankArtifactError, match="dtype"):
        load_artifact_root_answer(root, "정답")


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
    max_delta = max(
        (
            abs(int(artifact.ranks[index]) - int(expected_ranks[index]))
            for index in mismatches
        ),
        default=0,
    )
    sample = [
        (
            vocabulary.words[int(index)],
            int(artifact.ranks[index]),
            int(expected_ranks[index]),
        )
        for index in mismatches[:10]
    ]
    assert not len(mismatches), (
        f"rank mismatch count={len(mismatches)}, "
        f"max delta={max_delta}, "
        f"sample={sample}"
    )
    for index in random.Random(20260823).sample(range(len(vocabulary.words)), 100):
        entry = table.lookup(vocabulary.words[index])
        assert entry is not None
        assert int(artifact.ranks[index]) == entry.rank
        assert float(artifact.similarities[index]) == pytest.approx(entry.similarity, abs=1e-6)
