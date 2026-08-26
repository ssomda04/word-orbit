"""Build and load compact per-answer similarity/rank artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from contextle_eval.frequency_calibration import canonical_pos_parts
from contextle_eval.rank_table import (
    WordVectorProvider,
    _zero_safe_cosine,
    normalize_vocabulary,
    normalize_word,
)

SCHEMA_VERSION = "1.0.0-prototype"
ARTIFACT_ROOT_SCHEMA_VERSION = "1.0"
ARTIFACT_ID_ALGORITHM = "sha256-nfkc-utf8"
RANKING_POLICY = "answer_rank_1_then_cosine_desc_lexical_tiebreak"
ARTIFACT_ROOT_RANKING_POLICY = {
    "metric": "cosine",
    "answer_rank": 1,
    "order": "similarity_desc",
    "tie_break": "lexical",
}
DEFAULT_SEED = 20260823
SUPPORTED_SIMILARITY_DTYPES = frozenset({"float32", "float16"})
RANKTABLE_REFINEMENT_EPSILON = 1e-7


class RankArtifactError(RuntimeError):
    """Raised when an artifact cannot be built, validated, or loaded."""


@dataclass(frozen=True, slots=True)
class VocabularyIndex:
    """Canonical vocabulary order, lookup map, and reproducibility hash."""

    words: tuple[str, ...]
    word_to_index: Mapping[str, int]
    sha256: str

    @classmethod
    def create(cls, words: Sequence[str]) -> VocabularyIndex:
        normalized = normalize_vocabulary(words)
        if not normalized:
            raise RankArtifactError("Vocabulary must contain at least one word.")
        mapping = {word: index for index, word in enumerate(normalized)}
        payload = "".join(f"{word}\n" for word in normalized).encode("utf-8")
        return cls(normalized, mapping, hashlib.sha256(payload).hexdigest())

    @classmethod
    def from_file(cls, path: Path) -> VocabularyIndex:
        try:
            payload = path.read_bytes()
            if payload.startswith(b"\xef\xbb\xbf"):
                raise RankArtifactError("Vocabulary must not contain a UTF-8 BOM.")
            decoded = payload.decode("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise RankArtifactError(f"Could not read vocabulary {path}: {exc}") from exc
        vocabulary = cls.create(decoded.splitlines())
        if payload != _vocabulary_payload(vocabulary):
            raise RankArtifactError(
                "Vocabulary is not canonical NFKC+strip UTF-8 line data."
            )
        return vocabulary

    def index_of(self, word: str) -> int | None:
        normalized = normalize_word(word)
        return self.word_to_index.get(normalized) if normalized else None


@dataclass(frozen=True, slots=True)
class RankArtifact:
    """Loaded answer metadata plus canonical-indexed arrays."""

    metadata: Mapping[str, Any]
    similarities: NDArray[np.floating[Any]]
    ranks: NDArray[np.unsignedinteger[Any]]

    def lookup(self, word: str, vocabulary: VocabularyIndex) -> tuple[float, int] | None:
        index = vocabulary.index_of(word)
        if index is None:
            return None
        return float(self.similarities[index]), int(self.ranks[index])


def rank_dtype_for_size(vocabulary_size: int) -> np.dtype[Any]:
    """Return the smallest unsigned dtype that represents ranks 1..N."""
    if vocabulary_size < 1:
        raise RankArtifactError("Vocabulary size must be positive.")
    for dtype in (np.uint16, np.uint32, np.uint64):
        if vocabulary_size <= np.iinfo(dtype).max:
            return np.dtype(dtype)
    raise RankArtifactError("Vocabulary is too large for uint64 ranks.")


def build_normalized_vector_matrix(
    vocabulary: VocabularyIndex, provider: WordVectorProvider
) -> NDArray[np.float32]:
    """Load each vocabulary vector once and return float32 unit vectors."""
    vectors: list[NDArray[np.float32]] = []
    dimension: int | None = None
    for word in vocabulary.words:
        try:
            vector = np.asarray(provider.vector(word), dtype=np.float32)
        except Exception as exc:
            raise RankArtifactError(f"Could not vectorize {word!r}: {exc}") from exc
        if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
            raise RankArtifactError(f"Embedding for {word!r} is empty or non-finite.")
        if dimension is None:
            dimension = int(vector.size)
        elif vector.size != dimension:
            raise RankArtifactError(
                f"Embedding dimensions differ for {word!r}: {vector.size} != {dimension}."
            )
        norm = float(np.linalg.norm(vector.astype(np.float64)))
        vectors.append(vector / norm if norm else np.zeros_like(vector))
    return np.stack(vectors)


def similarities_for_answer(
    answer_index: int, normalized_vectors: NDArray[np.float32]
) -> NDArray[np.float64]:
    """Calculate finite cosine similarities with float64 accumulation."""
    if not 0 <= answer_index < len(normalized_vectors):
        raise RankArtifactError(f"Answer vocabulary index is out of range: {answer_index}")
    answer = normalized_vectors[answer_index].astype(np.float64)
    similarities = normalized_vectors.astype(np.float64) @ answer
    if not np.isfinite(similarities).all():
        raise RankArtifactError("Calculated similarities contain NaN or infinity.")
    np.clip(similarities, -1.0, 1.0, out=similarities)
    similarities[answer_index] = 1.0
    return similarities


def ranks_from_similarities(
    similarities: NDArray[np.floating[Any]],
    vocabulary: VocabularyIndex,
    answer_index: int,
) -> NDArray[np.unsignedinteger[Any]]:
    """Apply RankTable ordering while storing ranks by vocabulary index."""
    size = len(vocabulary.words)
    if similarities.shape != (size,) or not np.isfinite(similarities).all():
        raise RankArtifactError("Similarity array has the wrong shape or non-finite values.")
    lexical_order = np.argsort(np.asarray(vocabulary.words), kind="stable")
    lexical_key = np.empty(size, dtype=np.uint32)
    lexical_key[lexical_order] = np.arange(size, dtype=np.uint32)
    candidate_indices = np.arange(size, dtype=np.uint32)
    candidate_indices = candidate_indices[candidate_indices != answer_index]
    ordered = candidate_indices[
        np.lexsort((lexical_key[candidate_indices], -similarities[candidate_indices]))
    ]
    ranks = np.empty(size, dtype=rank_dtype_for_size(size))
    ranks[answer_index] = 1
    ranks[ordered] = np.arange(2, size + 1, dtype=ranks.dtype)
    return ranks


def refine_ranktable_near_ties(
    similarities: NDArray[np.float64],
    vocabulary: VocabularyIndex,
    answer_index: int,
    provider: WordVectorProvider,
    *,
    epsilon: float = RANKTABLE_REFINEMENT_EPSILON,
) -> NDArray[np.float64]:
    """Recompute numerically close scores with RankTable's exact Python cosine.

    BLAS and Python sequential summation can order nearly equal values differently.
    Only candidates close enough to be affected are recomputed with RankTable's
    implementation, preserving vectorized performance for the rest.
    """
    order = np.argsort(-similarities, kind="stable")
    gaps = np.abs(np.diff(similarities[order]))
    close = gaps <= epsilon
    refine_mask = np.zeros(len(similarities), dtype=bool)
    refine_mask[order[:-1][close]] = True
    refine_mask[order[1:][close]] = True
    refine_mask[answer_index] = True
    answer_vector = tuple(float(value) for value in provider.vector(vocabulary.words[answer_index]))
    refined = similarities.copy()
    for index in np.flatnonzero(refine_mask):
        word_vector = (
            answer_vector
            if index == answer_index
            else tuple(float(value) for value in provider.vector(vocabulary.words[int(index)]))
        )
        refined[index] = _zero_safe_cosine(answer_vector, word_vector)
    refined[answer_index] = 1.0
    return refined


def build_artifact(
    answer: str,
    vocabulary: VocabularyIndex,
    normalized_vectors: NDArray[np.float32],
    *,
    embedding_model: str,
    similarity_dtype: str = "float32",
    ranktable_compatibility_provider: WordVectorProvider | None = None,
) -> RankArtifact:
    """Build one artifact without repeating vocabulary strings."""
    if similarity_dtype not in SUPPORTED_SIMILARITY_DTYPES:
        raise RankArtifactError(f"Unsupported similarity dtype: {similarity_dtype}")
    answer_index = vocabulary.index_of(answer)
    if answer_index is None:
        raise RankArtifactError(f"Answer is missing from canonical vocabulary: {answer!r}")
    full_precision = similarities_for_answer(answer_index, normalized_vectors)
    if ranktable_compatibility_provider is not None:
        full_precision = refine_ranktable_near_ties(
            full_precision,
            vocabulary,
            answer_index,
            ranktable_compatibility_provider,
        )
    ranks = ranks_from_similarities(full_precision, vocabulary, answer_index)
    return artifact_from_calculated(
        answer,
        vocabulary,
        full_precision,
        ranks,
        embedding_model=embedding_model,
        similarity_dtype=similarity_dtype,
    )


def artifact_from_calculated(
    answer: str,
    vocabulary: VocabularyIndex,
    full_precision: NDArray[np.float64],
    ranks: NDArray[np.unsignedinteger[Any]],
    *,
    embedding_model: str,
    similarity_dtype: str,
) -> RankArtifact:
    """Package already-calculated scores/ranks for serialization benchmarks."""
    if similarity_dtype not in SUPPORTED_SIMILARITY_DTYPES:
        raise RankArtifactError(f"Unsupported similarity dtype: {similarity_dtype}")
    answer_index = vocabulary.index_of(answer)
    if answer_index is None:
        raise RankArtifactError(f"Answer is missing from canonical vocabulary: {answer!r}")
    stored_similarities = full_precision.astype(similarity_dtype)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": embedding_model,
        "vocabulary_size": len(vocabulary.words),
        "vocabulary_sha256": vocabulary.sha256,
        "answer": vocabulary.words[answer_index],
        "answer_vocab_index": answer_index,
        "similarity_dtype": stored_similarities.dtype.name,
        "rank_dtype": ranks.dtype.name,
        "ranking_policy": RANKING_POLICY,
    }
    artifact = RankArtifact(metadata, stored_similarities, ranks)
    validate_artifact(artifact, vocabulary)
    return artifact


def validate_artifact(artifact: RankArtifact, vocabulary: VocabularyIndex) -> None:
    """Validate schema, vocabulary identity, arrays, and answer rank."""
    metadata = artifact.metadata
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise RankArtifactError("Unsupported artifact schema version.")
    if metadata.get("vocabulary_sha256") != vocabulary.sha256:
        raise RankArtifactError("Artifact vocabulary hash does not match canonical vocabulary.")
    size = len(vocabulary.words)
    if metadata.get("vocabulary_size") != size:
        raise RankArtifactError("Artifact vocabulary size does not match canonical vocabulary.")
    if artifact.similarities.shape != (size,) or artifact.ranks.shape != (size,):
        raise RankArtifactError("Artifact arrays do not match vocabulary size.")
    if not np.isfinite(artifact.similarities).all():
        raise RankArtifactError("Artifact similarities contain NaN or infinity.")
    if np.any((artifact.similarities < -1.0) | (artifact.similarities > 1.0)):
        raise RankArtifactError("Artifact similarities must be within [-1.0, 1.0].")
    answer_index = int(metadata.get("answer_vocab_index", -1))
    if not 0 <= answer_index < size or artifact.ranks[answer_index] != 1:
        raise RankArtifactError("Artifact answer must have rank 1.")
    if artifact.similarities[answer_index] != 1.0:
        raise RankArtifactError("Artifact answer similarity must be 1.0.")
    expected = np.arange(1, size + 1, dtype=artifact.ranks.dtype)
    if not np.array_equal(np.sort(artifact.ranks), expected):
        raise RankArtifactError("Artifact ranks must be unique and continuous from 1..N.")


def artifact_id_for_answer(answer: str) -> str:
    """Return the deterministic, answer-hiding artifact identifier."""
    normalized = normalize_word(answer)
    if not normalized:
        raise RankArtifactError("Artifact answer must not be empty after normalization.")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def artifact_relative_directory(answer: str) -> Path:
    """Return ``artifacts/<prefix>/<full sha256>`` for a normalized answer."""
    artifact_id = artifact_id_for_answer(answer)
    return Path("artifacts") / artifact_id[:2] / artifact_id


def save_artifact_npy(directory: Path, artifact: RankArtifact) -> Path:
    """Save a legacy standalone artifact for benchmark compatibility.

    Production generation uses :func:`write_artifact_root`; this standalone
    representation retains metadata only so the existing benchmark can load it.
    """
    target = directory / artifact_id_for_answer(str(artifact.metadata["answer"]))
    if target.exists():
        raise RankArtifactError(f"Artifact directory already exists: {target}")
    target.mkdir(parents=True)
    (target / "metadata.json").write_text(
        json.dumps(dict(artifact.metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.save(target / "similarity.npy", artifact.similarities, allow_pickle=False)
    np.save(target / "rank.npy", artifact.ranks, allow_pickle=False)
    return target


def save_artifact_npz(directory: Path, artifact: RankArtifact) -> Path:
    """Save one compressed .npz containing arrays and UTF-8 JSON metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{artifact_id_for_answer(str(artifact.metadata['answer']))}.npz"
    if target.exists():
        raise RankArtifactError(f"Artifact file already exists: {target}")
    metadata = json.dumps(dict(artifact.metadata), ensure_ascii=False, separators=(",", ":"))
    np.savez_compressed(
        target,
        similarity=artifact.similarities,
        rank=artifact.ranks,
        metadata=metadata,
    )
    return target


def load_artifact_npy(
    directory: Path, vocabulary: VocabularyIndex, *, mmap: bool = False
) -> RankArtifact:
    """Load a legacy .npy directory, using ordinary in-memory arrays by default."""
    try:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        mode = "r" if mmap else None
        similarities = np.load(directory / "similarity.npy", mmap_mode=mode, allow_pickle=False)
        ranks = np.load(directory / "rank.npy", mmap_mode=mode, allow_pickle=False)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RankArtifactError(f"Could not load .npy artifact {directory}: {exc}") from exc
    artifact = RankArtifact(metadata, similarities, ranks)
    validate_artifact(artifact, vocabulary)
    return artifact


def load_artifact_npz(path: Path, vocabulary: VocabularyIndex) -> RankArtifact:
    """Load a compressed .npz artifact into memory."""
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            similarities = data["similarity"].copy()
            ranks = data["rank"].copy()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise RankArtifactError(f"Could not load .npz artifact {path}: {exc}") from exc
    artifact = RankArtifact(metadata, similarities, ranks)
    validate_artifact(artifact, vocabulary)
    return artifact


def _vocabulary_payload(vocabulary: VocabularyIndex) -> bytes:
    return "".join(f"{word}\n" for word in vocabulary.words).encode("utf-8")


def write_artifact_root(
    root: Path,
    vocabulary: VocabularyIndex,
    artifacts: Iterable[RankArtifact],
    *,
    embedding_model_name: str,
    embedding_model_source: str,
) -> dict[str, Any]:
    """Write the production artifact-root contract without per-answer metadata."""
    if not embedding_model_name.strip() or not embedding_model_source.strip():
        raise RankArtifactError("Embedding model name and source must not be blank.")
    if root.exists() and any(root.iterdir()):
        raise RankArtifactError(f"Artifact root must be new or empty: {root}")

    root.mkdir(parents=True, exist_ok=True)
    vocabulary_payload = _vocabulary_payload(vocabulary)
    vocabulary_path = root / "vocabulary.txt"
    vocabulary_path.write_bytes(vocabulary_payload)
    vocabulary_sha256 = hashlib.sha256(vocabulary_payload).hexdigest()
    if vocabulary_sha256 != vocabulary.sha256:
        raise RankArtifactError("Written vocabulary hash differs from the canonical vocabulary.")

    answer_entries: dict[str, dict[str, Any]] = {}
    similarity_dtype: str | None = None
    rank_dtype: str | None = None
    for artifact in artifacts:
        validate_artifact(artifact, vocabulary)
        answer = normalize_word(str(artifact.metadata.get("answer", "")))
        if not answer:
            raise RankArtifactError("Artifact metadata is missing a canonical answer.")
        if answer in answer_entries:
            raise RankArtifactError(f"Duplicate normalized answer: {answer!r}")
        current_similarity_dtype = artifact.similarities.dtype.name
        current_rank_dtype = artifact.ranks.dtype.name
        similarity_dtype = similarity_dtype or current_similarity_dtype
        rank_dtype = rank_dtype or current_rank_dtype
        if (
            current_similarity_dtype != similarity_dtype
            or current_rank_dtype != rank_dtype
        ):
            raise RankArtifactError("All artifacts in one root must use the same dtypes.")
        artifact_id = artifact_id_for_answer(answer)
        relative_directory = artifact_relative_directory(answer)
        target = root / relative_directory
        target.mkdir(parents=True, exist_ok=False)
        similarity_relative = relative_directory / "similarity.npy"
        rank_relative = relative_directory / "rank.npy"
        np.save(root / similarity_relative, artifact.similarities, allow_pickle=False)
        np.save(root / rank_relative, artifact.ranks, allow_pickle=False)
        answer_entries[answer] = {
            "artifact_id": artifact_id,
            "answer_vocab_index": int(artifact.metadata["answer_vocab_index"]),
            "similarity_path": similarity_relative.as_posix(),
            "rank_path": rank_relative.as_posix(),
        }

    if not answer_entries or similarity_dtype is None or rank_dtype is None:
        raise RankArtifactError("Artifact root must contain at least one answer.")

    manifest: dict[str, Any] = {
        "schema_version": ARTIFACT_ROOT_SCHEMA_VERSION,
        "embedding_model": {
            "name": embedding_model_name,
            "source": embedding_model_source,
        },
        "vocabulary": {
            "path": "vocabulary.txt",
            "size": len(vocabulary.words),
            "sha256": vocabulary_sha256,
        },
        "similarity_dtype": similarity_dtype,
        "rank_dtype": rank_dtype,
        "artifact_id_algorithm": ARTIFACT_ID_ALGORITHM,
        "ranking_policy": dict(ARTIFACT_ROOT_RANKING_POLICY),
        "answers": answer_entries,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_artifact_root_manifest(root: Path) -> tuple[dict[str, Any], VocabularyIndex]:
    """Load and validate authoritative root metadata and canonical vocabulary."""
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RankArtifactError(f"Could not load artifact manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise RankArtifactError("Artifact manifest root must be an object.")
    if manifest.get("schema_version") != ARTIFACT_ROOT_SCHEMA_VERSION:
        raise RankArtifactError("Unsupported artifact-root schema version.")
    if manifest.get("artifact_id_algorithm") != ARTIFACT_ID_ALGORITHM:
        raise RankArtifactError("Unsupported artifact id algorithm.")
    if manifest.get("ranking_policy") != ARTIFACT_ROOT_RANKING_POLICY:
        raise RankArtifactError("Unsupported ranking policy.")
    embedding_model = manifest.get("embedding_model")
    if not isinstance(embedding_model, dict) or any(
        not isinstance(embedding_model.get(field), str)
        or not embedding_model[field].strip()
        for field in ("name", "source")
    ):
        raise RankArtifactError("Manifest embedding model metadata is invalid.")

    vocabulary_metadata = manifest.get("vocabulary")
    if not isinstance(vocabulary_metadata, dict):
        raise RankArtifactError("Manifest vocabulary metadata must be an object.")
    if vocabulary_metadata.get("path") != "vocabulary.txt":
        raise RankArtifactError("Manifest vocabulary path must be vocabulary.txt.")
    vocabulary_path = root / "vocabulary.txt"
    try:
        payload = vocabulary_path.read_bytes()
        if payload.startswith(b"\xef\xbb\xbf"):
            raise RankArtifactError("vocabulary.txt must not contain a UTF-8 BOM.")
        decoded = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RankArtifactError(f"Could not read canonical vocabulary: {exc}") from exc
    vocabulary = VocabularyIndex.create(decoded.splitlines())
    if payload != _vocabulary_payload(vocabulary):
        raise RankArtifactError("vocabulary.txt is not canonical NFKC+strip UTF-8 line data.")
    if vocabulary_metadata.get("size") != len(vocabulary.words):
        raise RankArtifactError("Manifest vocabulary size does not match vocabulary.txt.")
    actual_hash = hashlib.sha256(payload).hexdigest()
    if vocabulary_metadata.get("sha256") != actual_hash:
        raise RankArtifactError("Manifest vocabulary hash does not match vocabulary.txt.")

    try:
        similarity_dtype = np.dtype(manifest["similarity_dtype"])
        rank_dtype = np.dtype(manifest["rank_dtype"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RankArtifactError("Manifest contains a missing or invalid dtype.") from exc
    if similarity_dtype.kind != "f" or rank_dtype.kind != "u":
        raise RankArtifactError("Manifest dtypes must be floating similarity and unsigned rank.")
    answers = manifest.get("answers")
    if not isinstance(answers, dict) or not answers:
        raise RankArtifactError("Manifest answers must be a non-empty object.")
    return manifest, vocabulary


def load_artifact_root_answer(
    root: Path,
    answer: str,
    *,
    manifest: Mapping[str, Any] | None = None,
    vocabulary: VocabularyIndex | None = None,
) -> RankArtifact:
    """Load one answer with ordinary ``np.load`` and validate the root contract."""
    if manifest is None or vocabulary is None:
        loaded_manifest, loaded_vocabulary = load_artifact_root_manifest(root)
        manifest = loaded_manifest
        vocabulary = loaded_vocabulary
    normalized_answer = normalize_word(answer)
    answer_metadata = manifest["answers"].get(normalized_answer)
    if not isinstance(answer_metadata, dict):
        raise RankArtifactError("Requested answer is missing from the artifact manifest.")

    artifact_id = artifact_id_for_answer(normalized_answer)
    expected_directory = artifact_relative_directory(normalized_answer)
    expected_similarity = (expected_directory / "similarity.npy").as_posix()
    expected_rank = (expected_directory / "rank.npy").as_posix()
    if answer_metadata.get("artifact_id") != artifact_id:
        raise RankArtifactError("Manifest answer artifact id is invalid.")
    if answer_metadata.get("similarity_path") != expected_similarity:
        raise RankArtifactError("Manifest similarity path is not the canonical hash-only path.")
    if answer_metadata.get("rank_path") != expected_rank:
        raise RankArtifactError("Manifest rank path is not the canonical hash-only path.")

    try:
        similarities = np.load(root / expected_similarity, allow_pickle=False)
        ranks = np.load(root / expected_rank, allow_pickle=False)
    except Exception as exc:
        raise RankArtifactError(
            f"Could not load answer artifact {artifact_id} ({type(exc).__name__})."
        ) from None
    size = len(vocabulary.words)
    if similarities.shape != (size,) or ranks.shape != (size,):
        raise RankArtifactError("Artifact arrays do not match manifest vocabulary size.")
    if similarities.dtype.name != manifest.get("similarity_dtype"):
        raise RankArtifactError("Similarity array dtype does not match manifest.")
    if ranks.dtype.name != manifest.get("rank_dtype"):
        raise RankArtifactError("Rank array dtype does not match manifest.")
    answer_index = answer_metadata.get("answer_vocab_index")
    if not isinstance(answer_index, int) or vocabulary.index_of(normalized_answer) != answer_index:
        raise RankArtifactError("Manifest answer vocabulary index is invalid.")

    legacy_metadata = {
        "schema_version": SCHEMA_VERSION,
        "vocabulary_size": size,
        "vocabulary_sha256": vocabulary.sha256,
        "answer": normalized_answer,
        "answer_vocab_index": answer_index,
        "similarity_dtype": similarities.dtype.name,
        "rank_dtype": ranks.dtype.name,
        "ranking_policy": RANKING_POLICY,
    }
    artifact = RankArtifact(legacy_metadata, similarities, ranks)
    validate_artifact(artifact, vocabulary)
    return artifact


def validate_artifact_root(root: Path) -> dict[str, Any]:
    """Validate every answer mapping and array in an artifact root."""
    manifest, vocabulary = load_artifact_root_manifest(root)
    for answer in manifest["answers"]:
        load_artifact_root_answer(
            root, answer, manifest=manifest, vocabulary=vocabulary
        )
    return manifest


def artifact_size_bytes(path: Path) -> int:
    """Return total bytes for either a file or artifact directory."""
    if path.is_file():
        return path.stat().st_size
    return sum(member.stat().st_size for member in path.rglob("*") if member.is_file())


def select_benchmark_answers(
    answer_rows: Sequence[tuple[str, str]], *, count: int, seed: int = DEFAULT_SEED
) -> tuple[str, ...]:
    """Select a deterministic POS-mixed sample; the first 10 of 50 are stable."""
    import random

    if count not in {10, 50}:
        raise RankArtifactError("Prototype benchmark count must be 10 or 50.")
    by_pos: defaultdict[str, list[str]] = defaultdict(list)
    for word, raw_pos in answer_rows:
        parts = canonical_pos_parts(raw_pos)
        if len(parts) == 1:
            by_pos[next(iter(parts))].append(normalize_word(word))
    required = {"noun", "verb", "adjective", "adverb"}
    if any(not by_pos[pos] for pos in required):
        raise RankArtifactError("Answer pool must contain every content POS.")
    rng = random.Random(seed)
    shuffled = {pos: rng.sample(sorted(by_pos[pos]), len(by_pos[pos])) for pos in sorted(required)}
    quotas_50 = {"noun": 30, "verb": 8, "adjective": 6, "adverb": 6}
    selected_50: list[str] = []
    for round_index in range(max(quotas_50.values())):
        for pos in ("noun", "verb", "adjective", "adverb"):
            if round_index < quotas_50[pos]:
                selected_50.append(shuffled[pos][round_index])
    first_10 = selected_50[:10]
    return tuple(first_10 if count == 10 else selected_50)


def process_rss_bytes() -> int | None:
    """Return current process RSS on platforms exposing resource usage."""
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_current_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
            get_current_process.restype = wintypes.HANDLE
            get_process_memory_info.argtypes = (
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            )
            get_process_memory_info.restype = wintypes.BOOL
            if get_process_memory_info(
                get_current_process(), ctypes.byref(counters), counters.cb
            ):
                return int(counters.WorkingSetSize)
        except (AttributeError, OSError):
            return None
        return None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss * (1024 if os.name != "nt" else 1))
    except (ImportError, OSError):
        return None


def timed_lookup(artifact: RankArtifact, indices: NDArray[np.integer[Any]]) -> dict[str, float]:
    """Benchmark scalar and vectorized array lookup without model work."""
    one_started = time.perf_counter_ns()
    _ = (artifact.similarities[int(indices[0])], artifact.ranks[int(indices[0])])
    one_ns = time.perf_counter_ns() - one_started
    many_started = time.perf_counter_ns()
    _ = (artifact.similarities[indices], artifact.ranks[indices])
    many_ns = time.perf_counter_ns() - many_started
    return {"one_lookup_ns": float(one_ns), "batch_lookup_ns": float(many_ns)}
