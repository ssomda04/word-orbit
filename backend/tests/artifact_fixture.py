"""Writes a synthetic artifact root, from scratch.

Not a test module — a builder the artifact tests use. It deliberately shares no
code with ``app.services.scoring.artifact``: it recomputes the id, the canonical
vocabulary bytes, the hash, the paths, and the ranking policy from the format
description in ``docs/ARTIFACT_FORMAT.md``. A fixture that called the reader's
own helpers would agree with the reader about a mistake in either of them, which
is exactly what these tests exist to catch. It also imports nothing from ``ml``,
needs no model, and needs no downloaded data.

Everything it writes is valid by default, so a test changes the single thing it
is about — one manifest field, one array element, one file — and asserts that
the reader rejects it.
"""

from __future__ import annotations

import hashlib
import json
import struct
import unicodedata
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

# Small, unambiguous, and deliberately not in code-point order, so a test that
# depends on lexical ordering cannot pass by accident.
VOCABULARY: tuple[str, ...] = ("바다", "하늘", "학생", "선생", "커피", "자동차")

# The answer used unless a test asks for another. Also the string every secrecy
# assertion looks for.
ANSWER = "바다"

SIMILARITY_DTYPE = "float32"
RANK_DTYPE = "uint16"

SCHEMA_VERSION = "1.0"
ARTIFACT_ID_ALGORITHM = "sha256-nfkc-utf8"
RANKING_POLICY: dict[str, Any] = {
    "metric": "cosine",
    "answer_rank": 1,
    "order": "similarity_desc",
    "tie_break": "lexical",
}
EMBEDDING_MODEL = {"name": "synthetic-test-model", "source": "synthetic.bin"}


def artifact_id(answer: str) -> str:
    """sha256 of the NFKC-normalized, trimmed answer — the directory name."""
    normalized = unicodedata.normalize("NFKC", answer).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def artifact_directory(answer: str) -> str:
    identifier = artifact_id(answer)
    return f"artifacts/{identifier[:2]}/{identifier}"


def similarity_relative_path(answer: str) -> str:
    return f"{artifact_directory(answer)}/similarity.npy"


def rank_relative_path(answer: str) -> str:
    return f"{artifact_directory(answer)}/rank.npy"


def vocabulary_bytes(vocabulary: Sequence[str]) -> bytes:
    """One word per line, LF separated, trailing newline — the hashed payload."""
    return "".join(f"{word}\n" for word in vocabulary).encode("utf-8")


def similarities_for(answer: str, vocabulary: Sequence[str]) -> list[float]:
    """Deterministic similarities spread across [-0.9, 0.9], answer at 1.0.

    Spread rather than arbitrary so a small vocabulary gets distinct scores —
    ties would make the expected ranks depend on the tie-break rule, and a
    fixture should not have to be right about that to produce a valid root. The
    span is relative to the vocabulary size so the range holds for any length.
    """
    span = max(len(vocabulary) - 1, 1)
    scores: list[float] = []
    for offset, word in enumerate(vocabulary):
        scores.append(1.0 if word == answer else 0.9 - 1.8 * offset / span)
    return scores


def ranks_for(answer: str, vocabulary: Sequence[str], similarities: Sequence[float]) -> list[int]:
    """Rank 1 for the answer, then similarity descending, ties broken by word.

    Indexes are precomputed rather than looked up per comparison: the wide
    vocabulary a dtype test needs would otherwise make this quadratic.
    """
    positions = {word: index for index, word in enumerate(vocabulary)}
    others = [word for word in vocabulary if word != answer]
    by_score = sorted(others, key=lambda word: (-similarities[positions[word]], word))
    order = {answer: 1}
    for position, word in enumerate(by_score, start=2):
        order[word] = position
    return [order[word] for word in vocabulary]


def write_root(
    root: Path,
    *,
    vocabulary: Sequence[str] = VOCABULARY,
    answers: Sequence[str] = (ANSWER,),
    similarity_dtype: str = SIMILARITY_DTYPE,
    rank_dtype: str = RANK_DTYPE,
) -> dict[str, Any]:
    """Write a complete, valid artifact root and return the manifest it wrote."""
    root.mkdir(parents=True, exist_ok=True)
    payload = vocabulary_bytes(vocabulary)
    (root / "vocabulary.txt").write_bytes(payload)

    positions = {word: index for index, word in enumerate(vocabulary)}
    entries: dict[str, Any] = {}
    for answer in answers:
        similarities = similarities_for(answer, vocabulary)
        ranks = ranks_for(answer, vocabulary, similarities)
        directory = root / artifact_directory(answer)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(
            directory / "similarity.npy",
            np.asarray(similarities, dtype=similarity_dtype),
            allow_pickle=False,
        )
        np.save(directory / "rank.npy", np.asarray(ranks, dtype=rank_dtype), allow_pickle=False)
        entries[answer] = {
            "artifact_id": artifact_id(answer),
            "answer_vocab_index": positions[answer],
            "similarity_path": similarity_relative_path(answer),
            "rank_path": rank_relative_path(answer),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_id_algorithm": ARTIFACT_ID_ALGORITHM,
        "embedding_model": dict(EMBEDDING_MODEL),
        "vocabulary": {
            "path": "vocabulary.txt",
            "size": len(vocabulary),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "similarity_dtype": similarity_dtype,
        "rank_dtype": rank_dtype,
        "ranking_policy": dict(RANKING_POLICY),
        "answers": entries,
    }
    write_manifest(root, manifest)
    return manifest


def read_manifest(root: Path) -> dict[str, Any]:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def write_manifest(root: Path, manifest: dict[str, Any]) -> None:
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def mutate_manifest(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Apply one change to a written manifest and save it again."""
    manifest = read_manifest(root)
    mutate(manifest)
    write_manifest(root, manifest)


def overwrite_similarity(root: Path, answer: str, values: np.ndarray) -> None:
    np.save(root / similarity_relative_path(answer), values, allow_pickle=False)


def overwrite_rank(root: Path, answer: str, values: np.ndarray) -> None:
    np.save(root / rank_relative_path(answer), values, allow_pickle=False)


def read_similarity(root: Path, answer: str) -> np.ndarray:
    return np.load(root / similarity_relative_path(answer), allow_pickle=False)


def read_rank(root: Path, answer: str) -> np.ndarray:
    return np.load(root / rank_relative_path(answer), allow_pickle=False)


# --- Deliberately malformed files ---------------------------------------------

# `.npy` container, hand-assembled because `np.save` cannot write a broken one.
# Magic, then a version, then the header length, then the header text itself.
_NPY_MAGIC = b"\x93NUMPY"

# Version 3.0 decodes its header as UTF-8 (1.0 and 2.0 use latin1, which would
# turn Korean into mojibake before numpy ever quoted it). It is a real version —
# numpy writes it whenever a dtype carries non-latin1 field names.
_NPY_VERSION = (3, 0)


def write_npy_with_header(root: Path, answer: str, filename: str, header: str) -> Path:
    """Replace one of an answer's arrays with a `.npy` carrying `header` verbatim.

    The container is well-formed enough that `np.load` reads the header and then
    fails on it, which is the point: numpy quotes what it read back into its own
    exception message. Pass a `header` that embeds a secret to prove the backend
    does not pass that message along.
    """
    body = header.encode("utf-8")
    body += b" " * (-(len(_NPY_MAGIC) + 2 + 4 + len(body)) % 64) + b"\n"
    payload = _NPY_MAGIC + bytes(_NPY_VERSION) + struct.pack("<I", len(body)) + body

    path = root / f"artifacts/{artifact_id(answer)[:2]}/{artifact_id(answer)}/{filename}"
    path.write_bytes(payload)
    return path
