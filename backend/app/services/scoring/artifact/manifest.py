"""Reading and validating ``manifest.json`` — the authority for an artifact root.

An artifact root is data the server did not produce, addressed by files the
server did not write, describing arrays the server cannot re-derive. So the
manifest is treated as untrusted input and checked exhaustively *before*
anything is served, not lazily on the first guess: a root that is wrong should
stop the process, the way a bad ``FASTTEXT_MODEL_PATH`` does.

Three properties are worth stating outright, because they are what the checks
below exist to guarantee:

1. **Paths are derived, never accepted.** The manifest records the two file
   paths per answer, but they are recomputed from the ``artifact_id`` and
   compared for exact equality. A manifest therefore cannot choose an arbitrary
   artifact path — `..`, an absolute path, and a different spelling of the same
   path all fail one comparison, with no rule of their own. That constrains the
   *path*, not where the bytes ultimately live: a symlink at the canonical
   location is still followed, which ``docs/ARTIFACT_FORMAT.md`` records as out
   of scope for the current threat model.
2. **The vocabulary is verified three ways**, not one: the file must be
   canonical on its own terms, its size must match the manifest, and its sha256
   must match the manifest. Any single mismatch means the stored arrays are
   indexed by a different word list than the server would use, which would
   silently mis-score every guess rather than fail.
3. **No message here names an answer.** Entries are identified by
   ``artifact_id``, or by position when the id itself is what is wrong. See
   ``errors``.

Validation deliberately stricter than the writer
------------------------------------------------
Some checks below have no counterpart in ``ml/src/contextle_eval/rank_artifact.py``
as of the root-contract change: an empty ``answers`` object, and the byte-order
mark refused in ``vocabulary``. They are kept because this is the consuming side
and a consumer that trusts a producer's invariants has no invariants of its own.
``docs/ARTIFACT_FORMAT.md`` tracks the differences.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.services.scoring.artifact.errors import ArtifactError
from app.services.scoring.artifact.paths import (
    ARTIFACT_ID_ALGORITHM,
    artifact_id_for,
    rank_path,
    similarity_path,
)
from app.services.scoring.artifact.vocabulary import (
    CanonicalVocabulary,
    read_canonical_vocabulary,
)

MANIFEST_FILENAME = "manifest.json"
VOCABULARY_FILENAME = "vocabulary.txt"

# The one root layout this backend understands. Widening it is a decision, not a
# patch: a new version may reorder or reinterpret the arrays.
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0"})

# Enumerated rather than "any floating type" / "any unsigned type". The writer
# emits exactly these (`SUPPORTED_SIMILARITY_DTYPES` and `rank_dtype_for_size`
# in the ML harness), and a wider contract would let the server accept an array
# it can read but was never taught to reason about — float64 similarities would
# quietly double every artifact, and uint8 ranks cannot address a real game
# vocabulary at all.
SUPPORTED_SIMILARITY_DTYPES = frozenset({"float32", "float16"})
SUPPORTED_RANK_DTYPES = frozenset({"uint16", "uint32", "uint64"})

# The ordering the stored ranks were produced by. Compared for exact equality:
# if the producer changes any part of it, the ranks mean something different and
# the server must not serve them as though they did not.
EXPECTED_RANKING_POLICY: Mapping[str, Any] = {
    "metric": "cosine",
    "answer_rank": 1,
    "order": "similarity_desc",
    "tie_break": "lexical",
}


@dataclass(frozen=True, slots=True)
class AnswerEntry:
    """Where one answer's arrays live, and which slot in them is the answer.

    The manifest's two path strings are validated and then dropped: they are a
    function of ``artifact_id``, so keeping them would store the same fact twice
    and invite the two copies to disagree. Paths are recomputed on load.
    """

    artifact_id: str
    vocab_index: int


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """A validated artifact root, ready to serve answers."""

    root: Path
    schema_version: str
    embedding_model_name: str
    embedding_model_source: str
    similarity_dtype: str
    rank_dtype: str
    vocabulary: CanonicalVocabulary
    _entries: Mapping[str, AnswerEntry] = field(repr=False, compare=False)

    @property
    def answers(self) -> tuple[str, ...]:
        """Every answer this root can serve, in manifest order.

        Plain answer words: internal server data, and the source of truth a
        future answer selector will draw from. Never log this.
        """
        return tuple(self._entries)

    @property
    def answer_count(self) -> int:
        return len(self._entries)

    def entry_for(self, answer: str) -> AnswerEntry | None:
        """Return the entry for an already-normalized ``answer``, or None.

        Returns ``None`` rather than raising, so that "this root has no artifact
        for that answer" can never be reported through a message that would have
        to name it.
        """
        return self._entries.get(answer)

    def index_of(self, word: str) -> int | None:
        """Return the canonical vocabulary index of ``word``, or None."""
        return self.vocabulary.index_of(word)


def load_manifest(root: Path) -> ArtifactManifest:
    """Read and fully validate the artifact root at ``root``.

    Every referenced array file is checked for existence, but none is read: a
    root of a few thousand answers is hundreds of megabytes, and the arrays are
    validated when an answer is actually loaded.

    Raises:
        ArtifactError: anything about the root is missing, malformed, mutually
            inconsistent, or outside the supported contract.
    """
    manifest = _read_manifest_document(root / MANIFEST_FILENAME)

    _validate_schema(manifest)
    _validate_ranking_policy(manifest)
    name, source = _validate_embedding_model(manifest)
    vocabulary = _validate_vocabulary(root, manifest)
    similarity_dtype, rank_dtype = _validate_dtypes(manifest, vocabulary.size)
    entries = _validate_answers(root, manifest, vocabulary)

    return ArtifactManifest(
        root=root,
        schema_version=str(manifest["schema_version"]),
        embedding_model_name=name,
        embedding_model_source=source,
        similarity_dtype=similarity_dtype,
        rank_dtype=rank_dtype,
        vocabulary=vocabulary,
        _entries=entries,
    )


# --- Document -----------------------------------------------------------------


def _read_manifest_document(path: Path) -> Mapping[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ArtifactError(f"Could not read the artifact manifest: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ArtifactError(f"Artifact manifest is not valid UTF-8: {exc}") from exc

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        # `JSONDecodeError.__str__` reports the message and a position, never the
        # document body, so chaining cannot leak an answer from the manifest.
        raise ArtifactError(f"Artifact manifest is not valid JSON: {exc}") from exc

    if not isinstance(document, dict):
        raise ArtifactError("Artifact manifest must be a JSON object.")
    return document


# --- Schema and policy --------------------------------------------------------


def _validate_schema(manifest: Mapping[str, Any]) -> None:
    version = manifest.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        supported = ", ".join(sorted(SUPPORTED_SCHEMA_VERSIONS))
        raise ArtifactError(
            f"Unsupported artifact schema_version: {version!r}. Supported: {supported}."
        )
    algorithm = manifest.get("artifact_id_algorithm")
    if algorithm != ARTIFACT_ID_ALGORITHM:
        raise ArtifactError(
            f"Unsupported artifact_id_algorithm: {algorithm!r}. "
            f"Expected {ARTIFACT_ID_ALGORITHM!r}."
        )


def _validate_ranking_policy(manifest: Mapping[str, Any]) -> None:
    policy = manifest.get("ranking_policy")
    if policy != EXPECTED_RANKING_POLICY:
        raise ArtifactError(
            "Artifact ranking_policy does not match the policy this server "
            f"serves. Expected {dict(EXPECTED_RANKING_POLICY)}."
        )


def _validate_embedding_model(manifest: Mapping[str, Any]) -> tuple[str, str]:
    model = manifest.get("embedding_model")
    if not isinstance(model, dict):
        raise ArtifactError("Artifact embedding_model must be an object.")
    values: list[str] = []
    for key in ("name", "source"):
        value = model.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ArtifactError(
                f"Artifact embedding_model.{key} must be a non-empty string."
            )
        values.append(value)
    return values[0], values[1]


# --- Vocabulary ---------------------------------------------------------------


def _validate_vocabulary(root: Path, manifest: Mapping[str, Any]) -> CanonicalVocabulary:
    declared = manifest.get("vocabulary")
    if not isinstance(declared, dict):
        raise ArtifactError("Artifact vocabulary metadata must be an object.")
    if declared.get("path") != VOCABULARY_FILENAME:
        raise ArtifactError(
            f"Artifact vocabulary.path must be {VOCABULARY_FILENAME!r}; the "
            "vocabulary always travels inside the root."
        )

    vocabulary = read_canonical_vocabulary(root / VOCABULARY_FILENAME)

    if declared.get("size") != vocabulary.size:
        raise ArtifactError(
            f"Artifact vocabulary.size is {declared.get('size')!r} but "
            f"{VOCABULARY_FILENAME} holds {vocabulary.size} words."
        )
    if declared.get("sha256") != vocabulary.sha256:
        raise ArtifactError(
            f"Artifact vocabulary.sha256 does not match {VOCABULARY_FILENAME}. "
            "The stored arrays were built against a different word list."
        )
    return vocabulary


# --- Dtypes -------------------------------------------------------------------


def _validate_dtypes(manifest: Mapping[str, Any], vocabulary_size: int) -> tuple[str, str]:
    similarity_dtype = manifest.get("similarity_dtype")
    if similarity_dtype not in SUPPORTED_SIMILARITY_DTYPES:
        supported = ", ".join(sorted(SUPPORTED_SIMILARITY_DTYPES))
        raise ArtifactError(
            f"Unsupported similarity_dtype: {similarity_dtype!r}. Supported: {supported}."
        )

    rank_dtype = manifest.get("rank_dtype")
    if rank_dtype not in SUPPORTED_RANK_DTYPES:
        supported = ", ".join(sorted(SUPPORTED_RANK_DTYPES))
        raise ArtifactError(
            f"Unsupported rank_dtype: {rank_dtype!r}. Supported: {supported}."
        )
    # A rank runs 1..N, so the type has to reach N. Without this a root built for
    # a smaller vocabulary and re-labelled would wrap silently rather than fail.
    representable = int(np.iinfo(np.dtype(rank_dtype)).max)
    if vocabulary_size > representable:
        raise ArtifactError(
            f"rank_dtype {rank_dtype!r} reaches {representable} but the "
            f"vocabulary holds {vocabulary_size} words."
        )
    return str(similarity_dtype), str(rank_dtype)


# --- Answers ------------------------------------------------------------------


def _validate_answers(
    root: Path, manifest: Mapping[str, Any], vocabulary: CanonicalVocabulary
) -> Mapping[str, AnswerEntry]:
    answers = manifest.get("answers")
    if not isinstance(answers, dict):
        raise ArtifactError("Artifact answers must be an object.")
    # A root that can serve nothing is a misconfiguration, not an empty game:
    # every game needs an answer, so serving from here would fail at the first
    # request instead of at startup.
    if not answers:
        raise ArtifactError("Artifact answers must not be empty.")

    entries: dict[str, AnswerEntry] = {}
    for position, (answer, declared) in enumerate(answers.items()):
        entry = _validate_answer_entry(root, position, answer, declared, vocabulary)
        entries[answer] = entry
    return entries


def _validate_answer_entry(
    root: Path,
    position: int,
    answer: Any,
    declared: Any,
    vocabulary: CanonicalVocabulary,
) -> AnswerEntry:
    """Validate one ``answers`` member.

    ``position`` identifies the entry while its ``artifact_id`` is still in
    doubt; once the id is known to be correct it identifies the entry instead.
    Neither is the answer word, which is what this function must never quote.
    """
    where = f"answers entry #{position}"
    if not isinstance(answer, str) or not answer:
        raise ArtifactError(f"{where} has a key that is not a non-empty string.")
    if not isinstance(declared, dict):
        raise ArtifactError(f"{where} must be an object.")

    try:
        expected_id = artifact_id_for(answer)
    except ValueError as exc:
        raise ArtifactError(f"{where} has a blank key after normalization.") from exc
    # The key is the lookup key at runtime, so it has to be the same string a
    # normalized guess produces — otherwise the answer is simply unreachable.
    index = vocabulary.index_of(answer)
    if index is None:
        raise ArtifactError(
            f"{where} is not a canonical vocabulary word; keys must be NFKC "
            "normalized and present in vocabulary.txt."
        )

    if declared.get("artifact_id") != expected_id:
        raise ArtifactError(
            f"{where} declares an artifact_id that is not the sha256 of its key."
        )
    where = f"answers entry {expected_id}"

    declared_index = declared.get("answer_vocab_index")
    # `bool` is an `int`; a JSON `true` here would otherwise pass as index 1.
    if not isinstance(declared_index, int) or isinstance(declared_index, bool):
        raise ArtifactError(f"{where} has a non-integer answer_vocab_index.")
    if declared_index != index:
        raise ArtifactError(
            f"{where} has answer_vocab_index {declared_index}, but its word sits "
            f"at index {index} in vocabulary.txt."
        )

    for key, expected_path in (
        ("similarity_path", similarity_path(expected_id)),
        ("rank_path", rank_path(expected_id)),
    ):
        if declared.get(key) != expected_path:
            raise ArtifactError(
                f"{where} has a {key} that is not its canonical hash-only path. "
                f"Expected {expected_path!r}."
            )
        # Existence only. Reading every array at startup would cost hundreds of
        # megabytes and seconds; a missing file, though, should fail here rather
        # than mid-game.
        if not (root / expected_path).is_file():
            raise ArtifactError(f"{where} refers to a missing file: {expected_path}.")

    return AnswerEntry(artifact_id=expected_id, vocab_index=index)
