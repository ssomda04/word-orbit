"""Loading and validating one answer's two arrays.

Loading strategy
----------------
Ordinary ``np.load``, fully into memory, no ``mmap_mode``. At a game vocabulary
of roughly 60k words an artifact is about 350 KB — small enough that mapping it
costs more than reading it, and the validation below touches every element
anyway, so laziness would buy nothing. Memory mapping also pins the file open on
Windows, which would make replacing an artifact root impossible while the server
runs. See ``docs/ARTIFACT_FORMAT.md``.

``allow_pickle=False`` everywhere: a ``.npy`` file is data from disk, and object
arrays would make it executable.

What is checked, and why each one matters
-----------------------------------------
The arrays cannot be re-derived — the server has no model — so every property
the game relies on has to be asserted on the way in, or it is simply assumed:

- **shape and dtype** against the manifest, or the arrays belong to a different
  root than the vocabulary does;
- **finite similarities**, or a NaN silently sorts below everything;
- **similarities within [-1, 1]**, because that range is the published API
  contract for ``similarity`` (docs/API_SPEC.md), not merely a property of cosine;
- **the answer scoring exactly 1.0**, or the winning guess would not report a
  perfect score;
- **the answer ranking 1**, and **the ranks forming a permutation of 1..N**,
  which together are the strongest cheap statement that the file is intact.

No message in this module names the answer; failures are identified by
``artifact_id`` (see ``errors``).

``np.load`` is a third-party boundary
-------------------------------------
Its failures are *not* safe to pass along. ``numpy.lib._format_impl`` quotes the
file back at you when a header will not parse — ``Cannot parse header: {!r}``,
``Header is not a dictionary: {!r}``, ``Header does not contain the correct
keys: {!r}``, ``descr is not a valid dtype descriptor: {!r}`` — so whatever
happens to sit in a malformed ``.npy`` lands in the message verbatim. Hash-only
paths keep an answer out of the *filename*; they say nothing about a file's
*contents*. A root carrying an answer inside a broken array would therefore
reach the log twice over: through an interpolated ``{exc}``, and through the
rendered ``__cause__`` once ``app.main`` logs the traceback.

So this boundary is treated exactly like the FastText one in
``embedding.fasttext_service``: the message keeps only what an operator needs —
which artifact, and what kind of failure — and ``from None`` drops numpy's chain
rather than leaving it to be rendered. The catch is broad for the same reason it
is there at all: the failure modes are numpy's to choose, and it already raises
at least one type outside ``(OSError, ValueError, EOFError)``
(``tokenize.TokenError``, from a header the tokenizer cannot even finish
reading), which would otherwise escape unsanitized.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.services.scoring.artifact.errors import ArtifactError
from app.services.scoring.artifact.manifest import AnswerEntry, ArtifactManifest
from app.services.scoring.artifact.paths import rank_path, similarity_path
from app.services.scoring.artifact.vocabulary import CanonicalVocabulary

# The similarity stored for the answer itself. The producer assigns a literal
# 1.0 before casting, and 1.0 is exactly representable in every supported
# similarity dtype (float32 and float16 alike), so this is compared exactly —
# a tolerance here would only widen what counts as a valid artifact.
ANSWER_SIMILARITY = 1.0

# The rank stored for the answer itself, by construction rather than by score.
ANSWER_RANK = 1


@dataclass(frozen=True, slots=True)
class AnswerArtifact:
    """One answer's similarity and rank over the whole canonical vocabulary."""

    artifact_id: str
    vocab_index: int
    similarities: np.ndarray
    ranks: np.ndarray

    @property
    def size(self) -> int:
        return int(self.similarities.shape[0])

    def score_at(self, index: int) -> tuple[float, int]:
        """Return ``(similarity, rank)`` for a known-valid vocabulary index."""
        return float(self.similarities[index]), int(self.ranks[index])

    def lookup(self, word: str, vocabulary: CanonicalVocabulary) -> tuple[float, int] | None:
        """Return ``(similarity, rank)`` for an already-normalized ``word``.

        ``None`` means the word is outside the canonical vocabulary. That is a
        fact about this root, not yet a decision about what the API should do
        with it — the guess-rule consequence is wired up separately.
        """
        index = vocabulary.index_of(word)
        if index is None:
            return None
        return self.score_at(index)


def load_answer(manifest: ArtifactManifest, entry: AnswerEntry) -> AnswerArtifact:
    """Load and validate the arrays for one answer of a validated root.

    Raises:
        ArtifactError: either file is missing or unreadable, or the arrays
            contradict the manifest or the contract.
    """
    similarities = _load_array(manifest.root, similarity_path(entry.artifact_id), entry)
    ranks = _load_array(manifest.root, rank_path(entry.artifact_id), entry)

    size = manifest.vocabulary.size
    _validate_array(similarities, "similarity", entry, size, manifest.similarity_dtype)
    _validate_array(ranks, "rank", entry, size, manifest.rank_dtype)
    _validate_similarities(similarities, entry)
    _validate_ranks(ranks, entry, size)

    return AnswerArtifact(
        artifact_id=entry.artifact_id,
        vocab_index=entry.vocab_index,
        similarities=similarities,
        ranks=ranks,
    )


def _load_array(root: Path, relative: str, entry: AnswerEntry) -> np.ndarray:
    try:
        loaded = np.load(root / relative, allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - third-party boundary; see above.
        raise ArtifactError(
            f"Could not load artifact {entry.artifact_id} ({type(exc).__name__})."
        ) from None
    if not isinstance(loaded, np.ndarray):
        raise ArtifactError(
            f"Artifact {entry.artifact_id} does not contain a plain array."
        )
    return loaded


def _validate_array(
    array: np.ndarray,
    kind: str,
    entry: AnswerEntry,
    size: int,
    expected_dtype: str,
) -> None:
    if array.shape != (size,):
        raise ArtifactError(
            f"Artifact {entry.artifact_id} {kind} array has shape {array.shape}, "
            f"expected ({size},) to match the canonical vocabulary."
        )
    if array.dtype.name != expected_dtype:
        raise ArtifactError(
            f"Artifact {entry.artifact_id} {kind} array is {array.dtype.name}, "
            f"but the manifest declares {expected_dtype}."
        )


def _validate_similarities(similarities: np.ndarray, entry: AnswerEntry) -> None:
    if not np.isfinite(similarities).all():
        raise ArtifactError(
            f"Artifact {entry.artifact_id} similarity array contains NaN or infinity."
        )
    if not bool(((similarities >= -1.0) & (similarities <= 1.0)).all()):
        raise ArtifactError(
            f"Artifact {entry.artifact_id} similarity array leaves the documented "
            "cosine range [-1.0, 1.0]."
        )
    if float(similarities[entry.vocab_index]) != ANSWER_SIMILARITY:
        raise ArtifactError(
            f"Artifact {entry.artifact_id} does not score its own answer "
            f"{ANSWER_SIMILARITY}; the winning guess would not report a perfect score."
        )


def _validate_ranks(ranks: np.ndarray, entry: AnswerEntry, size: int) -> None:
    if int(ranks[entry.vocab_index]) != ANSWER_RANK:
        raise ArtifactError(
            f"Artifact {entry.artifact_id} does not rank its own answer {ANSWER_RANK}."
        )
    expected = np.arange(1, size + 1, dtype=ranks.dtype)
    if not np.array_equal(np.sort(ranks), expected):
        raise ArtifactError(
            f"Artifact {entry.artifact_id} ranks are not a permutation of 1..{size}."
        )
