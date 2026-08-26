"""What one answer's arrays have to prove before a guess is scored from them.

The server has no model, so it cannot re-derive these numbers or notice that
they drifted. Every property the game depends on is therefore asserted on the
way in — and every one of those assertions is a test here, driven by breaking a
valid root from `artifact_fixture` one element at a time.

The manifest is loaded first in every case, so these tests also pin the division
of labour: `load_manifest` says nothing about array *contents*, and `load_answer`
says nothing about the root's *shape*.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from app.services.scoring.artifact import (
    AnswerEntry,
    ArtifactError,
    ArtifactManifest,
    load_answer,
    load_manifest,
)
from tests import artifact_fixture as fixture

ANSWER = fixture.ANSWER
ANSWER_INDEX = fixture.VOCABULARY.index(ANSWER)
SIZE = len(fixture.VOCABULARY)


@pytest.fixture
def root(tmp_path: Path) -> Path:
    target = tmp_path / "artifacts-root"
    fixture.write_root(target)
    return target


@pytest.fixture
def manifest(root: Path) -> ArtifactManifest:
    return load_manifest(root)


@pytest.fixture
def entry(manifest: ArtifactManifest) -> AnswerEntry:
    resolved = manifest.entry_for(ANSWER)
    assert resolved is not None
    return resolved


def _forbidden_forms(word: str) -> tuple[str, ...]:
    return (word, repr(word), word.encode("unicode_escape").decode())


def _reject(manifest: ArtifactManifest, entry: AnswerEntry, match: str) -> ArtifactError:
    with pytest.raises(ArtifactError, match=match) as caught:
        load_answer(manifest, entry)
    return caught.value


# --- A valid answer ----------------------------------------------------------


def test_a_valid_answer_loads(manifest: ArtifactManifest, entry: AnswerEntry) -> None:
    artifact = load_answer(manifest, entry)

    assert artifact.artifact_id == entry.artifact_id
    assert artifact.size == SIZE
    assert artifact.similarities.dtype.name == "float32"
    assert artifact.ranks.dtype.name == "uint16"


def test_the_answer_scores_one_and_ranks_one(
    manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    similarity, rank = load_answer(manifest, entry).score_at(entry.vocab_index)

    assert similarity == 1.0
    assert rank == 1


def test_every_vocabulary_word_is_scored(
    manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    artifact = load_answer(manifest, entry)

    scores = [artifact.lookup(word, manifest.vocabulary) for word in fixture.VOCABULARY]

    assert all(score is not None for score in scores)
    assert sorted(rank for _, rank in scores) == list(range(1, SIZE + 1))  # type: ignore[misc]


def test_ranks_follow_descending_similarity(
    manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    """The fixture writes the documented policy; this reads it back."""
    artifact = load_answer(manifest, entry)
    scores = {word: artifact.lookup(word, manifest.vocabulary) for word in fixture.VOCABULARY}
    assert all(score is not None for score in scores.values())

    by_rank = sorted(fixture.VOCABULARY, key=lambda word: scores[word][1])  # type: ignore[index]

    assert by_rank[0] == ANSWER
    similarities = [scores[word][0] for word in by_rank]  # type: ignore[index]
    assert similarities == sorted(similarities, reverse=True)


def test_a_word_outside_the_vocabulary_has_no_score(
    manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    """A fact about this root — not yet a decision about the API."""
    assert load_answer(manifest, entry).lookup("어휘밖단어", manifest.vocabulary) is None


def test_float16_roots_load_too(tmp_path: Path) -> None:
    target = tmp_path / "half"
    fixture.write_root(target, similarity_dtype="float16")
    half_manifest = load_manifest(target)
    half_entry = half_manifest.entry_for(ANSWER)
    assert half_entry is not None

    artifact = load_answer(half_manifest, half_entry)

    assert artifact.similarities.dtype.name == "float16"
    # 1.0 is exactly representable in float16, which is why the answer check
    # needs no tolerance.
    assert artifact.score_at(half_entry.vocab_index) == (1.0, 1)


# --- Files -------------------------------------------------------------------


@pytest.mark.parametrize("filename", ["similarity.npy", "rank.npy"])
def test_a_file_removed_after_startup_is_reported(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry, filename: str
) -> None:
    """Existence was checked at startup; the root can still change underneath."""
    (root / fixture.artifact_directory(ANSWER) / filename).unlink()

    _reject(manifest, entry, "Could not load artifact")


@pytest.mark.parametrize("filename", ["similarity.npy", "rank.npy"])
def test_a_truncated_array_is_reported(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry, filename: str
) -> None:
    (root / fixture.artifact_directory(ANSWER) / filename).write_bytes(b"\x93NUMPY broken")

    _reject(manifest, entry, "Could not load artifact")


def test_a_pickled_array_is_refused(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    """`allow_pickle=False`: a `.npy` is data from disk, never code."""
    path = root / fixture.artifact_directory(ANSWER) / "similarity.npy"
    np.save(path, np.array([{"payload": 1}], dtype=object), allow_pickle=True)

    _reject(manifest, entry, "Could not load artifact")


# --- Shape and dtype ---------------------------------------------------------


def test_a_short_similarity_array_is_rejected(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    fixture.overwrite_similarity(root, ANSWER, np.zeros(SIZE - 1, dtype="float32"))

    _reject(manifest, entry, "similarity array has shape")


def test_a_two_dimensional_array_is_rejected(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    fixture.overwrite_similarity(root, ANSWER, np.zeros((SIZE, 2), dtype="float32"))

    _reject(manifest, entry, "similarity array has shape")


def test_a_short_rank_array_is_rejected(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    fixture.overwrite_rank(root, ANSWER, np.arange(1, SIZE, dtype="uint16"))

    _reject(manifest, entry, "rank array has shape")


def test_a_similarity_dtype_that_contradicts_the_manifest_is_rejected(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    values = fixture.read_similarity(root, ANSWER).astype("float16")
    fixture.overwrite_similarity(root, ANSWER, values)

    _reject(manifest, entry, "similarity array is float16")


def test_a_rank_dtype_that_contradicts_the_manifest_is_rejected(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    values = fixture.read_rank(root, ANSWER).astype("uint32")
    fixture.overwrite_rank(root, ANSWER, values)

    _reject(manifest, entry, "rank array is uint32")


# --- Similarity values -------------------------------------------------------


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_a_non_finite_similarity_is_rejected(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry, value: float
) -> None:
    """A NaN would sort below everything without ever looking wrong."""
    values = fixture.read_similarity(root, ANSWER)
    values[1] = value
    fixture.overwrite_similarity(root, ANSWER, values)

    _reject(manifest, entry, "NaN or infinity")


@pytest.mark.parametrize("value", [5.0, 1.5, -1.5, -3.0])
def test_a_similarity_outside_the_cosine_range_is_rejected(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry, value: float
) -> None:
    """[-1.0, 1.0] is the published API contract, not just a cosine property."""
    values = fixture.read_similarity(root, ANSWER)
    values[1] = value
    fixture.overwrite_similarity(root, ANSWER, values)

    _reject(manifest, entry, r"cosine range \[-1.0, 1.0\]")


@pytest.mark.parametrize("value", [1.0, -1.0])
def test_the_range_boundaries_are_accepted(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry, value: float
) -> None:
    """Inclusive: a synonym really can score 1.0, and an antonym -1.0."""
    values = fixture.read_similarity(root, ANSWER)
    values[1] = value
    fixture.overwrite_similarity(root, ANSWER, values)

    assert load_answer(manifest, entry).score_at(1)[0] == value


@pytest.mark.parametrize("value", [0.0, 0.9999, -1.0])
def test_an_answer_that_does_not_score_one_is_rejected(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry, value: float
) -> None:
    """Otherwise the winning guess would not report a perfect score."""
    values = fixture.read_similarity(root, ANSWER)
    values[ANSWER_INDEX] = value
    fixture.overwrite_similarity(root, ANSWER, values)

    _reject(manifest, entry, "does not score its own answer")


# --- Rank values -------------------------------------------------------------


def test_an_answer_that_does_not_rank_one_is_rejected(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    values = fixture.read_rank(root, ANSWER)
    values[ANSWER_INDEX] = 2
    values[1] = 1
    fixture.overwrite_rank(root, ANSWER, values)

    _reject(manifest, entry, "does not rank its own answer")


def test_duplicated_ranks_are_rejected(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    """Ranks are dense and unique; a repeat means the file is damaged."""
    values = fixture.read_rank(root, ANSWER)
    values[2] = values[3]
    fixture.overwrite_rank(root, ANSWER, values)

    _reject(manifest, entry, "not a permutation")


def test_a_zero_rank_is_rejected(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    """Ranks start at 1. The answer still ranks 1 here, so only the permutation
    check can catch an off-by-one producer — which is the point."""
    values = fixture.read_rank(root, ANSWER)
    values[1] = 0
    fixture.overwrite_rank(root, ANSWER, values)

    _reject(manifest, entry, "not a permutation")


def test_a_fully_zero_based_rank_array_is_rejected(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    """Shifting every rank down also moves the answer off 1, which fails first."""
    fixture.overwrite_rank(root, ANSWER, np.arange(0, SIZE, dtype="uint16"))

    _reject(manifest, entry, "does not rank its own answer")


def test_out_of_range_ranks_are_rejected(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    values = fixture.read_rank(root, ANSWER)
    values[2] = SIZE + 100
    fixture.overwrite_rank(root, ANSWER, values)

    _reject(manifest, entry, "not a permutation")


# --- Secrecy -----------------------------------------------------------------


@pytest.mark.parametrize(
    "break_arrays",
    [
        pytest.param(
            lambda root: (root / fixture.artifact_directory(ANSWER) / "rank.npy").unlink(),
            id="missing-file",
        ),
        pytest.param(
            lambda root: (root / fixture.artifact_directory(ANSWER) / "similarity.npy").write_bytes(
                b"\x93NUMPY broken"
            ),
            id="corrupt-file",
        ),
        pytest.param(
            lambda root: fixture.overwrite_similarity(
                root, ANSWER, np.full(SIZE, 5.0, dtype="float32")
            ),
            id="out-of-range-similarity",
        ),
        pytest.param(
            lambda root: fixture.overwrite_similarity(
                root, ANSWER, np.zeros(SIZE, dtype="float32")
            ),
            id="answer-not-one",
        ),
        pytest.param(
            lambda root: fixture.overwrite_rank(root, ANSWER, np.zeros(SIZE, dtype="uint16")),
            id="broken-ranks",
        ),
        pytest.param(
            lambda root: fixture.overwrite_similarity(
                root, ANSWER, np.zeros(SIZE - 1, dtype="float32")
            ),
            id="wrong-shape",
        ),
    ],
)
def test_no_array_failure_names_the_answer(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry, break_arrays
) -> None:
    """These failures happen mid-game, so their traceback reaches the log."""
    break_arrays(root)

    with pytest.raises(ArtifactError) as caught:
        load_answer(manifest, entry)

    rendered = f"{caught.value}{caught.value.__cause__ or ''}"
    for form in _forbidden_forms(ANSWER):
        assert form not in rendered


def test_failures_identify_the_artifact_by_id(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    """Redaction must still leave an operator able to find the file."""
    fixture.overwrite_similarity(root, ANSWER, np.full(SIZE, 5.0, dtype="float32"))

    error = _reject(manifest, entry, "cosine range")

    assert entry.artifact_id in str(error)


# --- The `np.load` boundary --------------------------------------------------
#
# Hash-only paths keep an answer out of a *filename*. They say nothing about a
# file's *contents*, and numpy quotes contents: a malformed header comes back
# inside the exception message. These tests establish that premise against the
# installed numpy, then pin that the backend refuses to carry it.

# Headers numpy reads successfully enough to quote, each failing a later check.
# `{secret}` stands in for whatever the broken file happens to hold.
LEAKY_HEADERS = (
    pytest.param("{secret!r}", "Header is not a dictionary", id="not-a-dict"),
    pytest.param(
        "{{'descr': {secret!r}, 'fortran_order': False, 'shape': (6,)}}",
        "descr is not a valid dtype descriptor",
        id="bad-descr",
    ),
    pytest.param("{{{secret!r}: 1}}", "Header does not contain the correct keys", id="bad-keys"),
    pytest.param(
        "{{'descr': '<f4', 'fortran_order': False, 'shape': (6,), 'note': {secret!r}",
        "Cannot parse header",
        id="unparsable",
    ),
)


@pytest.mark.parametrize(("header", "numpy_message"), LEAKY_HEADERS)
def test_numpy_itself_echoes_the_file_into_its_message(
    root: Path, header: str, numpy_message: str
) -> None:
    """The premise the sanitising exists for. If this ever fails, re-read it."""
    path = fixture.write_npy_with_header(
        root, ANSWER, "similarity.npy", header.format(secret=ANSWER)
    )

    with pytest.raises(Exception) as caught:  # noqa: PT011 - numpy chooses the type.
        np.load(path, allow_pickle=False)

    assert numpy_message in str(caught.value)
    assert ANSWER in str(caught.value)


@pytest.mark.parametrize(("header", "numpy_message"), LEAKY_HEADERS)
def test_a_leaky_numpy_message_is_not_carried_into_the_error(
    root: Path,
    manifest: ArtifactManifest,
    entry: AnswerEntry,
    header: str,
    numpy_message: str,
) -> None:
    fixture.write_npy_with_header(root, ANSWER, "similarity.npy", header.format(secret=ANSWER))

    with pytest.raises(ArtifactError) as caught:
        load_answer(manifest, entry)

    error = caught.value
    assert numpy_message not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None or error.__suppress_context__
    for form in _forbidden_forms(ANSWER):
        assert form not in str(error)


def test_a_sanitised_load_failure_still_says_which_artifact_and_what_kind(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    """Redaction must not cost an operator the ability to act on the failure."""
    fixture.write_npy_with_header(root, ANSWER, "similarity.npy", repr(ANSWER))

    error = _reject(manifest, entry, "Could not load artifact")

    assert entry.artifact_id in str(error)
    assert "ValueError" in str(error)


def test_a_header_numpy_cannot_tokenize_is_still_an_artifact_error(
    root: Path, manifest: ArtifactManifest, entry: AnswerEntry
) -> None:
    """`tokenize.TokenError` is not an `OSError`/`ValueError`/`EOFError`.

    numpy picks its own exception types, so the boundary catches broadly; a
    narrower catch let this one escape `load_answer` unsanitised.
    """
    fixture.write_npy_with_header(root, ANSWER, "similarity.npy", f"{ANSWER!r} + (")

    error = _reject(manifest, entry, "Could not load artifact")

    assert error.__cause__ is None
    for form in _forbidden_forms(ANSWER):
        assert form not in str(error)


def test_a_logged_traceback_of_a_load_failure_carries_no_answer(
    root: Path,
    manifest: ArtifactManifest,
    entry: AnswerEntry,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The whole point: what `app.main`'s INTERNAL_ERROR handler would write out.

    The reader is not wired into a request yet, so this reproduces that handler's
    one relevant act — `logger.exception` on the escaped error — and reads back
    the fully rendered record, traceback included.
    """
    fixture.write_npy_with_header(root, ANSWER, "similarity.npy", repr(ANSWER))
    logger = logging.getLogger("app.main")

    with caplog.at_level(logging.ERROR, logger="app.main"):
        try:
            load_answer(manifest, entry)
        except ArtifactError:
            logger.exception("Unhandled exception while handling POST /api/games/1/guesses.")

    rendered = caplog.text
    # Prove the failure path was taken and the traceback really was rendered,
    # so this cannot pass by capturing nothing.
    assert caplog.records, "nothing was logged"
    assert "ArtifactError: Could not load artifact" in rendered
    assert "Traceback (most recent call last)" in rendered
    assert entry.artifact_id in rendered

    assert "Header is not a dictionary" not in rendered
    assert "direct cause" not in rendered
    assert "During handling" not in rendered
    for form in _forbidden_forms(ANSWER):
        assert form not in rendered
