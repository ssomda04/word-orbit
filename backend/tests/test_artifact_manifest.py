"""What an artifact root has to prove before the server will serve from it.

The reader treats a root as untrusted input, so almost every test here writes a
valid root with `artifact_fixture` and then breaks exactly one thing. Two rules
apply throughout:

- **one mutation per test**, so a rejection identifies its cause;
- **no rejection may name the answer**, checked as a sweep at the bottom rather
  than repeated in every case.

`artifact_fixture` builds the root without touching the reader's own helpers, so
these are two independent implementations of one format agreeing — not one
implementation agreeing with itself.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.services.scoring.artifact import (
    ArtifactError,
    artifact_id_for,
    load_manifest,
    read_canonical_vocabulary,
)
from tests import artifact_fixture as fixture

ANSWER = fixture.ANSWER


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A complete, valid artifact root."""
    target = tmp_path / "artifacts-root"
    fixture.write_root(target)
    return target


def _forbidden_forms(word: str) -> tuple[str, ...]:
    """Every spelling the answer could survive as: raw, repr, and escaped."""
    return (word, repr(word), word.encode("unicode_escape").decode())


def _reject(root: Path, match: str) -> ArtifactError:
    with pytest.raises(ArtifactError, match=match) as caught:
        load_manifest(root)
    return caught.value


# --- A valid root ------------------------------------------------------------


def test_a_valid_root_loads(root: Path) -> None:
    manifest = load_manifest(root)

    assert manifest.schema_version == "1.0"
    assert manifest.similarity_dtype == "float32"
    assert manifest.rank_dtype == "uint16"
    assert manifest.vocabulary.size == len(fixture.VOCABULARY)
    assert manifest.vocabulary.words == fixture.VOCABULARY


def test_the_embedding_model_is_carried_through(root: Path) -> None:
    """Recorded for auditing: which model produced these numbers."""
    manifest = load_manifest(root)

    assert manifest.embedding_model_name == fixture.EMBEDDING_MODEL["name"]
    assert manifest.embedding_model_source == fixture.EMBEDDING_MODEL["source"]


def test_answers_are_exposed_for_selection(root: Path) -> None:
    manifest = load_manifest(root)

    assert manifest.answers == (ANSWER,)
    assert manifest.answer_count == 1


def test_a_known_answer_resolves_to_its_entry(root: Path) -> None:
    manifest = load_manifest(root)

    entry = manifest.entry_for(ANSWER)

    assert entry is not None
    assert entry.artifact_id == artifact_id_for(ANSWER)
    assert entry.vocab_index == fixture.VOCABULARY.index(ANSWER)


def test_an_unknown_answer_resolves_to_none_rather_than_raising(root: Path) -> None:
    """`None` cannot carry a word; an exception message could."""
    assert load_manifest(root).entry_for("없는정답") is None


def test_vocabulary_indexes_are_exposed(root: Path) -> None:
    manifest = load_manifest(root)

    assert manifest.index_of("학생") == fixture.VOCABULARY.index("학생")
    assert manifest.index_of("어휘밖") is None


def test_several_answers_are_supported(tmp_path: Path) -> None:
    target = tmp_path / "multi"
    fixture.write_root(target, answers=("바다", "커피", "학생"))

    manifest = load_manifest(target)

    assert manifest.answer_count == 3
    assert set(manifest.answers) == {"바다", "커피", "학생"}


# --- The manifest document ---------------------------------------------------


def test_a_missing_manifest_is_rejected(root: Path) -> None:
    (root / "manifest.json").unlink()

    _reject(root, "Could not read")


def test_invalid_json_is_rejected(root: Path) -> None:
    (root / "manifest.json").write_text("{not json", encoding="utf-8")

    _reject(root, "not valid JSON")


def test_a_manifest_that_is_not_an_object_is_rejected(root: Path) -> None:
    (root / "manifest.json").write_text("[]", encoding="utf-8")

    _reject(root, "must be a JSON object")


def test_a_non_utf8_manifest_is_rejected(root: Path) -> None:
    (root / "manifest.json").write_bytes(b"\xff\xfe{}")

    _reject(root, "not valid UTF-8")


# --- Schema and policy -------------------------------------------------------


@pytest.mark.parametrize("version", ["1.1", "2.0", "1.0.0-prototype", None, 1.0])
def test_an_unsupported_schema_version_is_rejected(root: Path, version: Any) -> None:
    """A new layout may reorder or reinterpret the arrays; refuse to guess."""
    fixture.mutate_manifest(root, lambda m: m.update(schema_version=version))

    _reject(root, "Unsupported artifact schema_version")


def test_an_unsupported_artifact_id_algorithm_is_rejected(root: Path) -> None:
    fixture.mutate_manifest(root, lambda m: m.update(artifact_id_algorithm="md5"))

    _reject(root, "Unsupported artifact_id_algorithm")


@pytest.mark.parametrize(
    "field,value",
    [
        ("metric", "euclidean"),
        ("answer_rank", 0),
        ("order", "similarity_asc"),
        ("tie_break", "insertion"),
    ],
)
def test_a_changed_ranking_policy_field_is_rejected(root: Path, field: str, value: Any) -> None:
    """Ranks mean something different under a different policy."""
    fixture.mutate_manifest(root, lambda m: m["ranking_policy"].update({field: value}))

    _reject(root, "ranking_policy")


def test_an_extended_ranking_policy_is_rejected(root: Path) -> None:
    """Exact equality: an added rule is still a rule this server does not know."""
    fixture.mutate_manifest(root, lambda m: m["ranking_policy"].update(secondary="length"))

    _reject(root, "ranking_policy")


def test_a_missing_ranking_policy_is_rejected(root: Path) -> None:
    fixture.mutate_manifest(root, lambda m: m.pop("ranking_policy"))

    _reject(root, "ranking_policy")


# --- Embedding model ---------------------------------------------------------


def test_a_non_object_embedding_model_is_rejected(root: Path) -> None:
    fixture.mutate_manifest(root, lambda m: m.update(embedding_model="fasttext"))

    _reject(root, "embedding_model must be an object")


@pytest.mark.parametrize("field", ["name", "source"])
@pytest.mark.parametrize("value", ["", "   ", None, 7])
def test_a_blank_or_missing_embedding_field_is_rejected(
    root: Path, field: str, value: Any
) -> None:
    fixture.mutate_manifest(root, lambda m: m["embedding_model"].update({field: value}))

    _reject(root, f"embedding_model.{field}")


# --- Vocabulary metadata -----------------------------------------------------


def test_a_vocabulary_path_outside_the_root_is_rejected(root: Path) -> None:
    """The vocabulary always travels inside the root; it is never a pointer."""
    fixture.mutate_manifest(root, lambda m: m["vocabulary"].update(path="/etc/words.txt"))

    _reject(root, "vocabulary.path")


def test_a_missing_vocabulary_file_is_rejected(root: Path) -> None:
    (root / "vocabulary.txt").unlink()

    _reject(root, "Could not read the artifact vocabulary")


def test_a_vocabulary_size_mismatch_is_rejected(root: Path) -> None:
    fixture.mutate_manifest(root, lambda m: m["vocabulary"].update(size=999))

    _reject(root, "vocabulary.size")


def test_a_vocabulary_hash_mismatch_is_rejected(root: Path) -> None:
    """The strongest single statement that the arrays match the word list."""
    fixture.mutate_manifest(root, lambda m: m["vocabulary"].update(sha256="0" * 64))

    _reject(root, "vocabulary.sha256")


def test_editing_the_vocabulary_breaks_the_recorded_hash(root: Path) -> None:
    """Not a separate rule — the hash catches it, which is the point."""
    (root / "vocabulary.txt").write_bytes(fixture.vocabulary_bytes(("가", "나", "다")))

    _reject(root, "vocabulary")


# --- The vocabulary file itself ----------------------------------------------


def _write_vocabulary(tmp_path: Path, payload: bytes) -> Path:
    path = tmp_path / "vocabulary.txt"
    path.write_bytes(payload)
    return path


def test_a_canonical_vocabulary_is_accepted(tmp_path: Path) -> None:
    path = _write_vocabulary(tmp_path, fixture.vocabulary_bytes(fixture.VOCABULARY))

    vocabulary = read_canonical_vocabulary(path)

    assert vocabulary.words == fixture.VOCABULARY
    assert vocabulary.size == len(fixture.VOCABULARY)
    assert vocabulary.word_at(0) == fixture.VOCABULARY[0]
    assert vocabulary.word_at(99) is None


def test_a_byte_order_mark_is_rejected(tmp_path: Path) -> None:
    """A BOM survives NFKC and strip, so it would merge into the first word."""
    path = _write_vocabulary(
        tmp_path, b"\xef\xbb\xbf" + fixture.vocabulary_bytes(fixture.VOCABULARY)
    )

    with pytest.raises(ArtifactError, match="byte-order mark"):
        read_canonical_vocabulary(path)


def test_a_non_utf8_vocabulary_is_rejected(tmp_path: Path) -> None:
    path = _write_vocabulary(tmp_path, b"\xff\xfe\x00")

    with pytest.raises(ArtifactError, match="not valid UTF-8"):
        read_canonical_vocabulary(path)


def test_an_empty_vocabulary_is_rejected(tmp_path: Path) -> None:
    path = _write_vocabulary(tmp_path, b"")

    with pytest.raises(ArtifactError, match="no usable words"):
        read_canonical_vocabulary(path)


@pytest.mark.parametrize(
    "payload,reason",
    [
        (b"\xea\xb0\x80\n\n\xeb\x82\x98\n", "blank line"),
        (b"\xea\xb0\x80\n\xea\xb0\x80\n", "duplicate"),
        (b"\xea\xb0\x80\n\xeb\x82\x98", "no trailing newline"),
        (b"\xea\xb0\x80\r\n\xeb\x82\x98\r\n", "CRLF"),
        (b" \xea\xb0\x80\n\xeb\x82\x98\n", "leading whitespace"),
        (b"\xea\xb0\x80 \n\xeb\x82\x98\n", "trailing whitespace"),
        ("Ａ\n나\n".encode(), "un-normalized fullwidth"),
    ],
)
def test_a_non_canonical_vocabulary_is_rejected(
    tmp_path: Path, payload: bytes, reason: str
) -> None:
    """Anything a forgiving reader would silently repair is a silent mismatch."""
    path = _write_vocabulary(tmp_path, payload)

    with pytest.raises(ArtifactError, match="canonical form"):
        read_canonical_vocabulary(path)


# --- Dtypes ------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float64", "float", "int32", "", None])
def test_an_unsupported_similarity_dtype_is_rejected(root: Path, dtype: Any) -> None:
    fixture.mutate_manifest(root, lambda m: m.update(similarity_dtype=dtype))

    _reject(root, "Unsupported similarity_dtype")


@pytest.mark.parametrize("dtype", ["uint8", "int16", "float32", None])
def test_an_unsupported_rank_dtype_is_rejected(root: Path, dtype: Any) -> None:
    """uint8 cannot address a real game vocabulary; int16 is not a rank type."""
    fixture.mutate_manifest(root, lambda m: m.update(rank_dtype=dtype))

    _reject(root, "Unsupported rank_dtype")


def test_a_rank_dtype_too_narrow_for_the_vocabulary_is_rejected(tmp_path: Path) -> None:
    """uint16 reaches 65535, so one more word than that must fail loudly."""
    target = tmp_path / "wide"
    words = tuple(f"w{index}" for index in range(np.iinfo(np.uint16).max + 1))
    fixture.write_root(target, vocabulary=words, answers=("w0",), rank_dtype="uint32")
    fixture.mutate_manifest(target, lambda m: m.update(rank_dtype="uint16"))

    _reject(target, "reaches 65535")


def test_a_wide_vocabulary_is_accepted_with_a_wide_rank_dtype(tmp_path: Path) -> None:
    target = tmp_path / "wide-ok"
    words = tuple(f"w{index}" for index in range(np.iinfo(np.uint16).max + 1))
    fixture.write_root(target, vocabulary=words, answers=("w0",), rank_dtype="uint32")

    manifest = load_manifest(target)

    assert manifest.rank_dtype == "uint32"
    assert manifest.vocabulary.size == np.iinfo(np.uint16).max + 1


def test_float16_similarity_is_accepted(tmp_path: Path) -> None:
    """The writer supports it, so the reader must not narrow the contract."""
    target = tmp_path / "half"
    fixture.write_root(target, similarity_dtype="float16")

    assert load_manifest(target).similarity_dtype == "float16"


# --- Answers -----------------------------------------------------------------


def test_a_non_object_answers_field_is_rejected(root: Path) -> None:
    fixture.mutate_manifest(root, lambda m: m.update(answers=[ANSWER]))

    _reject(root, "answers must be an object")


def test_an_empty_answers_object_is_rejected(root: Path) -> None:
    """A root that can serve nothing fails at startup, not at the first game."""
    fixture.mutate_manifest(root, lambda m: m.update(answers={}))

    _reject(root, "answers must not be empty")


def test_an_answer_outside_the_vocabulary_is_rejected(root: Path) -> None:
    def move(manifest: dict[str, Any]) -> None:
        manifest["answers"] = {"어휘밖정답": manifest["answers"][ANSWER]}

    fixture.mutate_manifest(root, move)

    _reject(root, "canonical vocabulary word")


def test_an_un_normalized_answer_key_is_rejected(root: Path) -> None:
    """A key that a normalized guess can never equal is an unreachable answer."""

    def move(manifest: dict[str, Any]) -> None:
        manifest["answers"] = {f" {ANSWER} ": manifest["answers"][ANSWER]}

    fixture.mutate_manifest(root, move)

    _reject(root, "canonical vocabulary word")


def test_a_non_object_answer_entry_is_rejected(root: Path) -> None:
    fixture.mutate_manifest(root, lambda m: m["answers"].update({ANSWER: "path"}))

    _reject(root, "must be an object")


def test_a_wrong_artifact_id_is_rejected(root: Path) -> None:
    fixture.mutate_manifest(root, lambda m: m["answers"][ANSWER].update(artifact_id="0" * 64))

    _reject(root, "artifact_id that is not the sha256")


def test_a_wrong_answer_vocab_index_is_rejected(root: Path) -> None:
    fixture.mutate_manifest(root, lambda m: m["answers"][ANSWER].update(answer_vocab_index=3))

    _reject(root, "answer_vocab_index 3")


@pytest.mark.parametrize("value", ["0", None, 1.0, True])
def test_a_non_integer_answer_vocab_index_is_rejected(root: Path, value: Any) -> None:
    """`True` is an `int` in Python; JSON `true` must not pass as index 1."""
    fixture.mutate_manifest(
        root, lambda m: m["answers"][ANSWER].update(answer_vocab_index=value)
    )

    _reject(root, "non-integer answer_vocab_index")


@pytest.mark.parametrize("key", ["similarity_path", "rank_path"])
@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "C:/Windows/system32/config",
        "../../outside.npy",
        "artifacts/../outside.npy",
        "artifacts/xx/yy/similarity.npy",
        "",
    ],
)
def test_a_path_that_is_not_the_canonical_one_is_rejected(
    root: Path, key: str, path: str
) -> None:
    """Paths are recomputed and compared, so traversal needs no rule of its own."""
    fixture.mutate_manifest(root, lambda m: m["answers"][ANSWER].update({key: path}))

    _reject(root, "canonical hash-only path")


@pytest.mark.parametrize("filename", ["similarity.npy", "rank.npy"])
def test_a_missing_referenced_file_is_rejected_at_load(root: Path, filename: str) -> None:
    """Existence is checked at startup; contents are not read until needed."""
    (root / fixture.artifact_directory(ANSWER) / filename).unlink()

    _reject(root, "missing file")


def test_arrays_are_not_read_while_validating_the_manifest(root: Path, tmp_path: Path) -> None:
    """A root of thousands of answers must not be read into memory at startup.

    Truncating the arrays to nothing leaves them present but unreadable; the
    manifest still validates, which is only possible if nothing opened them.
    """
    for filename in ("similarity.npy", "rank.npy"):
        (root / fixture.artifact_directory(ANSWER) / filename).write_bytes(b"")

    assert load_manifest(root).answer_count == 1


# --- Secrecy -----------------------------------------------------------------


@pytest.mark.parametrize(
    "break_root",
    [
        pytest.param(
            lambda root: fixture.mutate_manifest(
                root, lambda m: m.update(answers={"어휘밖정답": m["answers"][ANSWER]})
            ),
            id="answer-outside-vocabulary",
        ),
        pytest.param(
            lambda root: fixture.mutate_manifest(
                root, lambda m: m["answers"][ANSWER].update(artifact_id="0" * 64)
            ),
            id="wrong-artifact-id",
        ),
        pytest.param(
            lambda root: fixture.mutate_manifest(
                root, lambda m: m["answers"][ANSWER].update(answer_vocab_index=3)
            ),
            id="wrong-vocab-index",
        ),
        pytest.param(
            lambda root: fixture.mutate_manifest(
                root, lambda m: m["answers"][ANSWER].update(similarity_path="/etc/passwd")
            ),
            id="non-canonical-path",
        ),
        pytest.param(
            lambda root: (root / fixture.artifact_directory(ANSWER) / "rank.npy").unlink(),
            id="missing-file",
        ),
        pytest.param(
            lambda root: fixture.mutate_manifest(
                root, lambda m: m["vocabulary"].update(sha256="0" * 64)
            ),
            id="vocabulary-hash-mismatch",
        ),
    ],
)
def test_no_rejection_names_the_answer(root: Path, break_root: Any) -> None:
    """The manifest holds answers in plain text; no failure may repeat one."""
    break_root(root)

    with pytest.raises(ArtifactError) as caught:
        load_manifest(root)

    rendered = "".join(
        str(part) for part in (caught.value, caught.value.__cause__ or "")
    )
    for form in _forbidden_forms(ANSWER):
        assert form not in rendered


def test_the_manifest_body_is_not_echoed_by_a_parse_error(root: Path) -> None:
    """A JSON error reports a position, never the document — including answers."""
    document = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    (root / "manifest.json").write_text(
        json.dumps(document, ensure_ascii=False)[:-1], encoding="utf-8"
    )

    with pytest.raises(ArtifactError) as caught:
        load_manifest(root)

    for form in _forbidden_forms(ANSWER):
        assert form not in str(caught.value)
