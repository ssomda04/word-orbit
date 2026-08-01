"""Unit tests for `FastTextEmbeddingService` — no model file, no `fasttext` install.

The service takes an already-loaded model object, so a tiny fake covers the whole
contract. These tests run in CI and in a default `uv sync` environment.
"""

import math
import sys
from pathlib import Path

import pytest

from app.services.embedding import (
    EmbeddingService,
    FastTextConfigurationError,
    FastTextEmbeddingService,
)

# Deliberately 3-dimensional so expected cosines are obvious by hand.
VECTORS: dict[str, list[float]] = {
    "학생": [1.0, 0.0, 0.0],
    "선생": [0.0, 1.0, 0.0],  # orthogonal to 학생 -> 0.0
    "반대": [-1.0, 0.0, 0.0],  # opposite of 학생 -> -1.0
    "같은방향": [2.0, 0.0, 0.0],  # parallel to 학생 -> 1.0 (magnitude must not matter)
    "영벡터": [0.0, 0.0, 0.0],
    "빈벡터": [],
    "난수": [float("nan"), 0.0, 0.0],
    "무한": [float("inf"), 0.0, 0.0],
    "AB": [0.0, 0.0, 1.0],
}

# What the real model returns for a word it composes from character n-grams.
SUBWORD_VECTOR = [0.5, 0.5, 0.5]


class FakeFastTextModel:
    """Stand-in exposing only the one method the service uses."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self._vectors = VECTORS if vectors is None else vectors
        self.calls: list[str] = []

    def get_word_vector(self, word: str) -> list[float]:
        self.calls.append(word)
        # FastText always answers: an unknown word is composed from subwords.
        return list(self._vectors.get(word, SUBWORD_VECTOR))


class ExplodingFastTextModel:
    """Model whose native call fails, to check the error is wrapped."""

    def get_word_vector(self, word: str) -> list[float]:
        raise RuntimeError("native failure")


@pytest.fixture
def service() -> FastTextEmbeddingService:
    return FastTextEmbeddingService(FakeFastTextModel())


# --- Protocol conformance --------------------------------------------------


def test_satisfies_the_embedding_service_protocol(service: FastTextEmbeddingService) -> None:
    assert isinstance(service, EmbeddingService)


# --- encode ----------------------------------------------------------------


def test_encode_returns_a_list_of_floats(service: FastTextEmbeddingService) -> None:
    vector = service.encode("학생")

    assert vector == [1.0, 0.0, 0.0]
    assert all(isinstance(value, float) for value in vector)


def test_encode_is_deterministic(service: FastTextEmbeddingService) -> None:
    """The Protocol requires stable output for a given input."""
    assert service.encode("학생") == service.encode("학생")


def test_encode_returns_a_zero_vector_unchanged(service: FastTextEmbeddingService) -> None:
    """Mirrors DeterministicEmbeddingService: no normalization of a zero vector."""
    assert service.encode("영벡터") == [0.0, 0.0, 0.0]


def test_encode_allows_out_of_vocabulary_words(service: FastTextEmbeddingService) -> None:
    """FastText composes subwords, so an unknown word is a normal input."""
    vector = service.encode("존재하지않는긴단어")

    assert vector == SUBWORD_VECTOR
    assert all(math.isfinite(value) for value in vector)


# --- encode_many -----------------------------------------------------------


def test_encode_many_preserves_order(service: FastTextEmbeddingService) -> None:
    words = ["학생", "선생", "반대"]

    assert service.encode_many(words) == [service.encode(word) for word in words]


def test_encode_many_handles_an_empty_sequence(service: FastTextEmbeddingService) -> None:
    assert service.encode_many([]) == []


# --- similarity ------------------------------------------------------------


def test_identical_words_score_one(service: FastTextEmbeddingService) -> None:
    """The game asserts similarity == 1.0 for a winning guess."""
    assert service.similarity("학생", "학생") == pytest.approx(1.0)


def test_opposite_vectors_score_minus_one(service: FastTextEmbeddingService) -> None:
    assert service.similarity("학생", "반대") == pytest.approx(-1.0)


def test_orthogonal_vectors_score_zero(service: FastTextEmbeddingService) -> None:
    assert service.similarity("학생", "선생") == pytest.approx(0.0)


def test_similarity_ignores_magnitude(service: FastTextEmbeddingService) -> None:
    """Cosine, not dot product: a parallel vector of any length scores 1.0."""
    assert service.similarity("학생", "같은방향") == pytest.approx(1.0)


def test_similarity_is_symmetric(service: FastTextEmbeddingService) -> None:
    assert service.similarity("학생", "선생") == service.similarity("선생", "학생")


@pytest.mark.parametrize(
    ("first", "second"),
    [("학생", "선생"), ("학생", "반대"), ("학생", "학생"), ("선생", "존재하지않는단어")],
)
def test_similarity_stays_inside_the_documented_range(
    service: FastTextEmbeddingService, first: str, second: str
) -> None:
    assert -1.0 <= service.similarity(first, second) <= 1.0


# --- zero vectors ----------------------------------------------------------


def test_zero_vector_similarity_is_zero_not_an_error(service: FastTextEmbeddingService) -> None:
    """Cosine is undefined for a zero vector; a guess must never become a 500."""
    assert service.similarity("영벡터", "학생") == 0.0
    assert service.similarity("학생", "영벡터") == 0.0
    assert service.similarity("영벡터", "영벡터") == 0.0


# --- invalid vectors -------------------------------------------------------


@pytest.mark.parametrize("word", ["난수", "무한"])
def test_non_finite_vectors_are_rejected(service: FastTextEmbeddingService, word: str) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        service.encode(word)


def test_empty_vector_is_rejected(service: FastTextEmbeddingService) -> None:
    with pytest.raises(ValueError, match="empty vector"):
        service.encode("빈벡터")


def test_a_failing_model_call_is_wrapped_as_value_error() -> None:
    service = FastTextEmbeddingService(ExplodingFastTextModel())

    with pytest.raises(ValueError, match="could not produce a vector"):
        service.encode("학생")


# --- input normalization ---------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_blank_input_is_rejected(service: FastTextEmbeddingService, blank: str) -> None:
    """Same contract as DeterministicEmbeddingService, so the two swap cleanly."""
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        service.encode(blank)


def test_blank_input_is_rejected_by_similarity_too(service: FastTextEmbeddingService) -> None:
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        service.similarity("   ", "학생")


def test_surrounding_whitespace_is_trimmed(service: FastTextEmbeddingService) -> None:
    assert service.encode("  학생  ") == service.encode("학생")
    assert service.similarity(" 학생 ", "학생") == pytest.approx(1.0)


def test_input_is_nfkc_normalized(service: FastTextEmbeddingService) -> None:
    """Full-width 'ＡＢ' must reach the model as 'AB'."""
    model = FakeFastTextModel()
    normalizing_service = FastTextEmbeddingService(model)

    assert normalizing_service.encode("ＡＢ") == [0.0, 0.0, 1.0]
    assert model.calls == ["AB"]


def test_normalization_matches_the_domain_word_rule() -> None:
    """The vector lookup and the answer comparison must agree.

    `app.domain.game.normalize_word` is NFKC + strip; if this service normalized
    differently a guess could equal the answer yet not score 1.0.
    """
    from app.domain.game import normalize_word

    model = FakeFastTextModel()
    service = FastTextEmbeddingService(model)
    raw = "  학생  "

    service.encode(raw)

    assert model.calls == [normalize_word(raw)]


# --- project_3d ------------------------------------------------------------


def test_project_3d_is_not_implemented(service: FastTextEmbeddingService) -> None:
    with pytest.raises(NotImplementedError) as excinfo:
        service.project_3d(["학생", "선생"])

    message = str(excinfo.value)
    assert "not implemented" in message
    assert "Phase 2" in message
    assert "PCA" in message
    assert "placeholder" in message


# --- load() ----------------------------------------------------------------


def test_load_rejects_a_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "cc.ko.300.bin"

    with pytest.raises(FastTextConfigurationError) as excinfo:
        FastTextEmbeddingService.load(missing)

    message = str(excinfo.value)
    assert str(missing) in message
    assert "FASTTEXT_MODEL_PATH" in message
    assert "never downloads" in message


def test_load_rejects_a_directory(tmp_path: Path) -> None:
    with pytest.raises(FastTextConfigurationError) as excinfo:
        FastTextEmbeddingService.load(tmp_path)

    assert "directory, not a model file" in str(excinfo.value)


def test_load_reports_a_missing_optional_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the extra installed, the error must name the install command.

    `None` in `sys.modules` makes `import fasttext` raise, so this holds whether
    or not the extra happens to be installed locally.
    """
    model_file = tmp_path / "cc.ko.300.bin"
    model_file.write_bytes(b"not a real model")
    monkeypatch.setitem(sys.modules, "fasttext", None)

    with pytest.raises(FastTextConfigurationError) as excinfo:
        FastTextEmbeddingService.load(model_file)

    assert "uv sync --extra fasttext" in str(excinfo.value)
