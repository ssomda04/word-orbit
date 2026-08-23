"""Rank-policy parity with the ML harness.

`VectorRankProvider` computes a rank by counting the words that would sort ahead
of a guess, while `ml/src/contextle_eval/rank_table.py` materializes the whole
ordering and reads off the index. The two must agree for every input, so this
module contains an independent line-by-line port of the ML algorithm
(`_reference_ranks`) and uses it as an oracle, alongside every policy case from
`ml/tests/test_rank_table.py`.

Float caveat, stated once
-------------------------
Ties are compared with exact float equality, and the two implementations do not
perform identical arithmetic: the ML harness sums Python floats left to right
and takes norms with `math.hypot`, while NumPy uses pairwise summation and its
own norm. Two classes of tie behave differently:

- **Structural ties** — a zero vector (an explicit `0.0` on both sides), or two
  words with identical vectors (identical inputs, identical code path, hence
  identical output *within* each implementation). These are exactly reproducible
  and are what the tests below construct.
- **Coincidental ties** — two different vectors whose cosines happen to land on
  the same float. Bit-level agreement is not guaranteed for these, and no test
  asserts it. In a 300-dimensional real model they are vanishingly rare and the
  consequence is a one-place rank difference between the harness and the service.
"""

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.api.deps import answer_selector, game_repository, rank_provider
from app.core.config import get_settings
from app.domain.vocabulary import normalize_vocabulary
from app.main import create_app
from app.services.embedding import reset_embedding_service
from app.services.game import InMemoryGameRepository
from app.services.ranking import (
    NonFiniteEmbeddingError,
    NullRankProvider,
    RankingError,
    VectorRankProvider,
    get_rank_provider,
    reset_rank_provider,
)
from tests.conftest import TEST_ANSWER


class FakeEmbeddingService:
    """Returns configured vectors and records every lookup.

    The counterpart of `FakeVectorProvider` in `ml/tests/test_rank_table.py`, so
    both implementations can be driven with byte-identical inputs.
    """

    def __init__(self, vectors: Mapping[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls: list[str] = []

    def encode(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self.vectors[text])

    def encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.encode(text) for text in texts]

    def similarity(self, first: str, second: str) -> float:
        raise AssertionError("ranking must not go through EmbeddingService.similarity")

    def project_3d(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


# --- Oracle: an independent port of the ML algorithm -------------------------


def _reference_normalize(word: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFKC", word).strip()


def _reference_normalize_vocabulary(words: Iterable[str]) -> tuple[str, ...]:
    unique: dict[str, None] = {}
    for word in words:
        normalized = _reference_normalize(word)
        if normalized:
            unique.setdefault(normalized, None)
    return tuple(unique)


def _reference_cosine(first: Sequence[float], second: Sequence[float]) -> float:
    """Copied from ml/src/contextle_eval/rank_table.py::_zero_safe_cosine."""
    if len(first) != len(second):
        raise ValueError("dimensions differ")
    first_norm = math.hypot(*first)
    second_norm = math.hypot(*second)
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    similarity = sum(
        (left / first_norm) * (right / second_norm)
        for left, right in zip(first, second, strict=True)
    )
    return max(-1.0, min(1.0, similarity))


def _reference_ranks(
    answer: str,
    vocabulary: Iterable[str],
    vectors: Mapping[str, Sequence[float]],
) -> dict[str, int]:
    """Port of ml/src/contextle_eval/rank_table.py::build_rank_table.

    Materializes the full ordering exactly as the harness does, then reads off
    each word's 1-based index.
    """
    normalized_answer = _reference_normalize(answer)
    words = list(_reference_normalize_vocabulary(vocabulary))
    if normalized_answer not in words:
        words.append(normalized_answer)

    answer_vector = vectors[normalized_answer]
    similarities = {
        word: _reference_cosine(
            answer_vector, answer_vector if word == normalized_answer else vectors[word]
        )
        for word in words
    }

    ranked = [normalized_answer]
    ranked.extend(
        sorted(
            (word for word in words if word != normalized_answer),
            key=lambda word: (-similarities[word], word),
        )
    )
    return {word: rank for rank, word in enumerate(ranked, start=1)}


def _provider(
    vocabulary: Iterable[str],
    vectors: Mapping[str, Sequence[float]],
    **kwargs: Any,
) -> VectorRankProvider:
    return VectorRankProvider(
        normalize_vocabulary(vocabulary), FakeEmbeddingService(vectors), **kwargs
    )


# --- Policy cases ported from ml/tests/test_rank_table.py --------------------


def test_ranks_by_descending_similarity() -> None:
    vectors = {
        "정답": [1.0, 0.0],
        "가까움": [0.8, 0.6],
        "직교": [0.0, 1.0],
        "반대": [-1.0, 0.0],
    }
    provider = _provider(["반대", "직교", "가까움", "정답"], vectors)

    assert provider.rank_of("정답", "정답") == 1
    assert provider.rank_of("정답", "가까움") == 2
    assert provider.rank_of("정답", "직교") == 3
    assert provider.rank_of("정답", "반대") == 4


def test_similarity_ties_use_lexical_order_but_answer_stays_first() -> None:
    """Ranks stay dense and unique across a tie — not competition ranking.

    Competition ranking would give both tied words rank 2. The ML harness
    separates them by word, ascending.
    """
    vectors = {"정답": [1.0, 0.0], "가": [1.0, 0.0], "나": [1.0, 0.0]}
    provider = _provider(["나", "가"], vectors)

    assert provider.rank_of("정답", "정답") == 1
    assert provider.rank_of("정답", "가") == 2
    assert provider.rank_of("정답", "나") == 3


def test_answer_outside_the_vocabulary_is_still_rank_one() -> None:
    """Ported from ml/tests/test_rank_table.py::test_missing_answer_is_automatically_added.

    The harness appends the answer to the word list; this provider never needs to,
    because the answer is excluded from the sorted tail either way.
    """
    vectors = {"정답": [1.0, 0.0], "후보": [0.0, 1.0]}
    provider = _provider(["후보"], vectors)

    assert provider.rank_of(" 정답 ", "정답") == 1
    assert provider.rank_of("정답", "후보") == 2


def test_answer_inside_the_vocabulary_does_not_shift_other_ranks() -> None:
    """Whether the answer is in the vocabulary must not change anyone else's rank."""
    vectors = {"정답": [1.0, 0.0], "가까움": [0.8, 0.6], "직교": [0.0, 1.0]}
    with_answer = _provider(["정답", "가까움", "직교"], vectors)
    without_answer = _provider(["가까움", "직교"], vectors)

    for word in ("가까움", "직교"):
        assert with_answer.rank_of("정답", word) == without_answer.rank_of("정답", word)


def test_answer_zero_vector_is_rank_one_and_everything_ties() -> None:
    """Ported from ml/tests/test_rank_table.py::test_answer_zero_vector_is_rank_one...

    A zero answer makes every similarity 0.0, so ordering falls back entirely to
    the lexicographic tie-break.
    """
    vectors = {"정답": [0.0, 0.0], "가": [1.0, 0.0], "나": [0.0, 0.0]}
    provider = _provider(["나", "가"], vectors)

    assert provider.rank_of("정답", "정답") == 1
    assert provider.rank_of("정답", "가") == 2
    assert provider.rank_of("정답", "나") == 3


def test_candidate_zero_vector_scores_zero_not_negative() -> None:
    """Ported from ml/tests/test_rank_table.py::test_candidate_zero_vector_similarity...

    A zero candidate scores 0.0, which outranks a genuinely opposite word.
    """
    vectors = {"정답": [1.0, 0.0], "영벡터": [0.0, 0.0], "반대": [-1.0, 0.0]}
    provider = _provider(["반대", "영벡터"], vectors)

    assert provider.rank_of("정답", "영벡터") == 2
    assert provider.rank_of("정답", "반대") == 3


def test_multiple_zero_vectors_tie_and_break_lexically() -> None:
    """Several zero vectors all score exactly 0.0, so only the word separates them."""
    vectors = {
        "정답": [1.0, 0.0],
        "영가": [0.0, 0.0],
        "영나": [0.0, 0.0],
        "영다": [0.0, 0.0],
        "반대": [-1.0, 0.0],
    }
    provider = _provider(["영다", "반대", "영나", "영가"], vectors)

    assert provider.rank_of("정답", "영가") == 2
    assert provider.rank_of("정답", "영나") == 3
    assert provider.rank_of("정답", "영다") == 4
    assert provider.rank_of("정답", "반대") == 5


def test_result_is_deterministic_across_vocabulary_order() -> None:
    """Ported from ml/tests/test_rank_table.py::test_result_is_deterministic..."""
    vectors = {
        "정답": [1.0, 0.0],
        "가": [0.5, 0.5],
        "나": [0.5, 0.5],
        "다": [0.0, 1.0],
    }
    first = _provider(["다", "나", "가", "나"], vectors)
    second = _provider(["가", "나", "다"], vectors)

    for word in ("가", "나", "다"):
        assert first.rank_of("정답", word) == second.rank_of("정답", word)


def test_word_outside_the_vocabulary_has_no_rank() -> None:
    """Ported from the `lookup("없는단어") is None` assertion in the ML tests."""
    vectors = {"정답": [1.0, 0.0], "가": [0.5, 0.5]}
    provider = _provider(["가"], vectors)

    assert provider.rank_of("정답", "없는단어") is None


def test_empty_answer_is_rejected() -> None:
    """Ported from ml/tests/test_rank_table.py::test_empty_answer_is_rejected."""
    provider = _provider(["가"], {"가": [1.0, 0.0]})

    with pytest.raises(RankingError, match="Answer must not be empty"):
        provider.rank_of("  ", "가")


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_vector_has_clear_error(non_finite: float) -> None:
    """Ported from ml/tests/test_rank_table.py::test_non_finite_vector_has_clear_error."""
    with pytest.raises(NonFiniteEmbeddingError, match="NaN or infinity"):
        _provider(["오류"], {"오류": [non_finite, 0.0]})


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
def test_non_finite_norm_has_clear_error() -> None:
    """Ported from ml/tests/test_rank_table.py::test_non_finite_norm_has_clear_error.

    NumPy warns while overflowing the sum of squares to infinity; the detection
    is the point, so the warning is filtered here rather than silenced globally.
    """
    with pytest.raises(NonFiniteEmbeddingError, match="norm"):
        _provider(["후보"], {"후보": [1.7e308, 1.7e308]})


def test_dimension_mismatch_in_vocabulary_has_clear_error() -> None:
    """Ported from ml/tests/test_rank_table.py::test_dimension_mismatch_has_clear_error."""
    with pytest.raises(RankingError, match="dimensions differ"):
        _provider(["정답", "오류"], {"정답": [1.0, 0.0], "오류": [1.0]})


def test_answer_dimension_mismatch_has_clear_error() -> None:
    provider = _provider(["가"], {"가": [1.0, 0.0], "정답": [1.0, 0.0, 0.0]})

    with pytest.raises(RankingError, match="dimensions differ"):
        provider.rank_of("정답", "가")


def test_subword_oov_answer_is_ranked_without_rejection() -> None:
    """Ported from ml/tests/test_rank_table.py::test_fasttext_adapter_allows_subword_oov.

    An answer the model has never seen still produces a vector, so ranking works.
    """
    vectors = {"사전에없는답": [1.0, 0.0], "가": [0.25, 0.75]}
    provider = _provider(["가"], vectors)

    assert provider.rank_of("사전에없는답", "가") == 2


# --- Equivalence against the oracle -----------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_matches_the_reference_implementation_on_every_word(seed: int) -> None:
    """Every word's rank equals the one the ML algorithm would assign.

    Vectors are drawn from a small pool so exact ties occur constantly — the case
    the counting formula has to get right. Identical vectors produce identical
    similarities within each implementation, so these ties are reproducible on
    both sides (see the module docstring).
    """
    rng = random.Random(seed)
    pool = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.5, 0.5, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
    ]
    words = [f"단어{index:03d}" for index in range(60)]
    rng.shuffle(words)
    vectors = {word: rng.choice(pool) for word in words}
    answer = words[0]

    provider = _provider(words, vectors)
    expected = _reference_ranks(answer, words, vectors)

    for word in words:
        assert provider.rank_of(answer, word) == expected[word], word


def test_reference_and_provider_agree_when_the_answer_is_outside_the_vocabulary() -> None:
    rng = random.Random(99)
    pool = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [-1.0, 0.0], [0.6, 0.8]]
    vocabulary = [f"어휘{index:02d}" for index in range(30)]
    vectors = {word: rng.choice(pool) for word in vocabulary}
    vectors["바깥정답"] = [0.6, 0.8]

    provider = _provider(vocabulary, vectors)
    expected = _reference_ranks("바깥정답", vocabulary, vectors)

    assert provider.rank_of("바깥정답", "바깥정답") == 1
    for word in vocabulary:
        assert provider.rank_of("바깥정답", word) == expected[word], word


def test_ranks_are_a_permutation_of_one_to_n() -> None:
    """Dense and unique: no rank is shared and none is skipped."""
    rng = random.Random(7)
    pool = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 0.0]]
    words = [f"단어{index:02d}" for index in range(25)]
    vectors = {word: rng.choice(pool) for word in words}
    answer = words[0]

    provider = _provider(words, vectors)
    ranks = sorted(provider.rank_of(answer, word) for word in words)

    assert ranks == list(range(1, len(words) + 1))


# --- Construction and caching ------------------------------------------------


def test_empty_vocabulary_is_rejected() -> None:
    with pytest.raises(RankingError, match="must not be empty"):
        VectorRankProvider((), FakeEmbeddingService({}))


def test_unnormalized_vocabulary_with_duplicates_is_rejected() -> None:
    """Guards the precondition that the tie-break relies on: unique words."""
    with pytest.raises(RankingError, match="duplicates"):
        VectorRankProvider(("가", "가"), FakeEmbeddingService({"가": [1.0, 0.0]}))


def test_similarities_are_computed_once_per_answer() -> None:
    """A second guess on the same answer must not re-embed the vocabulary."""
    vectors = {"정답": [1.0, 0.0], "가": [0.5, 0.5], "나": [0.0, 1.0]}
    embedder = FakeEmbeddingService(vectors)
    provider = VectorRankProvider(normalize_vocabulary(["가", "나"]), embedder)
    calls_after_build = list(embedder.calls)

    for _ in range(3):
        assert provider.rank_of("정답", "가") == 2
        assert provider.rank_of("정답", "나") == 3

    assert embedder.calls == [*calls_after_build, "정답"]


def test_answer_cache_is_bounded() -> None:
    """Memory is bounded by answers held, not by games played."""
    vectors = {"가": [1.0, 0.0], "나": [0.0, 1.0]}
    vectors.update({f"답{index}": [1.0, float(index)] for index in range(10)})
    embedder = FakeEmbeddingService(vectors)
    provider = VectorRankProvider(normalize_vocabulary(["가", "나"]), embedder, cache_size=2)

    for index in range(10):
        provider.rank_of(f"답{index}", "가")

    assert provider.cached_answer_count == 2


def test_cache_size_must_be_positive() -> None:
    with pytest.raises(RankingError, match="cache_size"):
        VectorRankProvider(("가",), FakeEmbeddingService({"가": [1.0, 0.0]}), cache_size=0)


def test_vocabulary_size_and_dimension_are_exposed() -> None:
    provider = _provider(["가", "나"], {"가": [1.0, 0.0, 0.0], "나": [0.0, 1.0, 0.0]})

    assert provider.vocabulary_size == 2
    assert provider.dimension == 3


# --- The null provider -------------------------------------------------------


def test_null_provider_ranks_nothing() -> None:
    provider = NullRankProvider()

    assert provider.rank_of("정답", "정답") is None
    assert provider.rank_of("정답", "아무단어") is None


# --- Wiring: settings, factory, service, API ---------------------------------


def _write_vocabulary(tmp_path: Path, words: Sequence[str]) -> Path:
    path = tmp_path / "game_words.txt"
    path.write_text("\n".join(words) + "\n", encoding="utf-8")
    return path


def test_factory_returns_the_null_provider_without_a_vocabulary_path() -> None:
    assert isinstance(get_rank_provider(), NullRankProvider)


def test_factory_builds_a_vector_provider_from_the_vocabulary_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_vocabulary(tmp_path, [TEST_ANSWER, "학생", "선생"])
    monkeypatch.setenv("VOCABULARY_PATH", str(path))
    get_settings.cache_clear()
    reset_embedding_service()
    reset_rank_provider()

    provider = get_rank_provider()

    assert isinstance(provider, VectorRankProvider)
    assert provider.vocabulary_size == 3
    # Built once and shared, like the embedding service.
    assert get_rank_provider() is provider


def test_factory_rejects_a_vocabulary_file_with_no_usable_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_vocabulary(tmp_path, ["", "   "])
    monkeypatch.setenv("VOCABULARY_PATH", str(path))
    get_settings.cache_clear()
    reset_rank_provider()

    with pytest.raises(ValueError, match="no usable words"):
        get_rank_provider()


def test_guess_response_carries_a_rank_when_a_vocabulary_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the rank reaches the wire under its camelCase name."""
    path = _write_vocabulary(tmp_path, [TEST_ANSWER, "학생", "선생", "바다", "커피"])
    monkeypatch.setenv("VOCABULARY_PATH", str(path))
    get_settings.cache_clear()
    reset_embedding_service()
    reset_rank_provider()

    application = create_app()
    repository = InMemoryGameRepository()
    application.dependency_overrides[game_repository] = lambda: repository
    application.dependency_overrides[answer_selector] = lambda: (lambda: TEST_ANSWER)
    client = TestClient(application)

    game_id = client.post("/api/games").json()["gameId"]
    guess = client.post(f"/api/games/{game_id}/guesses", json={"word": "학생"})
    winning = client.post(f"/api/games/{game_id}/guesses", json={"word": TEST_ANSWER})

    body = guess.json()
    assert guess.status_code == 200
    assert isinstance(body["rank"], int)
    assert 2 <= body["rank"] <= 5
    # The answer is rank 1 by construction.
    assert winning.json()["rank"] == 1


def test_guess_outside_the_vocabulary_reports_a_null_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_vocabulary(tmp_path, [TEST_ANSWER, "학생"])
    monkeypatch.setenv("VOCABULARY_PATH", str(path))
    get_settings.cache_clear()
    reset_embedding_service()
    reset_rank_provider()

    application = create_app()
    repository = InMemoryGameRepository()
    application.dependency_overrides[game_repository] = lambda: repository
    application.dependency_overrides[answer_selector] = lambda: (lambda: TEST_ANSWER)
    client = TestClient(application)

    game_id = client.post("/api/games").json()["gameId"]
    guess = client.post(f"/api/games/{game_id}/guesses", json={"word": "어휘밖단어"})

    assert guess.status_code == 200
    assert guess.json()["rank"] is None


def test_rank_stays_null_by_default(client: TestClient) -> None:
    """No vocabulary configured: the contract's null rank, unchanged."""
    game_id = client.post("/api/games").json()["gameId"]
    guess = client.post(f"/api/games/{game_id}/guesses", json={"word": "학생"})

    assert guess.json()["rank"] is None


def test_a_replayed_guess_keeps_its_original_rank(app: Any) -> None:
    """Idempotent replay returns the stored guess, rank included."""
    ranks = iter([7, 99])
    app.dependency_overrides[rank_provider] = lambda: _FixedRankProvider(ranks)
    client = TestClient(app)

    game_id = client.post("/api/games").json()["gameId"]
    first = client.post(f"/api/games/{game_id}/guesses", json={"word": "학생"})
    replay = client.post(f"/api/games/{game_id}/guesses", json={"word": " 학생 "})

    assert first.json()["rank"] == 7
    assert replay.json()["rank"] == 7


class _FixedRankProvider:
    """Hands out preset ranks so a replay can be told apart from a fresh guess."""

    def __init__(self, ranks: Any) -> None:
        self._ranks = ranks

    def rank_of(self, answer: str, word: str) -> int | None:
        return next(self._ranks)
