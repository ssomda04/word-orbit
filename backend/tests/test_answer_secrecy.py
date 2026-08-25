"""The answer word must not reach the server log, even through a traceback.

`tests/test_games_api.py` and `tests/test_error_handling.py` already pin the
half of this rule that faces the client. This module pins the other half. It is
a separate concern because it fails differently: a response is built by code
that knows what it is allowed to say, while a traceback is assembled from
whatever every exception along the chain happened to put in its message. One
`{answer!r}` anywhere in that chain is enough.

Every test here provokes a *real* failure through the real classes — no
monkeypatched messages — and then reads back the exception chain, the rendered
traceback, and everything that was logged.

The one boundary this cannot cover is named explicitly at the bottom: a message
raised by a third-party library, which we can decline to interpolate but cannot
rewrite.
"""

import logging
import math
import traceback
from collections.abc import Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import rank_provider
from app.services.embedding import FastTextEmbeddingService
from app.services.ranking import RankingError, VectorRankProvider
from tests.conftest import TEST_ANSWER

VOCABULARY = ("학생", "선생")
GUESS = "학생"
GOOD_VECTOR = [1.0, 0.0]


def _forbidden_forms(word: str) -> tuple[str, ...]:
    """Every spelling the word could survive as: raw, repr, and escaped."""
    return (word, repr(word), word.encode("unicode_escape").decode())


def _rendered_traceback(exc: BaseException) -> str:
    """The traceback exactly as a logging handler would render it, chain included."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _assert_absent(haystack: str, word: str) -> None:
    for form in _forbidden_forms(word):
        assert form not in haystack


class AnswerHostileEmbedder:
    """Embeds the vocabulary fine and fails on anything else.

    The asymmetry is the point: the provider must build successfully, so the only
    word it can fail on afterwards is the answer.

    Its own error deliberately does *not* quote its input, because neither
    production embedder does any more. A fake that leaked would be testing the
    fake.
    """

    def __init__(self, failure: str = "raise") -> None:
        self.failure = failure

    def encode(self, text: str) -> list[float]:
        if text in VOCABULARY:
            return list(GOOD_VECTOR)
        if self.failure == "raise":
            raise RuntimeError("embedding backend is unavailable")
        if self.failure == "empty":
            return []
        return [1.0, 0.0, 0.0]  # wrong dimension

    def encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.encode(text) for text in texts]

    def similarity(self, first: str, second: str) -> float:
        raise NotImplementedError

    def project_3d(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class AnswerHostileModel:
    """A FastText model object that returns a bad vector for anything unknown.

    Used to drive the *real* `FastTextEmbeddingService`, so that a rank lookup
    fails through two layers of production code rather than one fake.
    """

    def get_word_vector(self, word: str) -> list[float]:
        if word in VOCABULARY:
            return list(GOOD_VECTOR)
        return [math.nan, 0.0]


class WordEchoingModel:
    """A model whose own native error quotes its input, as pybind11 might."""

    def get_word_vector(self, word: str) -> list[float]:
        raise RuntimeError(f"native loader failed on {word!r}")


# --- VectorRankProvider: the answer never appears -----------------------------


@pytest.mark.parametrize("failure", ["raise", "empty", "dimension"])
def test_a_failure_on_the_answer_never_names_it(failure: str) -> None:
    provider = VectorRankProvider(VOCABULARY, AnswerHostileEmbedder(failure))

    with pytest.raises(RankingError) as caught:
        provider.rank_of(TEST_ANSWER, GUESS)

    _assert_absent(str(caught.value), TEST_ANSWER)


def test_the_message_still_says_what_broke() -> None:
    """Secrecy must not cost debuggability: the failure stays identifiable."""
    provider = VectorRankProvider(VOCABULARY, AnswerHostileEmbedder("dimension"))

    with pytest.raises(RankingError, match="dimensions differ"):
        provider.rank_of(TEST_ANSWER, GUESS)


def test_the_underlying_cause_is_still_chained() -> None:
    """`from exc` is kept, so the real stack is not lost to the redaction."""
    provider = VectorRankProvider(VOCABULARY, AnswerHostileEmbedder("raise"))

    with pytest.raises(RankingError) as caught:
        provider.rank_of(TEST_ANSWER, GUESS)

    assert isinstance(caught.value.__cause__, RuntimeError)


@pytest.mark.parametrize("failure", ["raise", "empty", "dimension"])
def test_the_whole_rendered_traceback_is_clean(failure: str) -> None:
    """Not just our message — every frame and every chained message."""
    provider = VectorRankProvider(VOCABULARY, AnswerHostileEmbedder(failure))

    try:
        provider.rank_of(TEST_ANSWER, GUESS)
    except RankingError as exc:
        _assert_absent(_rendered_traceback(exc), TEST_ANSWER)
    else:  # pragma: no cover - the provider is built to fail here
        pytest.fail("the provider was expected to fail on the answer")


# --- FastTextEmbeddingService: the word never appears -------------------------


@pytest.mark.parametrize("failure", ["empty", "non-finite"])
def test_a_fasttext_failure_never_names_the_word(failure: str) -> None:
    """`_vector` gets the guess *and* the answer and cannot tell them apart."""

    class Model:
        def get_word_vector(self, word: str) -> list[float]:
            return [] if failure == "empty" else [math.nan, 0.0]

    service = FastTextEmbeddingService(Model())

    with pytest.raises(ValueError) as caught:
        service.encode(TEST_ANSWER)

    _assert_absent(_rendered_traceback(caught.value), TEST_ANSWER)


def test_a_fasttext_failure_still_identifies_the_stage() -> None:
    service = FastTextEmbeddingService(AnswerHostileModel())

    with pytest.raises(ValueError, match="non-finite"):
        service.encode(TEST_ANSWER)


# --- Both layers together, as production wires them ---------------------------


def test_a_real_two_layer_failure_is_clean() -> None:
    """The production chain: rank provider -> FastText service -> bad vector.

    Every message in this traceback is one this repository writes, which is what
    makes the guarantee hold end to end rather than one class at a time.
    """
    provider = VectorRankProvider(VOCABULARY, FastTextEmbeddingService(AnswerHostileModel()))

    try:
        provider.rank_of(TEST_ANSWER, GUESS)
    except RankingError as exc:
        rendered = _rendered_traceback(exc)
    else:  # pragma: no cover - the model is built to fail here
        pytest.fail("the provider was expected to fail on the answer")

    _assert_absent(rendered, TEST_ANSWER)
    assert "ValueError" in rendered, "the inner failure must still be visible"
    assert "RankingError" in rendered


# --- End to end: nothing reaches the log -------------------------------------


@pytest.fixture
def failing_rank_client(app: FastAPI) -> TestClient:
    """A real app whose real rank provider fails on the real hidden answer."""
    provider = VectorRankProvider(VOCABULARY, FastTextEmbeddingService(AnswerHostileModel()))
    app.dependency_overrides[rank_provider] = lambda: provider
    return TestClient(app, raise_server_exceptions=False)


def test_the_answer_is_absent_from_everything_logged(
    failing_rank_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The whole point: a 500 mid-game must not print the hidden word."""
    game_id = failing_rank_client.post("/api/games").json()["gameId"]

    with caplog.at_level(logging.DEBUG):
        response = failing_rank_client.post(
            f"/api/games/{game_id}/guesses", json={"word": GUESS}
        )

    assert response.status_code == 500
    assert any(record.exc_info for record in caplog.records), (
        "a traceback must actually have been logged, or this test proves nothing"
    )
    _assert_absent(caplog.text, TEST_ANSWER)


def test_the_response_stays_clean_too(failing_rank_client: TestClient) -> None:
    """Belt and braces with `test_error_handling`: same failure, client side."""
    game_id = failing_rank_client.post("/api/games").json()["gameId"]

    response = failing_rank_client.post(
        f"/api/games/{game_id}/guesses", json={"word": GUESS}
    )

    assert response.json()["code"] == "INTERNAL_ERROR"
    _assert_absent(response.text, TEST_ANSWER)


def test_a_guess_word_is_still_safe_to_report(client: TestClient) -> None:
    """Redaction is scoped: a rejected guess still echoes back to its player."""
    game_id = client.post("/api/games").json()["gameId"]

    response = client.post(f"/api/games/{game_id}/guesses", json={"word": "두 단어"})

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_WORD"


# --- The boundary we do not control ------------------------------------------


def test_a_third_party_message_is_declined_not_rewritten() -> None:
    """The one gap, recorded rather than hidden.

    A native loader may quote its own input. We do not interpolate it, so *our*
    message is clean — but `from exc` keeps that message in the chain. It stays
    chained deliberately: this branch is only reachable when the model file
    itself is broken, and dropping the cause would discard the only diagnostic
    for it. `VectorRankProvider` above has no such gap, because every exception
    it chains is one of ours.
    """
    service = FastTextEmbeddingService(WordEchoingModel())

    with pytest.raises(ValueError) as caught:
        service.encode(TEST_ANSWER)

    _assert_absent(str(caught.value), TEST_ANSWER)
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert TEST_ANSWER in str(caught.value.__cause__), (
        "if this ever stops being true the library changed; re-check the gap"
    )
