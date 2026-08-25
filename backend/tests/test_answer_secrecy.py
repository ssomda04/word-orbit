"""The answer word must not reach the server log, even through a traceback.

`tests/test_games_api.py` and `tests/test_error_handling.py` already pin the
half of this rule that faces the client. This module pins the other half. It is
a separate concern because it fails differently: a response is built by code
that knows what it is allowed to say, while a traceback is assembled from
whatever every exception along the chain happened to put in its message. One
`{answer!r}` anywhere in that chain is enough — and a chain includes exceptions
this repository did not raise.

The policy these tests enforce, in full:

- the outer exception's message does not name the answer;
- no ``__cause__`` names it either;
- the whole rendered traceback does not contain it;
- nothing logged during the failure contains it.

Every test provokes a *real* failure through the real classes — no monkeypatched
messages — and then reads back the exception chain, the rendered traceback, and
`caplog`.
"""

import logging
import math
import traceback
from collections.abc import Sequence

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import embedding_service, rank_provider
from app.services.embedding import FastTextEmbeddingService
from app.services.ranking import RankingError, VectorRankProvider
from tests.conftest import TEST_ANSWER

VOCABULARY = ("학생", "선생")
GUESS = "학생"
GOOD_VECTOR = [1.0, 0.0]

# The distinctive part of a native error that quotes its input. Asserting on this
# separately from the answer catches a leak even if the word itself were escaped
# or transliterated on the way into the log.
NATIVE_FAILURE = "native model failed for"

# What the sanitized message must still say, so redaction is not mistaken for
# a test that would also pass if the log said nothing useful at all.
SANITIZED_MARKER = "could not produce a vector for the requested word"


def _forbidden_forms(word: str) -> tuple[str, ...]:
    """Every spelling the word could survive as: raw, repr, and escaped."""
    return (word, repr(word), word.encode("unicode_escape").decode())


def _rendered_traceback(exc: BaseException) -> str:
    """The traceback exactly as a logging handler renders it, chain included."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _assert_secret_free(haystack: str, word: str = TEST_ANSWER) -> None:
    for form in _forbidden_forms(word):
        assert form not in haystack
    assert NATIVE_FAILURE not in haystack


class AnswerHostileEmbedder:
    """Embeds the vocabulary fine and fails on anything else.

    The asymmetry is the point: the provider must build successfully, so the only
    word it can fail on afterwards is the answer.

    Its own error deliberately does *not* quote its input, because neither
    production embedder does any more. `WordEchoingModel` covers the case where
    something outside this repository does.
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
        return self.encode(first)[0] * self.encode(second)[0]

    def project_3d(self, texts: Sequence[str]) -> list[list[float]]:
        raise NotImplementedError


class AnswerHostileModel:
    """A FastText model object returning a bad vector for anything unknown.

    Drives the *real* `FastTextEmbeddingService`, so a rank lookup fails through
    two layers of production code rather than one fake.
    """

    def get_word_vector(self, word: str) -> list[float]:
        if word in VOCABULARY:
            return list(GOOD_VECTOR)
        return [math.nan, 0.0]


class WordEchoingModel:
    """A native model that quotes its input in its own error, as pybind11 may.

    This is the adversary the secrecy rule actually has to survive: code outside
    this repository, whose message we cannot rewrite. The vocabulary succeeds so
    that a `VectorRankProvider` can be built on top of it; once built, the only
    word left to fail on is the answer.
    """

    def get_word_vector(self, word: str) -> list[float]:
        if word in VOCABULARY:
            return list(GOOD_VECTOR)
        raise RuntimeError(f"{NATIVE_FAILURE} {word!r}")


# --- VectorRankProvider: the answer never appears -----------------------------


@pytest.mark.parametrize("failure", ["raise", "empty", "dimension"])
def test_a_failure_on_the_answer_never_names_it(failure: str) -> None:
    provider = VectorRankProvider(VOCABULARY, AnswerHostileEmbedder(failure))

    with pytest.raises(RankingError) as caught:
        provider.rank_of(TEST_ANSWER, GUESS)

    _assert_secret_free(str(caught.value))


def test_the_message_still_says_what_broke() -> None:
    """Secrecy must not cost debuggability: the failure stays identifiable."""
    provider = VectorRankProvider(VOCABULARY, AnswerHostileEmbedder("dimension"))

    with pytest.raises(RankingError, match="dimensions differ"):
        provider.rank_of(TEST_ANSWER, GUESS)


def test_an_in_repo_cause_is_still_chained() -> None:
    """Chaining is only dropped at the third-party boundary, not everywhere.

    Here the cause is an `EmbeddingService`, whose messages this repository
    controls and keeps clean, so `from exc` stays and the real stack survives.
    """
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
        _assert_secret_free(_rendered_traceback(exc))
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

    _assert_secret_free(_rendered_traceback(caught.value))


def test_a_fasttext_failure_still_identifies_the_stage() -> None:
    service = FastTextEmbeddingService(AnswerHostileModel())

    with pytest.raises(ValueError, match="non-finite"):
        service.encode(TEST_ANSWER)


# --- The third-party boundary: the cause is suppressed, not merely unquoted ----


def test_a_native_cause_is_suppressed_not_chained() -> None:
    """`from None`: a message we cannot rewrite must not survive as `__cause__`."""
    service = FastTextEmbeddingService(WordEchoingModel())

    with pytest.raises(ValueError) as caught:
        service.encode(TEST_ANSWER)

    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_a_native_message_cannot_reach_a_rendered_traceback() -> None:
    service = FastTextEmbeddingService(WordEchoingModel())

    with pytest.raises(ValueError) as caught:
        service.encode(TEST_ANSWER)

    _assert_secret_free(_rendered_traceback(caught.value))


def test_suppression_survives_being_chained_further_up() -> None:
    """`VectorRankProvider` re-chains with `from exc`; the suppression holds.

    Rendering a cause renders *its* chain too, so the guarantee is only worth
    anything if it is transitive. This is the two-layer proof.
    """
    provider = VectorRankProvider(VOCABULARY, FastTextEmbeddingService(WordEchoingModel()))

    try:
        provider.rank_of(TEST_ANSWER, GUESS)
    except RankingError as exc:
        rendered = _rendered_traceback(exc)
    else:  # pragma: no cover - the model is built to fail here
        pytest.fail("the provider was expected to fail on the answer")

    _assert_secret_free(rendered)
    # The sanitized inner failure is still there, or the redaction went too far.
    assert SANITIZED_MARKER in rendered
    assert "RankingError" in rendered


def test_the_failing_exception_type_is_still_reported() -> None:
    """Type names carry no input, so they replace the message we dropped."""
    service = FastTextEmbeddingService(WordEchoingModel())

    with pytest.raises(ValueError, match=r"\(RuntimeError\)"):
        service.encode(TEST_ANSWER)


# --- Both layers together, without a native message ---------------------------


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

    _assert_secret_free(rendered)
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
    """A 500 mid-game must not print the hidden word."""
    game_id = failing_rank_client.post("/api/games").json()["gameId"]

    with caplog.at_level(logging.DEBUG):
        response = failing_rank_client.post(
            f"/api/games/{game_id}/guesses", json={"word": GUESS}
        )

    assert response.status_code == 500
    assert any(record.exc_info for record in caplog.records), (
        "a traceback must actually have been logged, or this test proves nothing"
    )
    _assert_secret_free(caplog.text)


def test_the_response_stays_clean_too(failing_rank_client: TestClient) -> None:
    """Belt and braces with `test_error_handling`: same failure, client side."""
    game_id = failing_rank_client.post("/api/games").json()["gameId"]

    response = failing_rank_client.post(
        f"/api/games/{game_id}/guesses", json={"word": GUESS}
    )

    assert response.json()["code"] == "INTERNAL_ERROR"
    _assert_secret_free(response.text)


@pytest.mark.parametrize("wiring", ["embedding", "ranking"])
def test_a_native_error_that_echoes_its_input_never_reaches_the_log(
    app: FastAPI, caplog: pytest.LogCaptureFixture, wiring: str
) -> None:
    """The full production path, with an adversarial native model.

    `embedding`: FastTextEmbeddingService -> EmbeddingGuessScorer -> GameService
                 -> the INTERNAL_ERROR handler -> logger.exception.
    `ranking`:   the same service one layer deeper, behind VectorRankProvider,
                 so the suppression is exercised through a further `from exc`.

    In both cases the native error names the hidden answer, and in both cases
    nothing that reaches the client or the log may.
    """
    service = FastTextEmbeddingService(WordEchoingModel())
    if wiring == "embedding":
        app.dependency_overrides[embedding_service] = lambda: service
    else:
        provider = VectorRankProvider(VOCABULARY, service)
        app.dependency_overrides[rank_provider] = lambda: provider
    client = TestClient(app, raise_server_exceptions=False)

    game_id = client.post("/api/games").json()["gameId"]
    with caplog.at_level(logging.DEBUG):
        response = client.post(f"/api/games/{game_id}/guesses", json={"word": GUESS})

    assert response.status_code == 500
    assert response.json()["code"] == "INTERNAL_ERROR"
    assert any(record.exc_info for record in caplog.records), (
        "a traceback must actually have been logged, or this test proves nothing"
    )
    # The log is still useful — this is redaction, not silence.
    assert SANITIZED_MARKER in caplog.text
    _assert_secret_free(caplog.text)
    _assert_secret_free(response.text)


def test_a_guess_word_is_still_safe_to_report(client: TestClient) -> None:
    """Redaction is scoped: a rejected guess still echoes back to its player."""
    game_id = client.post("/api/games").json()["gameId"]

    response = client.post(f"/api/games/{game_id}/guesses", json={"word": "두 단어"})

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_WORD"
