"""The `INTERNAL_ERROR` catch-all, and proof it did not shadow the other handlers.

Every other documented error code is raised deliberately by the application, so
it can be tested by asking for something invalid. `INTERNAL_ERROR` is the
opposite: it exists for the failures nobody planned. These tests therefore
*inject* a failure — one through a synthetic route, one through the real guess
path — and assert two things about the response: it is the documented envelope,
and it says nothing else.

`raise_server_exceptions=False` is required throughout. Starlette's
ServerErrorMiddleware sends the handler's response and then re-raises so the
server still logs the failure; with the default `True`, `TestClient` would
surface that re-raise instead of the response a real client receives.
"""

import logging
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import rank_provider
from app.main import INTERNAL_ERROR_MESSAGE, create_app
from tests.conftest import TEST_ANSWER

# Everything a leak would look like, in one string: an internal exception
# message, a filesystem path, a model file name, and the answer word.
SECRET_DETAIL = f"C:/models/cc.ko.300.bin exploded while scoring {TEST_ANSWER!r}"


class ExplodingRankProvider:
    """A `RankProvider` that fails the way a real one would.

    Modelled on `VectorRankProvider._compute_similarities`, which interpolates
    the answer word into its `RankingError` message. That makes this the honest
    worst case for the secrecy rule rather than a contrived one.
    """

    def rank_of(self, answer: str, word: str) -> int | None:
        raise RuntimeError(SECRET_DETAIL)


@pytest.fixture
def failing_route_client() -> Iterator[TestClient]:
    """A client whose app has one route that raises an unhandled exception."""
    application = create_app()

    @application.get("/api/test/boom")
    def _boom() -> dict[str, str]:
        raise RuntimeError(SECRET_DETAIL)

    with TestClient(application, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def failing_guess_client(app: FastAPI) -> TestClient:
    """The real guess path, with a rank provider that blows up mid-scoring."""
    app.dependency_overrides[rank_provider] = ExplodingRankProvider
    return TestClient(app, raise_server_exceptions=False)


# --- The envelope ------------------------------------------------------------


def test_unhandled_exception_returns_500(failing_route_client: TestClient) -> None:
    assert failing_route_client.get("/api/test/boom").status_code == 500


def test_unhandled_exception_uses_the_documented_envelope(
    failing_route_client: TestClient,
) -> None:
    """Exactly the three documented keys, not Starlette's `detail` body."""
    body = failing_route_client.get("/api/test/boom").json()

    assert set(body) == {"code", "message", "details"}
    assert body["code"] == "INTERNAL_ERROR"
    assert body["message"] == INTERNAL_ERROR_MESSAGE
    assert body["details"] is None


def test_internal_error_message_is_fixed_not_derived(
    failing_route_client: TestClient, failing_guess_client: TestClient
) -> None:
    """Two unrelated failures must be indistinguishable to a client."""
    from_route = failing_route_client.get("/api/test/boom").json()

    game_id = failing_guess_client.post("/api/games").json()["gameId"]
    from_guess = failing_guess_client.post(
        f"/api/games/{game_id}/guesses", json={"word": "학생"}
    ).json()

    assert from_route == from_guess


# --- Nothing internal escapes ------------------------------------------------


@pytest.mark.parametrize(
    "leak",
    [
        SECRET_DETAIL,
        "cc.ko.300.bin",
        "C:/models",
        "RuntimeError",
        "Traceback",
        "main.py",
    ],
)
def test_response_body_carries_no_internal_detail(
    failing_route_client: TestClient, leak: str
) -> None:
    assert leak not in failing_route_client.get("/api/test/boom").text


def test_answer_is_not_exposed_by_a_failure_in_the_guess_path(
    failing_guess_client: TestClient,
) -> None:
    """The secrecy rule holds even when the crash message contains the answer."""
    game_id = failing_guess_client.post("/api/games").json()["gameId"]
    response = failing_guess_client.post(
        f"/api/games/{game_id}/guesses", json={"word": "학생"}
    )
    escaped = TEST_ANSWER.encode("unicode_escape").decode()

    assert response.status_code == 500
    assert TEST_ANSWER not in response.text
    assert escaped not in response.text
    assert SECRET_DETAIL not in response.text


def test_a_failed_guess_leaves_the_game_untouched(
    failing_guess_client: TestClient,
) -> None:
    """A 500 must not half-record a guess: `record_guess` never ran."""
    game_id = failing_guess_client.post("/api/games").json()["gameId"]
    failing_guess_client.post(f"/api/games/{game_id}/guesses", json={"word": "학생"})
    state = failing_guess_client.get(f"/api/games/{game_id}").json()

    assert state["guessCount"] == 0
    assert state["guesses"] == []
    assert state["status"] == "playing"


# --- What the handler logs ---------------------------------------------------


def test_the_handler_logs_the_failure_with_method_and_path(
    failing_route_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """A traceback in the log is wanted; the line we write adds no payload."""
    with caplog.at_level(logging.ERROR, logger="app.main"):
        failing_route_client.get("/api/test/boom")

    record = next(record for record in caplog.records if record.name == "app.main")

    assert record.exc_info is not None, "the traceback must reach the log"
    assert record.getMessage() == (
        "Unhandled exception while handling GET /api/test/boom; "
        "returning INTERNAL_ERROR."
    )


# --- The existing handlers still win -----------------------------------------


def test_app_error_handler_is_not_shadowed(client: TestClient) -> None:
    """A raised `AppError` subclass keeps its own code and status."""
    response = client.get("/api/games/does-not-exist")

    assert response.status_code == 404
    assert response.json()["code"] == "GAME_NOT_FOUND"


def test_validation_handler_is_not_shadowed(client: TestClient) -> None:
    game_id = client.post("/api/games").json()["gameId"]

    response = client.post(f"/api/games/{game_id}/guesses", json={"word": "   "})

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_INPUT"


def test_invalid_word_handler_is_not_shadowed(client: TestClient) -> None:
    game_id = client.post("/api/games").json()["gameId"]

    response = client.post(f"/api/games/{game_id}/guesses", json={"word": "두 단어"})

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_WORD"


def test_finished_game_conflict_is_not_shadowed(client: TestClient) -> None:
    game_id = client.post("/api/games").json()["gameId"]
    client.post(f"/api/games/{game_id}/guesses", json={"word": TEST_ANSWER})

    response = client.post(f"/api/games/{game_id}/guesses", json={"word": "학생"})

    assert response.status_code == 409
    assert response.json()["code"] == "GAME_ALREADY_FINISHED"


def test_unknown_route_still_returns_starlettes_404(client: TestClient) -> None:
    """`HTTPException` is untouched: the catch-all only covers real failures."""
    response = client.get("/api/definitely-not-a-route")

    assert response.status_code == 404
    assert response.json() == {"detail": "Not Found"}
