"""Contract tests for the single-player game API (docs/API_SPEC.md).

These assert the wire shape (camelCase, nullable fields, error envelope) and the
two rules that matter most: the answer word never leaks while a game is playing,
and a finished game stops accepting guesses.
"""

import logging
import re
import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.domain.game import Game
from app.schemas.game import to_give_up_response
from tests.conftest import OTHER_WRONG_WORD, TEST_ANSWER, WRONG_WORD

# `2026-07-24T11:22:57Z` — the documented timestamp shape.
ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def _create_game(client: TestClient) -> str:
    response = client.post("/api/games")
    assert response.status_code == 200
    return response.json()["gameId"]


def _guess(client: TestClient, game_id: str, word: str):
    return client.post(f"/api/games/{game_id}/guesses", json={"word": word})


def _give_up(client: TestClient, game_id: str):
    return client.post(f"/api/games/{game_id}/give-up")


def _assert_answer_absent(text: str) -> None:
    """Fail if the answer word appears, raw or as a JSON unicode escape."""
    escaped = TEST_ANSWER.encode("unicode_escape").decode()
    assert TEST_ANSWER not in text
    assert escaped not in text


# --- POST /api/games -------------------------------------------------------


def test_create_game_returns_documented_shape(client: TestClient) -> None:
    response = client.post("/api/games")
    body = response.json()

    assert response.status_code == 200
    assert set(body) == {"gameId", "status", "createdAt"}
    assert body["status"] == "playing"
    assert ISO_Z.match(body["createdAt"])


def test_create_game_never_returns_an_answer_field(client: TestClient) -> None:
    response = client.post("/api/games")

    assert "answer" not in response.json()
    _assert_answer_absent(response.text)


def test_game_id_is_an_opaque_uuid(client: TestClient) -> None:
    """Non-enumerable ids stop a client from reading another player's game."""
    game_id = _create_game(client)

    assert uuid.UUID(game_id)


def test_each_game_gets_a_distinct_id(client: TestClient) -> None:
    assert _create_game(client) != _create_game(client)


# --- POST /api/games/{gameId}/guesses --------------------------------------


def test_submit_guess_returns_documented_shape(client: TestClient) -> None:
    game_id = _create_game(client)

    response = _guess(client, game_id, WRONG_WORD)
    body = response.json()

    assert response.status_code == 200
    assert set(body) == {"guessId", "word", "similarity", "rank", "isAnswer", "coordinate"}
    assert body["guessId"] == "guess-001"
    assert body["word"] == WRONG_WORD
    assert body["isAnswer"] is False
    assert -1.0 <= body["similarity"] <= 1.0


def test_rank_and_coordinate_are_null_in_phase_1(client: TestClient) -> None:
    game_id = _create_game(client)

    body = _guess(client, game_id, WRONG_WORD).json()

    assert body["rank"] is None
    assert body["coordinate"] is None


def test_guess_word_is_normalized_in_the_response(client: TestClient) -> None:
    game_id = _create_game(client)

    body = _guess(client, game_id, f"  {WRONG_WORD}  ").json()

    assert body["word"] == WRONG_WORD


def test_blank_guess_is_rejected_as_invalid_input(client: TestClient) -> None:
    game_id = _create_game(client)

    response = _guess(client, game_id, "   ")
    body = response.json()

    assert response.status_code == 422
    assert body["code"] == "INVALID_INPUT"
    assert set(body) == {"code", "message", "details"}


def test_empty_guess_is_rejected_as_invalid_input(client: TestClient) -> None:
    game_id = _create_game(client)

    response = _guess(client, game_id, "")

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_INPUT"


def test_missing_word_field_is_rejected(client: TestClient) -> None:
    game_id = _create_game(client)

    response = client.post(f"/api/games/{game_id}/guesses", json={})

    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_INPUT"


def test_unprocessable_word_is_rejected_as_invalid_word(client: TestClient) -> None:
    game_id = _create_game(client)

    response = _guess(client, game_id, "가" * 51)
    body = response.json()

    assert response.status_code == 400
    assert body["code"] == "INVALID_WORD"
    assert set(body) == {"code", "message", "details"}


def test_guess_on_unknown_game_returns_game_not_found(client: TestClient) -> None:
    response = _guess(client, "00000000-0000-4000-8000-000000000000", WRONG_WORD)
    body = response.json()

    assert response.status_code == 404
    assert body["code"] == "GAME_NOT_FOUND"
    assert set(body) == {"code", "message", "details"}


# --- Duplicate guesses (idempotent) ---------------------------------------


def test_duplicate_guess_returns_the_stored_result(client: TestClient) -> None:
    game_id = _create_game(client)

    first = _guess(client, game_id, WRONG_WORD).json()
    repeated = _guess(client, game_id, WRONG_WORD)

    assert repeated.status_code == 200
    assert repeated.json() == first


def test_duplicate_guess_does_not_grow_the_history(client: TestClient) -> None:
    game_id = _create_game(client)

    _guess(client, game_id, WRONG_WORD)
    _guess(client, game_id, WRONG_WORD)
    _guess(client, game_id, f" {WRONG_WORD} ")  # same word after normalization

    state = client.get(f"/api/games/{game_id}").json()

    assert state["guessCount"] == 1
    assert len(state["guesses"]) == 1


def test_duplicate_guess_does_not_consume_a_guess_id(client: TestClient) -> None:
    game_id = _create_game(client)

    _guess(client, game_id, WRONG_WORD)
    _guess(client, game_id, WRONG_WORD)
    third = _guess(client, game_id, OTHER_WRONG_WORD).json()

    assert third["guessId"] == "guess-002"


# --- Finished games (409) -------------------------------------------------


def test_guess_after_win_is_rejected_with_conflict(client: TestClient) -> None:
    game_id = _create_game(client)
    _guess(client, game_id, TEST_ANSWER)

    response = _guess(client, game_id, WRONG_WORD)
    body = response.json()

    assert response.status_code == 409
    assert body["code"] == "GAME_ALREADY_FINISHED"
    assert set(body) == {"code", "message", "details"}


def test_replaying_the_winning_guess_after_win_is_also_rejected(client: TestClient) -> None:
    """The finished check runs before the duplicate check — see docs/API_SPEC.md."""
    game_id = _create_game(client)
    _guess(client, game_id, TEST_ANSWER)

    response = _guess(client, game_id, TEST_ANSWER)

    assert response.status_code == 409
    assert response.json()["code"] == "GAME_ALREADY_FINISHED"


def test_rejected_guess_after_win_does_not_change_state(client: TestClient) -> None:
    game_id = _create_game(client)
    _guess(client, game_id, TEST_ANSWER)

    _guess(client, game_id, WRONG_WORD)
    state = client.get(f"/api/games/{game_id}").json()

    assert state["guessCount"] == 1
    assert state["status"] == "won"


# --- GET /api/games/{gameId} ---------------------------------------------


def test_get_game_returns_documented_shape(client: TestClient) -> None:
    game_id = _create_game(client)
    _guess(client, game_id, WRONG_WORD)

    response = client.get(f"/api/games/{game_id}")
    body = response.json()

    assert response.status_code == 200
    assert set(body) == {"gameId", "status", "createdAt", "guessCount", "guesses", "answer"}
    assert body["gameId"] == game_id
    assert body["status"] == "playing"
    assert body["guessCount"] == 1
    assert ISO_Z.match(body["createdAt"])
    assert set(body["guesses"][0]) == {
        "guessId",
        "word",
        "similarity",
        "rank",
        "isAnswer",
        "coordinate",
    }


def test_get_game_returns_guesses_in_submission_order(client: TestClient) -> None:
    game_id = _create_game(client)
    words = [WRONG_WORD, OTHER_WRONG_WORD, "학교"]
    for word in words:
        _guess(client, game_id, word)

    body = client.get(f"/api/games/{game_id}").json()

    assert [guess["word"] for guess in body["guesses"]] == words


def test_get_unknown_game_returns_game_not_found(client: TestClient) -> None:
    response = client.get("/api/games/00000000-0000-4000-8000-000000000000")

    assert response.status_code == 404
    assert response.json()["code"] == "GAME_NOT_FOUND"


# --- Answer confidentiality ----------------------------------------------


def test_answer_is_never_exposed_while_playing(client: TestClient) -> None:
    """The core secrecy rule (AGENTS.md, docs/API_SPEC.md).

    Walks the whole in-progress flow and asserts the answer word appears in no
    response body at all — not just that `answer` is null.
    """
    create = client.post("/api/games")
    game_id = create.json()["gameId"]
    _assert_answer_absent(create.text)

    for word in (WRONG_WORD, OTHER_WRONG_WORD):
        guess = _guess(client, game_id, word)
        assert guess.status_code == 200
        _assert_answer_absent(guess.text)

    state = client.get(f"/api/games/{game_id}")
    assert state.json()["status"] == "playing"
    assert state.json()["answer"] is None
    _assert_answer_absent(state.text)


def test_answer_is_never_logged_while_playing(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The answer must not reach logs either (AGENTS.md)."""
    with caplog.at_level(logging.DEBUG):
        game_id = _create_game(client)
        _guess(client, game_id, WRONG_WORD)
        client.get(f"/api/games/{game_id}")

    _assert_answer_absent(caplog.text)


def test_error_responses_do_not_leak_the_answer(client: TestClient) -> None:
    game_id = _create_game(client)

    for response in (
        _guess(client, game_id, "   "),
        _guess(client, game_id, "가" * 51),
        _guess(client, "00000000-0000-4000-8000-000000000000", WRONG_WORD),
    ):
        _assert_answer_absent(response.text)


# --- Reveal on win -------------------------------------------------------


def test_winning_guess_marks_the_game_won(client: TestClient) -> None:
    game_id = _create_game(client)

    body = _guess(client, game_id, TEST_ANSWER).json()

    assert body["isAnswer"] is True
    assert body["similarity"] == pytest.approx(1.0)


def test_answer_is_revealed_after_the_game_is_won(client: TestClient) -> None:
    game_id = _create_game(client)
    _guess(client, game_id, WRONG_WORD)
    _guess(client, game_id, TEST_ANSWER)

    body = client.get(f"/api/games/{game_id}").json()

    assert body["status"] == "won"
    assert body["answer"] == TEST_ANSWER
    assert body["guessCount"] == 2


# --- POST /api/games/{gameId}/give-up ------------------------------------


def test_give_up_returns_documented_shape(client: TestClient) -> None:
    game_id = _create_game(client)

    response = _give_up(client, game_id)
    body = response.json()

    assert response.status_code == 200
    assert set(body) == {"gameId", "status", "finishReason", "answer"}
    assert body["gameId"] == game_id
    assert body["status"] == "abandoned"
    assert body["finishReason"] == "gave_up"


def test_give_up_reveals_the_real_answer(client: TestClient) -> None:
    """Not just *an* answer: the word this game was actually set on."""
    game_id = _create_game(client)

    body = _give_up(client, game_id).json()

    assert body["answer"] == TEST_ANSWER


def test_give_up_after_guessing_keeps_the_history(client: TestClient) -> None:
    """Giving up ends the round; it does not count as an attempt."""
    game_id = _create_game(client)
    _guess(client, game_id, WRONG_WORD)

    _give_up(client, game_id)
    state = client.get(f"/api/games/{game_id}").json()

    assert state["guessCount"] == 1
    assert state["status"] == "abandoned"
    assert state["answer"] == TEST_ANSWER


def test_guess_after_give_up_is_rejected_with_conflict(client: TestClient) -> None:
    """The block is server-side state, not a disabled input in the client."""
    game_id = _create_game(client)
    _give_up(client, game_id)

    response = _guess(client, game_id, WRONG_WORD)
    body = response.json()

    assert response.status_code == 409
    assert body["code"] == "GAME_ALREADY_FINISHED"
    assert set(body) == {"code", "message", "details"}


def test_guessing_the_answer_after_give_up_is_also_rejected(client: TestClient) -> None:
    """Even the winning word: an abandoned game accepts nothing at all."""
    game_id = _create_game(client)
    _give_up(client, game_id)

    response = _guess(client, game_id, TEST_ANSWER)

    assert response.status_code == 409
    assert response.json()["code"] == "GAME_ALREADY_FINISHED"


def test_give_up_on_unknown_game_returns_game_not_found(client: TestClient) -> None:
    response = _give_up(client, "00000000-0000-4000-8000-000000000000")
    body = response.json()

    assert response.status_code == 404
    assert body["code"] == "GAME_NOT_FOUND"
    assert set(body) == {"code", "message", "details"}
    _assert_answer_absent(response.text)


def test_give_up_on_a_won_game_is_rejected_with_conflict(client: TestClient) -> None:
    game_id = _create_game(client)
    _guess(client, game_id, TEST_ANSWER)

    response = _give_up(client, game_id)
    body = response.json()

    assert response.status_code == 409
    assert body["code"] == "GAME_ALREADY_FINISHED"
    assert set(body) == {"code", "message", "details"}


def test_give_up_on_a_won_game_does_not_change_its_status(client: TestClient) -> None:
    """A rejected give-up must not rewrite how the game actually ended."""
    game_id = _create_game(client)
    _guess(client, game_id, TEST_ANSWER)

    _give_up(client, game_id)

    assert client.get(f"/api/games/{game_id}").json()["status"] == "won"


def test_second_give_up_is_rejected_with_conflict(client: TestClient) -> None:
    game_id = _create_game(client)

    first = _give_up(client, game_id)
    second = _give_up(client, game_id)

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["code"] == "GAME_ALREADY_FINISHED"
    assert set(second.json()) == {"code", "message", "details"}


def test_give_up_does_not_log_the_answer(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The reveal is a response, not a log line (AGENTS.md)."""
    game_id = _create_game(client)

    with caplog.at_level(logging.DEBUG):
        _give_up(client, game_id)

    _assert_answer_absent(caplog.text)


# --- Answer secrecy regression: give-up is the only new reveal -------------


def test_the_new_endpoint_does_not_reveal_answers_early(client: TestClient) -> None:
    """The create/get contract is unchanged by this feature.

    Walks a *second*, untouched game through the in-progress flow while another
    game has been given up, so a reveal leaking through shared state — a cached
    response model, a mutated default — would fail here.
    """
    given_up_id = _create_game(client)
    _give_up(client, given_up_id)

    create = client.post("/api/games")
    playing_id = create.json()["gameId"]

    assert "answer" not in create.json()
    _assert_answer_absent(create.text)

    guess = _guess(client, playing_id, WRONG_WORD)
    assert "answer" not in guess.json()
    _assert_answer_absent(guess.text)

    state = client.get(f"/api/games/{playing_id}")
    assert state.json()["status"] == "playing"
    assert state.json()["answer"] is None
    _assert_answer_absent(state.text)


def test_the_give_up_mapper_refuses_to_reveal_a_playing_game() -> None:
    """The reveal is guarded by the game's status, not by the caller's promise.

    `to_give_up_response` is the only new place the answer word can be read into
    a response, so it must fail closed if it is ever reached with a game that has
    not finished — and fail without naming the word.
    """
    playing = Game(
        id="game-1",
        answer=TEST_ANSWER,
        created_at=datetime(2026, 7, 24, 11, 22, 57, tzinfo=UTC),
    )

    with pytest.raises(ValueError) as caught:
        to_give_up_response(playing)

    _assert_answer_absent(str(caught.value))
