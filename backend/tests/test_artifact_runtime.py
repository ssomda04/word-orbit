"""The production path, end to end, with no model anywhere in the process.

    POST /api/games/{id}/guesses
      -> GameService -> ArtifactGuessScorer -> ArtifactStore -> the root on disk
      -> similarity + rank -> GuessResponse

`tests/test_games_api.py` already pins the wire shape, idempotency, the finished
-game conflict and the reveal rule for embedding mode, and none of that is
re-asserted here for its own sake. What *is* re-asserted is the handful of those
behaviours that route through the new scorer, because a scorer that can return
"no score" is a new way for each of them to break.

The three things genuinely new in this mode get the most attention:

- the answer is drawn from the artifact root, not from `ANSWER_WORDS`;
- a guess outside the artifact's vocabulary is `400 INVALID_WORD`, not a 500;
- the embedding stack is never built — proven twice, once by watching for the
  calls and once by configuring a model path that would fail if one were made.

Every test runs on the synthetic root from `tests/artifact_fixture.py`: no
FastText, no downloaded data, no `ml` import.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.domain.words import ANSWER_WORDS
from app.services.game import InMemoryGameRepository
from app.services.scoring.artifact import load_manifest
from tests import artifact_fixture as fixture
from tests.conftest import artifact_app

# Deliberately disjoint from `ANSWER_WORDS`: if a game's answer is one of these,
# it can only have come from the manifest. The fixture's own vocabulary shares
# every word with the placeholder list, which would make that unprovable.
VOCABULARY: tuple[str, ...] = ("바람", "구름", "나무", "호수", "모래", "그림자")
ANSWERS: tuple[str, ...] = ("바람", "구름", "나무")

# In the vocabulary, never an answer — safe to guess in any game.
IN_VOCABULARY = "그림자"
# Outside the artifact entirely. A live model would happily score it.
OUT_OF_VOCABULARY = "존재하지않는단어"


@pytest.fixture
def root(tmp_path: Path) -> Path:
    target = tmp_path / "artifacts-root"
    fixture.write_root(target, vocabulary=VOCABULARY, answers=ANSWERS)
    return target


@pytest.fixture
def wiring(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> tuple[FastAPI, InMemoryGameRepository]:
    return artifact_app(monkeypatch, root)


@pytest.fixture
def client(wiring: tuple[FastAPI, InMemoryGameRepository]) -> Iterator[TestClient]:
    application, _ = wiring
    with TestClient(application) as running:
        yield running


@pytest.fixture
def repository(wiring: tuple[FastAPI, InMemoryGameRepository]) -> InMemoryGameRepository:
    """The store the app writes to — the only way to read an unrevealed answer."""
    return wiring[1]


def _create_game(client: TestClient) -> str:
    response = client.post("/api/games")
    assert response.status_code == 200
    return response.json()["gameId"]


def _guess(client: TestClient, game_id: str, word: str):
    return client.post(f"/api/games/{game_id}/guesses", json={"word": word})


def _answer_of(repository: InMemoryGameRepository, game_id: str) -> str:
    game = repository.get(game_id)
    assert game is not None
    return game.answer


# --- Where the answer comes from ---------------------------------------------


def test_a_new_games_answer_comes_from_the_manifest(
    client: TestClient, repository: InMemoryGameRepository, root: Path
) -> None:
    answers = set(load_manifest(root).answers)

    for _ in range(20):
        game_id = _create_game(client)
        assert _answer_of(repository, game_id) in answers


def test_the_placeholder_word_list_is_not_used(
    client: TestClient, repository: InMemoryGameRepository
) -> None:
    """The point of the whole selector change, stated as one assertion."""
    for _ in range(20):
        game_id = _create_game(client)
        assert _answer_of(repository, game_id) not in ANSWER_WORDS


def test_every_manifest_answer_can_actually_be_chosen(
    client: TestClient, repository: InMemoryGameRepository
) -> None:
    """A selector drawing from one entry would pass the tests above."""
    seen = {_answer_of(repository, _create_game(client)) for _ in range(200)}

    assert seen == set(ANSWERS)


# --- Scoring a guess ---------------------------------------------------------


def test_a_vocabulary_guess_is_scored_from_the_artifact(client: TestClient) -> None:
    game_id = _create_game(client)

    response = _guess(client, game_id, IN_VOCABULARY)
    body: Any = response.json()

    assert response.status_code == 200
    assert set(body) == {"guessId", "word", "similarity", "rank", "isAnswer", "coordinate"}
    assert -1.0 <= body["similarity"] <= 1.0
    assert body["isAnswer"] is False
    assert body["coordinate"] is None


def test_a_vocabulary_guess_always_has_a_rank(client: TestClient) -> None:
    """Unlike embedding mode: an artifact holds a rank for every word it holds."""
    game_id = _create_game(client)

    body: Any = _guess(client, game_id, IN_VOCABULARY).json()

    assert isinstance(body["rank"], int)
    assert 2 <= body["rank"] <= len(VOCABULARY)


def test_the_scored_values_are_the_stored_ones(
    client: TestClient, repository: InMemoryGameRepository, root: Path
) -> None:
    """Not merely plausible numbers — the exact cells the fixture wrote."""
    game_id = _create_game(client)
    answer = _answer_of(repository, game_id)
    index = VOCABULARY.index(IN_VOCABULARY)
    expected_similarity = fixture.read_similarity(root, answer)[index]
    expected_rank = fixture.read_rank(root, answer)[index]

    body: Any = _guess(client, game_id, IN_VOCABULARY).json()

    assert body["similarity"] == pytest.approx(float(expected_similarity))
    assert body["rank"] == int(expected_rank)


def test_a_duplicate_guess_is_idempotent(client: TestClient) -> None:
    game_id = _create_game(client)

    first: Any = _guess(client, game_id, IN_VOCABULARY).json()
    second: Any = _guess(client, game_id, IN_VOCABULARY).json()
    state: Any = client.get(f"/api/games/{game_id}").json()

    assert second == first
    assert state["guessCount"] == 1


# --- Out of vocabulary -------------------------------------------------------


def test_a_guess_outside_the_vocabulary_is_a_bad_request(client: TestClient) -> None:
    game_id = _create_game(client)

    response = _guess(client, game_id, OUT_OF_VOCABULARY)

    assert response.status_code == 400
    assert response.json()["code"] == "INVALID_WORD"


def test_an_out_of_vocabulary_guess_is_not_an_internal_error(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The failure mode this contract exists to prevent."""
    game_id = _create_game(client)

    with caplog.at_level("DEBUG"):
        response = _guess(client, game_id, OUT_OF_VOCABULARY)

    assert response.json()["code"] != "INTERNAL_ERROR"
    assert not any(record.exc_info for record in caplog.records)


def test_a_rejected_guess_is_not_recorded(client: TestClient) -> None:
    game_id = _create_game(client)

    _guess(client, game_id, OUT_OF_VOCABULARY)
    state: Any = client.get(f"/api/games/{game_id}").json()

    assert state["guessCount"] == 0
    assert state["status"] == "playing"


def test_the_error_envelope_is_the_documented_one(client: TestClient) -> None:
    """The frontend already handles this code; artifact mode must not invent one."""
    game_id = _create_game(client)

    body: Any = _guess(client, game_id, OUT_OF_VOCABULARY).json()

    assert set(body) == {"code", "message", "details"}
    assert body["code"] == "INVALID_WORD"
    assert body["message"]


# --- Winning -----------------------------------------------------------------


def test_guessing_the_answer_wins_with_a_perfect_score(
    client: TestClient, repository: InMemoryGameRepository
) -> None:
    game_id = _create_game(client)
    answer = _answer_of(repository, game_id)

    body: Any = _guess(client, game_id, answer).json()

    assert body["similarity"] == 1.0
    assert body["rank"] == 1
    assert body["isAnswer"] is True


def test_the_game_is_won_and_reveals_the_answer(
    client: TestClient, repository: InMemoryGameRepository
) -> None:
    game_id = _create_game(client)
    answer = _answer_of(repository, game_id)

    _guess(client, game_id, answer)
    state: Any = client.get(f"/api/games/{game_id}").json()

    assert state["status"] == "won"
    assert state["answer"] == answer


def test_a_finished_game_still_rejects_further_guesses(
    client: TestClient, repository: InMemoryGameRepository
) -> None:
    game_id = _create_game(client)

    _guess(client, game_id, _answer_of(repository, game_id))
    response = _guess(client, game_id, IN_VOCABULARY)

    assert response.status_code == 409
    assert response.json()["code"] == "GAME_ALREADY_FINISHED"


def test_a_finished_game_rejects_an_out_of_vocabulary_guess_as_a_conflict(
    client: TestClient, repository: InMemoryGameRepository
) -> None:
    """Order of checks: finished beats unscorable, as it beats duplicate."""
    game_id = _create_game(client)

    _guess(client, game_id, _answer_of(repository, game_id))
    response = _guess(client, game_id, OUT_OF_VOCABULARY)

    assert response.status_code == 409


# --- Game state --------------------------------------------------------------


def test_game_state_returns_the_history_in_submission_order(client: TestClient) -> None:
    game_id = _create_game(client)
    _guess(client, game_id, IN_VOCABULARY)
    _guess(client, game_id, VOCABULARY[0])

    state: Any = client.get(f"/api/games/{game_id}").json()

    assert state["guessCount"] == 2
    assert [guess["word"] for guess in state["guesses"]] == [IN_VOCABULARY, VOCABULARY[0]]


def test_an_in_progress_game_never_reveals_its_answer(
    client: TestClient, repository: InMemoryGameRepository
) -> None:
    game_id = _create_game(client)
    answer = _answer_of(repository, game_id)
    _guess(client, game_id, IN_VOCABULARY)

    state = client.get(f"/api/games/{game_id}")

    assert state.json()["answer"] is None
    assert answer not in state.text
    assert answer.encode("unicode_escape").decode() not in state.text


def test_creating_a_game_never_returns_an_answer_field(client: TestClient) -> None:
    response = client.post("/api/games")

    assert "answer" not in response.json()


# --- The embedding stack is not built ----------------------------------------


@pytest.fixture
def embedding_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any use of the embedding or ranking factories fail loudly.

    Patched in every namespace that imports them, because that is what "is not
    called" has to mean: `app.api.deps` reaches them per request, `app.main`
    reaches them at startup, and a name bound at import time is not affected by
    patching the factory module alone.
    """

    def _forbidden(*_: object, **__: object) -> object:
        raise AssertionError("the embedding path must not be reached in artifact mode")

    for module in ("app.api.deps", "app.main"):
        monkeypatch.setattr(f"{module}.get_embedding_service", _forbidden)
        monkeypatch.setattr(f"{module}.get_rank_provider", _forbidden)


def test_neither_startup_nor_a_guess_touches_the_embedding_factories(
    monkeypatch: pytest.MonkeyPatch, root: Path, embedding_tripwire: None
) -> None:
    application, repository = artifact_app(monkeypatch, root)

    with TestClient(application) as client:
        game_id = _create_game(client)
        scored = _guess(client, game_id, IN_VOCABULARY)
        rejected = _guess(client, game_id, OUT_OF_VOCABULARY)
        won = _guess(client, game_id, _answer_of(repository, game_id))
        state = client.get(f"/api/games/{game_id}")

    assert scored.status_code == 200
    assert rejected.status_code == 400
    assert won.json()["isAnswer"] is True
    assert state.json()["status"] == "won"


def test_a_model_that_could_not_load_does_not_stop_artifact_mode(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    """The black-box half of the proof, with no patching at all.

    `EMBEDDING_PROVIDER=fasttext` with no model path is a configuration that
    fails the instant anything asks for an embedding service. A full game played
    on top of it is a statement that nothing did.
    """
    application, repository = artifact_app(
        monkeypatch, root, EMBEDDING_PROVIDER="fasttext", FASTTEXT_MODEL_PATH=""
    )

    with TestClient(application) as client:
        game_id = _create_game(client)
        scored = _guess(client, game_id, IN_VOCABULARY)
        won = _guess(client, game_id, _answer_of(repository, game_id))

    assert scored.status_code == 200
    assert isinstance(scored.json()["rank"], int)
    assert won.json()["similarity"] == 1.0


def test_arrays_are_read_on_the_first_guess_and_not_before(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    """Laziness is a runtime property, not only a startup one."""
    from app.services.scoring.artifact import store as store_module

    loaded: list[str] = []
    real = store_module.load_answer

    def _counting_load(manifest, entry):
        loaded.append(entry.artifact_id)
        return real(manifest, entry)

    monkeypatch.setattr(store_module, "load_answer", _counting_load)
    application, _ = artifact_app(monkeypatch, root)

    with TestClient(application) as client:
        assert loaded == [], "startup must read no arrays"
        game_id = _create_game(client)
        assert loaded == [], "creating a game must read no arrays"

        _guess(client, game_id, IN_VOCABULARY)
        assert len(loaded) == 1

        _guess(client, game_id, VOCABULARY[0])
        assert len(loaded) == 1, "a second guess in the same game must hit the cache"
