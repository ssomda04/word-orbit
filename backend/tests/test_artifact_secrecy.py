"""The answer stays hidden through the artifact runtime path too.

`tests/test_answer_secrecy.py` pins this rule for the embedding path, and
`tests/test_artifact_answer.py` pins it for the reader in isolation — including
every malformed `.npy` numpy is willing to quote a file back over. Neither is
repeated here.

What is *not* covered by either is the join: a reader that sanitises correctly
still tells you nothing about what the server does with the result once it is
wrapped in a store, a scorer, a service and a FastAPI handler. Each of those
layers re-raises, chains, or logs, and any one of them could put back what the
reader took out.

So this module runs one representative secret-bearing failure — a malformed
array whose header carries the answer word, which is the one case where the
secret comes from *outside* this repository's control — all the way through the
real production wiring, and reads back the response, the log, and the rendered
traceback. One case is enough because the sanitising happens in one place; what
is being tested here is the wiring above it, and the wiring does not care which
malformed header it was.
"""

import logging
import traceback
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.services.scoring.artifact import (
    ArtifactError,
    ArtifactGuessScorer,
    ArtifactStore,
    artifact_id_for,
    load_manifest,
)
from tests import artifact_fixture as fixture
from tests.conftest import artifact_app

ANSWER = fixture.ANSWER
GUESS = "학생"

# A header numpy cannot parse, carrying the answer inside the file's *contents*.
# Hash-only paths keep the word out of filenames; they say nothing about this.
LEAKY_HEADER = "{{'descr': '<f4', 'fortran_order': False, 'shape': (6,), 'note': {secret!r}"

# What the sanitized failure must still say, so redaction is not mistaken for a
# test that would also pass if the log said nothing useful at all.
SANITIZED_MARKER = "Could not load artifact"


def _forbidden_forms(word: str) -> tuple[str, ...]:
    """Every spelling the word could survive as: raw, repr, and escaped."""
    return (word, repr(word), word.encode("unicode_escape").decode())


def _assert_secret_free(haystack: str, word: str = ANSWER) -> None:
    for form in _forbidden_forms(word):
        assert form not in haystack


def _rendered_traceback(exc: BaseException) -> str:
    """The traceback exactly as a logging handler renders it, chain included."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A one-answer root, so the game's answer is known and deterministic."""
    target = tmp_path / "artifacts-root"
    fixture.write_root(target)
    return target


@pytest.fixture
def leaky_root(root: Path) -> Path:
    """The same root with the answer's similarity array replaced by a trap."""
    fixture.write_npy_with_header(
        root, ANSWER, "similarity.npy", LEAKY_HEADER.format(secret=ANSWER)
    )
    return root


# --- Through the store and the scorer ----------------------------------------


def test_a_failing_load_through_the_store_never_names_the_answer(
    leaky_root: Path,
) -> None:
    """The store adds a cache, and a cache must not add a place to say the word."""
    store = ArtifactStore(load_manifest(leaky_root))
    entry = store.manifest.entry_for(ANSWER)
    assert entry is not None

    with pytest.raises(ArtifactError) as caught:
        store.get(entry)

    _assert_secret_free(_rendered_traceback(caught.value))
    assert caught.value.__cause__ is None


def test_a_failing_score_through_the_scorer_never_names_the_answer(
    leaky_root: Path,
) -> None:
    scorer = ArtifactGuessScorer(ArtifactStore(load_manifest(leaky_root)))

    with pytest.raises(ArtifactError) as caught:
        scorer.score(ANSWER, GUESS)

    rendered = _rendered_traceback(caught.value)
    _assert_secret_free(rendered)
    # Redaction, not silence: an operator can still find the file.
    assert SANITIZED_MARKER in rendered
    assert artifact_id_for(ANSWER) in rendered


def test_an_answer_the_root_cannot_serve_is_reported_by_id(root: Path) -> None:
    """The scorer's own error, rather than one it passes along.

    A selector and a root that disagree is a misconfiguration, and the answer is
    still a secret while the game is in progress — so the mismatch is named by
    the identifier that is already the directory name on disk.
    """
    scorer = ArtifactGuessScorer(ArtifactStore(load_manifest(root)))
    unknown = "학생"  # in the vocabulary, but not an answer of this root

    with pytest.raises(ArtifactError) as caught:
        scorer.score(unknown, GUESS)

    _assert_secret_free(str(caught.value), unknown)
    assert artifact_id_for(unknown) in str(caught.value)


# --- End to end, through the real application --------------------------------


@pytest.fixture
def failing_client(
    monkeypatch: pytest.MonkeyPatch, leaky_root: Path
) -> Iterator[TestClient]:
    """A real app whose real artifact root breaks on the real hidden answer."""
    application, _ = artifact_app(monkeypatch, leaky_root)
    with TestClient(application, raise_server_exceptions=False) as client:
        yield client


def test_the_broken_root_still_starts_the_server(failing_client: TestClient) -> None:
    """The premise: arrays are lazy, so this failure can only happen mid-game.

    A root whose *manifest* is broken never gets this far — startup rejects it
    (`tests/test_scoring_factory.py`). This one is valid right up to the bytes
    of one array, which is exactly the case that reaches a request.
    """
    assert failing_client.post("/api/games").status_code == 200


def test_the_answer_is_absent_from_everything_logged(
    failing_client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """A 500 mid-game must not print the hidden word."""
    game_id = failing_client.post("/api/games").json()["gameId"]

    with caplog.at_level(logging.DEBUG):
        response = failing_client.post(
            f"/api/games/{game_id}/guesses", json={"word": GUESS}
        )

    assert response.status_code == 500
    assert any(record.exc_info for record in caplog.records), (
        "a traceback must actually have been logged, or this test proves nothing"
    )
    assert SANITIZED_MARKER in caplog.text
    _assert_secret_free(caplog.text)


def test_the_response_carries_the_documented_envelope_and_nothing_else(
    failing_client: TestClient,
) -> None:
    game_id = failing_client.post("/api/games").json()["gameId"]

    response = failing_client.post(
        f"/api/games/{game_id}/guesses", json={"word": GUESS}
    )
    body: Any = response.json()

    assert body["code"] == "INTERNAL_ERROR"
    assert body["details"] is None
    _assert_secret_free(response.text)
    assert SANITIZED_MARKER not in response.text


def test_the_game_is_not_advanced_by_the_failure(failing_client: TestClient) -> None:
    """A failed guess must leave nothing behind that could reveal the answer."""
    game_id = failing_client.post("/api/games").json()["gameId"]

    failing_client.post(f"/api/games/{game_id}/guesses", json={"word": GUESS})
    state = failing_client.get(f"/api/games/{game_id}")

    assert state.json()["status"] == "playing"
    assert state.json()["guessCount"] == 0
    assert state.json()["answer"] is None
    _assert_secret_free(state.text)


# --- The successful path is clean too ----------------------------------------


def test_a_healthy_game_logs_nothing_that_names_the_answer(
    monkeypatch: pytest.MonkeyPatch, root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Secrecy is not only a property of failures."""
    application, _ = artifact_app(monkeypatch, root)

    with caplog.at_level(logging.DEBUG), TestClient(application) as client:
        game_id = client.post("/api/games").json()["gameId"]
        scored = client.post(f"/api/games/{game_id}/guesses", json={"word": GUESS})
        state = client.get(f"/api/games/{game_id}")

    assert scored.status_code == 200
    _assert_secret_free(caplog.text)
    _assert_secret_free(scored.text)
    _assert_secret_free(state.text)
