"""Provider selection, artifact-root validation, and what startup does with them.

Two rules are worth stating up front, because most of this module exists to pin
them:

- **The default is the behaviour that already existed.** A process with no
  `SCORING_PROVIDER` set scores exactly as it did before artifact mode existed.
- **A bad artifact root stops the server.** Not a 500 on the first guess: a
  failed startup, the same way a bad `FASTTEXT_MODEL_PATH` already fails.

"Bad" includes one case the reader cannot see. A manifest key must be a
canonical *vocabulary* word, and the vocabulary policy deliberately allows words
the *guess* policy rejects — a headword with a space, a word past the length cap.
An answer like that loads fine and then breaks `POST /api/games`, on the draws
that happen to pick it. The final section is about refusing such a root outright.

The last section covers a third rule that is easy to lose: **an app serves the
configuration it was created from**. `create_app` accepts an explicit `Settings`,
and every artifact-mode dependency has to be bound to *that* object rather than
reaching for the global one, or an app can select artifact mode from one root and
serve answers and scores from another.

`tests/test_artifact_runtime.py` covers what a *good* root then serves.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.deps import game_repository
from app.core.config import Settings, get_settings
from app.core.errors import InvalidWordError
from app.domain.game import MAX_WORD_LENGTH, normalize_word
from app.main import create_app
from app.services.game import InMemoryGameRepository
from app.services.scoring import (
    ArtifactStore,
    ScoringProvider,
    get_artifact_store,
    reset_artifact_store,
    resolve_scoring_provider,
)
from app.services.scoring.artifact import ArtifactError
from tests import artifact_fixture as fixture
from tests.conftest import configure_environment


@pytest.fixture
def root(tmp_path: Path) -> Path:
    target = tmp_path / "artifacts-root"
    fixture.write_root(target)
    return target


# --- Provider selection ------------------------------------------------------


def test_the_default_provider_is_embedding() -> None:
    """Nothing configured must mean nothing changed."""
    assert resolve_scoring_provider() is ScoringProvider.EMBEDDING


@pytest.mark.parametrize("value", ["embedding", "EMBEDDING", "  Embedding  "])
def test_embedding_is_selected_by_name(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    configure_environment(monkeypatch, SCORING_PROVIDER=value)

    assert resolve_scoring_provider() is ScoringProvider.EMBEDDING


@pytest.mark.parametrize("value", ["artifact", "ARTIFACT", "  Artifact  "])
def test_artifact_is_selected_by_name(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    configure_environment(monkeypatch, SCORING_PROVIDER=value)

    assert resolve_scoring_provider() is ScoringProvider.ARTIFACT


def test_an_unknown_provider_is_refused_rather_than_defaulted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo must not quietly start a server that scores through the wrong data."""
    configure_environment(monkeypatch, SCORING_PROVIDER="artifacts")

    with pytest.raises(ValueError) as excinfo:
        resolve_scoring_provider()

    message = str(excinfo.value)
    assert "artifacts" in message
    assert "embedding" in message and "artifact" in message


def test_an_explicit_settings_object_is_honoured() -> None:
    """`create_app(settings=...)` must select from the object it was handed."""
    assert (
        resolve_scoring_provider(Settings(scoring_provider="artifact"))
        is ScoringProvider.ARTIFACT
    )


# --- Cache size --------------------------------------------------------------


def test_the_default_cache_size_is_positive() -> None:
    assert get_settings().artifact_cache_size > 0


@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_non_positive_cache_size_is_refused_by_settings(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Validity here does not depend on the mode, so it belongs on Settings."""
    monkeypatch.setenv("ARTIFACT_CACHE_SIZE", value)
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="artifact_cache_size"):
        get_settings()


def test_a_non_integer_cache_size_is_refused_by_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARTIFACT_CACHE_SIZE", "many")
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="artifact_cache_size"):
        get_settings()


def test_the_configured_cache_size_reaches_the_store(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    configure_environment(
        monkeypatch,
        SCORING_PROVIDER="artifact",
        ARTIFACT_ROOT=str(root),
        ARTIFACT_CACHE_SIZE="1",
    )

    store = get_artifact_store(get_settings())
    for answer in store.answers:
        entry = store.manifest.entry_for(answer)
        assert entry is not None
        store.get(entry)

    assert store.cached_artifact_count == 1


# --- Building the store ------------------------------------------------------


def test_a_valid_root_builds_a_store(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root)
    )

    store = get_artifact_store(get_settings())

    assert isinstance(store, ArtifactStore)
    assert store.answers == (fixture.ANSWER,)


def test_the_store_is_built_once_per_process(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root)
    )

    assert get_artifact_store(get_settings()) is get_artifact_store(get_settings())


def test_reset_forces_a_rebuild(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root)
    )
    first = get_artifact_store(get_settings())

    reset_artifact_store()

    assert get_artifact_store(get_settings()) is not first


def test_a_missing_artifact_root_fails_with_actionable_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_environment(monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT="")

    with pytest.raises(ArtifactError) as excinfo:
        get_artifact_store(get_settings())

    message = str(excinfo.value)
    assert "ARTIFACT_ROOT" in message
    assert "never committed" in message


def test_a_blank_artifact_root_is_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_environment(monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT="   ")

    with pytest.raises(ArtifactError, match="ARTIFACT_ROOT"):
        get_artifact_store(get_settings())


def test_a_nonexistent_artifact_root_names_the_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "nowhere"
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(missing)
    )

    with pytest.raises(ArtifactError) as excinfo:
        get_artifact_store(get_settings())

    assert str(missing) in str(excinfo.value)


def test_a_file_instead_of_a_root_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{}", encoding="utf-8")
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(path)
    )

    with pytest.raises(ArtifactError, match="not a directory"):
        get_artifact_store(get_settings())


def test_a_malformed_manifest_is_refused(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    (root / "manifest.json").write_text("{not json", encoding="utf-8")
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root)
    )

    with pytest.raises(ArtifactError, match="not valid JSON"):
        get_artifact_store(get_settings())


def test_a_root_missing_an_array_file_is_refused(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    """Existence is checked at startup even though contents are not read."""
    (root / fixture.similarity_relative_path(fixture.ANSWER)).unlink()
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root)
    )

    with pytest.raises(ArtifactError, match="missing file"):
        get_artifact_store(get_settings())


# --- Startup -----------------------------------------------------------------


def test_startup_validates_the_root_before_any_request(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root)
    )

    # Entering the context manager runs the lifespan; no request has been sent.
    with TestClient(create_app()) as client:
        assert get_artifact_store(get_settings()).answers == (fixture.ANSWER,)

        assert client.get("/health").status_code == 200


def test_startup_does_not_read_any_answer_array(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    """The whole point of the lazy store: startup cost is flat in answer count."""
    from app.services.scoring.artifact import store as store_module

    loaded: list[str] = []
    real = store_module.load_answer
    monkeypatch.setattr(
        store_module,
        "load_answer",
        lambda manifest, entry: (loaded.append(entry.artifact_id), real(manifest, entry))[1],
    )
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root)
    )

    with TestClient(create_app()):
        assert loaded == []


@pytest.mark.parametrize("failure", ["unset", "nonexistent", "malformed"])
def test_startup_fails_loudly_on_a_bad_root(
    monkeypatch: pytest.MonkeyPatch, root: Path, tmp_path: Path, failure: str
) -> None:
    """A misconfigured root must stop the server, not surface as a runtime 500."""
    if failure == "unset":
        configured = ""
    elif failure == "nonexistent":
        configured = str(tmp_path / "nowhere")
    else:
        (root / "manifest.json").write_text("{not json", encoding="utf-8")
        configured = str(root)

    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=configured
    )

    with pytest.raises(ArtifactError), TestClient(create_app()):
        pass  # pragma: no cover - startup raises before the body runs


def test_an_unknown_provider_fails_before_the_app_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_environment(monkeypatch, SCORING_PROVIDER="nonsense")

    with pytest.raises(ValueError, match="SCORING_PROVIDER"):
        create_app()


def test_embedding_mode_startup_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression guard: the default path still warms what it always did."""
    configure_environment(monkeypatch, SCORING_PROVIDER="embedding")

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/dev/similarity", json={"first": "학생", "second": "선생"}
        )

    assert response.status_code == 200


# --- The answer pool has to be playable --------------------------------------

# Both are legal canonical vocabulary entries and illegal guesses. The spaced one
# is not contrived: `app.domain.vocabulary` names `"고유 명사"` as exactly the kind
# of headword the vocabulary must keep.
SPACED_ANSWER = "고유 명사"
OVERLONG_ANSWER = "가" * (MAX_WORD_LENGTH + 1)
UNPLAYABLE_ANSWERS = (SPACED_ANSWER, OVERLONG_ANSWER)


def _root_with_answer(target: Path, answer: str) -> Path:
    """A valid root offering `answer` alongside two ordinary ones."""
    fixture.write_root(
        target,
        vocabulary=(answer, "바다", "하늘"),
        answers=(answer, "바다"),
    )
    return target


def _forbidden_forms(word: str) -> tuple[str, ...]:
    return (word, repr(word), word.encode("unicode_escape").decode())


@pytest.mark.parametrize("answer", UNPLAYABLE_ANSWERS)
def test_the_reader_alone_would_accept_an_unplayable_answer(
    tmp_path: Path, answer: str
) -> None:
    """The premise. If this ever fails, the check below has become redundant."""
    from app.services.scoring.artifact import load_manifest

    root = _root_with_answer(tmp_path / "artifacts-root", answer)

    assert answer in load_manifest(root).answers
    with pytest.raises(InvalidWordError):
        normalize_word(answer)


@pytest.mark.parametrize("answer", UNPLAYABLE_ANSWERS)
def test_an_unplayable_answer_is_refused_at_build_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, answer: str
) -> None:
    root = _root_with_answer(tmp_path / "artifacts-root", answer)
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root)
    )

    with pytest.raises(ArtifactError, match="cannot set a game on"):
        get_artifact_store(get_settings())


@pytest.mark.parametrize("answer", UNPLAYABLE_ANSWERS)
def test_an_unplayable_answer_fails_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, answer: str
) -> None:
    """The whole point: fail here, not on an unlucky `POST /api/games`."""
    root = _root_with_answer(tmp_path / "artifacts-root", answer)
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root)
    )

    with pytest.raises(ArtifactError), TestClient(create_app()):
        pass  # pragma: no cover - startup raises before the body runs


@pytest.mark.parametrize("answer", UNPLAYABLE_ANSWERS)
def test_the_refusal_never_names_the_answer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    answer: str,
) -> None:
    """A rejected answer is still an answer of some root; it stays a secret.

    This also pins `normalize_word`'s messages: adding the offending word to one
    of them would surface here rather than in production.
    """
    import traceback

    root = _root_with_answer(tmp_path / "artifacts-root", answer)
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root)
    )

    with caplog.at_level("DEBUG"), pytest.raises(ArtifactError) as caught:
        get_artifact_store(get_settings())

    rendered = "".join(
        traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__)
    )
    for form in _forbidden_forms(answer):
        assert form not in rendered
        assert form not in caplog.text
    assert caught.value.__cause__ is None


def test_the_refusal_still_says_which_artifact_and_which_rule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Redaction must not cost an operator the ability to act on the failure."""
    from app.services.scoring.artifact import artifact_id_for

    root = _root_with_answer(tmp_path / "artifacts-root", SPACED_ANSWER)
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root)
    )

    with pytest.raises(ArtifactError) as caught:
        get_artifact_store(get_settings())

    message = str(caught.value)
    assert artifact_id_for(SPACED_ANSWER) in message
    assert "공백" in message, "the operator needs to know which rule was broken"


def test_one_unplayable_answer_condemns_the_whole_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No silent filtering: a partly-usable root is a root to rebuild."""
    root = tmp_path / "artifacts-root"
    fixture.write_root(
        root,
        vocabulary=(SPACED_ANSWER, "바다", "하늘", "학생"),
        answers=("바다", "하늘", SPACED_ANSWER),
    )
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root)
    )

    with pytest.raises(ArtifactError, match="cannot set a game on"):
        get_artifact_store(get_settings())


def test_ordinary_answers_still_build_a_store(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    """The check must not reject the roots it is not about."""
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root)
    )

    assert get_artifact_store(get_settings()).answers == (fixture.ANSWER,)


def test_an_accepted_answer_is_returned_unchanged_by_the_guess_normalizer(
    monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    """The invariant the build check relies on instead of re-asserting it.

    Both normalizers apply NFKC-then-strip, and the reader has already required
    each key to be canonical under the vocabulary one — so a key the guess rules
    accept is a fixed point of them. That is what makes the answer stored on a
    `Game` equal to the manifest key, and therefore findable and guessable.
    """
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root)
    )

    for answer in get_artifact_store(get_settings()).answers:
        assert normalize_word(answer) == answer


# --- One app, one configuration ----------------------------------------------

# Two roots that cannot be mistaken for one another. The answer sets are
# disjoint, so which root chose a game's answer is visible; `SHARED_GUESS` sits
# at a different offset in each vocabulary, so which root *scored* a guess is
# visible too, from a value rather than from an absence.
SHARED_GUESS = "공통단어"
ROOT_A_VOCABULARY = ("가가가", SHARED_GUESS, "나나나", "다다다")
ROOT_A_ANSWERS = ("가가가", "나나나")
ROOT_B_VOCABULARY = ("라라라", "마마마", "바바바", "사사사", SHARED_GUESS)
ROOT_B_ANSWERS = ("라라라", "마마마")


def _artifact_settings(root: Path, **overrides: object) -> Settings:
    """An explicit Settings for `root`, independent of the environment."""
    return Settings(scoring_provider="artifact", artifact_root=str(root), **overrides)


@pytest.fixture
def root_a(tmp_path: Path) -> Path:
    target = tmp_path / "root-a"
    fixture.write_root(target, vocabulary=ROOT_A_VOCABULARY, answers=ROOT_A_ANSWERS)
    return target


@pytest.fixture
def root_b(tmp_path: Path) -> Path:
    target = tmp_path / "root-b"
    fixture.write_root(target, vocabulary=ROOT_B_VOCABULARY, answers=ROOT_B_ANSWERS)
    return target


def _stored_similarity(root: Path, answer: str, vocabulary: tuple[str, ...]) -> float:
    """The similarity `root` actually holds for `SHARED_GUESS` under `answer`."""
    return float(fixture.read_similarity(root, answer)[vocabulary.index(SHARED_GUESS)])


def _play(app, rounds: int = 12) -> tuple[set[str], float]:
    """Create `rounds` games on `app`, returning the answers used and one score.

    The repository override is the only way to read an answer the API is not
    allowed to reveal while a game is in progress.
    """
    repository = InMemoryGameRepository()
    app.dependency_overrides[game_repository] = lambda: repository
    answers: set[str] = set()

    with TestClient(app) as client:
        for _ in range(rounds):
            game_id = client.post("/api/games").json()["gameId"]
            game = repository.get(game_id)
            assert game is not None
            answers.add(game.answer)

        last_id = client.post("/api/games").json()["gameId"]
        guess = client.post(f"/api/games/{last_id}/guesses", json={"word": SHARED_GUESS})
        assert guess.status_code == 200, guess.text
        last = repository.get(last_id)
        assert last is not None
        answers.add(last.answer)

    return answers, guess.json()["similarity"]


def test_explicit_settings_beat_the_environment(
    monkeypatch: pytest.MonkeyPatch, root_a: Path, root_b: Path
) -> None:
    """A. `create_app(settings=X)` serves X, whatever the environment says.

    The environment is pointed at root B and left there. Everything the app does
    — validating at startup, choosing an answer, scoring a guess — must come from
    root A, because root A is what `create_app` was handed.
    """
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root_b)
    )

    answers, similarity = _play(create_app(settings=_artifact_settings(root_a)))

    assert answers <= set(ROOT_A_ANSWERS)
    assert not answers & set(ROOT_B_ANSWERS)
    assert similarity == pytest.approx(
        _stored_similarity(root_a, next(iter(answers)), ROOT_A_VOCABULARY)
    )


def test_the_environments_root_is_not_even_read(
    monkeypatch: pytest.MonkeyPatch, root_a: Path, tmp_path: Path
) -> None:
    """The sharpest form of A: the environment names a root that cannot load.

    A broken root fails startup, so an app that starts at all is one that never
    looked at it.
    """
    broken = tmp_path / "root-broken"
    fixture.write_root(broken)
    (broken / "manifest.json").write_text("{not json", encoding="utf-8")
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(broken)
    )

    answers, _ = _play(create_app(settings=_artifact_settings(root_a)), rounds=4)

    assert answers <= set(ROOT_A_ANSWERS)


def test_two_apps_with_two_roots_do_not_share_a_store(
    monkeypatch: pytest.MonkeyPatch, root_a: Path, root_b: Path
) -> None:
    """B. No `reset_artifact_store()` between them: the key has to do the work."""
    configure_environment(monkeypatch, SCORING_PROVIDER="artifact")

    first_answers, first_similarity = _play(create_app(settings=_artifact_settings(root_a)))
    # Deliberately no reset here. This is the line the review was about.
    second_answers, second_similarity = _play(create_app(settings=_artifact_settings(root_b)))

    assert first_answers <= set(ROOT_A_ANSWERS)
    assert second_answers <= set(ROOT_B_ANSWERS)
    assert not second_answers & set(ROOT_A_ANSWERS), "the first root must not linger"
    assert first_similarity == pytest.approx(
        _stored_similarity(root_a, next(iter(first_answers)), ROOT_A_VOCABULARY)
    )
    assert second_similarity == pytest.approx(
        _stored_similarity(root_b, next(iter(second_answers)), ROOT_B_VOCABULARY)
    )
    assert first_similarity != pytest.approx(second_similarity), (
        "the fixture must give the shared guess different scores, or this proves nothing"
    )


def test_the_answer_selector_and_the_scorer_read_one_root(
    monkeypatch: pytest.MonkeyPatch, root_a: Path, root_b: Path
) -> None:
    """Half a fix would pass the tests above one at a time and fail this one.

    An app whose selector drew from one root and whose scorer read another could
    still produce answers from a valid set and scores from a valid set. What it
    could not do is win: the answer it set would have to be scorable, and score
    1.0, in the root the scorer actually reads.
    """
    configure_environment(
        monkeypatch, SCORING_PROVIDER="artifact", ARTIFACT_ROOT=str(root_b)
    )
    app = create_app(settings=_artifact_settings(root_a))
    repository = InMemoryGameRepository()
    app.dependency_overrides[game_repository] = lambda: repository

    with TestClient(app) as client:
        game_id = client.post("/api/games").json()["gameId"]
        game = repository.get(game_id)
        assert game is not None
        won = client.post(f"/api/games/{game_id}/guesses", json={"word": game.answer})
        state = client.get(f"/api/games/{game_id}")

    assert won.status_code == 200, won.text
    assert won.json()["similarity"] == 1.0
    assert won.json()["rank"] == 1
    assert state.json()["status"] == "won"


def test_the_cache_size_is_part_of_the_stores_identity(
    monkeypatch: pytest.MonkeyPatch, root_a: Path
) -> None:
    """C. Same root, different bound: reusing the old store would keep the old bound."""
    configure_environment(monkeypatch, SCORING_PROVIDER="artifact")
    small = get_artifact_store(_artifact_settings(root_a, artifact_cache_size=1))
    large = get_artifact_store(_artifact_settings(root_a, artifact_cache_size=2))

    assert large is not small
    for store, expected in ((small, 1), (large, 2)):
        for answer in store.answers:
            entry = store.manifest.entry_for(answer)
            assert entry is not None
            store.get(entry)
        assert store.cached_artifact_count == expected


def test_equal_configuration_shares_one_store(
    monkeypatch: pytest.MonkeyPatch, root_a: Path
) -> None:
    """D. Keyed by value, not by object identity — or nothing would ever be shared."""
    configure_environment(monkeypatch, SCORING_PROVIDER="artifact")

    first = get_artifact_store(_artifact_settings(root_a))
    second = get_artifact_store(_artifact_settings(root_a))

    assert first is second


def test_two_apps_on_one_root_share_its_store(
    monkeypatch: pytest.MonkeyPatch, root_a: Path
) -> None:
    """D, through the wiring: caching still works, it is just no longer blind."""
    configure_environment(monkeypatch, SCORING_PROVIDER="artifact")
    settings = _artifact_settings(root_a)

    with TestClient(create_app(settings=settings)):
        first = get_artifact_store(settings)
    with TestClient(create_app(settings=_artifact_settings(root_a))):
        second = get_artifact_store(settings)

    assert first is second
