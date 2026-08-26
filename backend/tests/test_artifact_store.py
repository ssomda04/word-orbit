"""The bounded cache in front of the artifact reader.

Everything here is about *when a file is read*, not what is in it — the reader's
own tests already cover the contents. So every test counts loads: the store is
correct exactly when a cold answer costs one read, a warm one costs none, and
the number of resident answers never passes the configured bound however many
answers a process sees.

Loads are counted by replacing `load_answer` in the store's own namespace. The
alternative would be a hook on `ArtifactStore` that exists only for tests, and a
cache that has to be asked whether it worked is not one this suite should trust.
"""

import threading
from pathlib import Path

import pytest

from app.services.scoring.artifact import (
    AnswerEntry,
    ArtifactError,
    ArtifactManifest,
    ArtifactStore,
    load_manifest,
)
from app.services.scoring.artifact import store as store_module
from tests import artifact_fixture as fixture

# Six answers over the fixture's six-word vocabulary: enough to fill a small
# cache several times over and still have a distinct answer left to evict.
ANSWERS: tuple[str, ...] = fixture.VOCABULARY


@pytest.fixture
def root(tmp_path: Path) -> Path:
    target = tmp_path / "artifacts-root"
    fixture.write_root(target, answers=ANSWERS)
    return target


@pytest.fixture
def manifest(root: Path) -> ArtifactManifest:
    return load_manifest(root)


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the artifact_id of every answer actually read from disk."""
    recorded: list[str] = []
    real = store_module.load_answer

    def _counting_load(manifest: ArtifactManifest, entry: AnswerEntry):
        recorded.append(entry.artifact_id)
        return real(manifest, entry)

    monkeypatch.setattr(store_module, "load_answer", _counting_load)
    return recorded


def _entry(manifest: ArtifactManifest, answer: str) -> AnswerEntry:
    entry = manifest.entry_for(answer)
    assert entry is not None, "the fixture root must serve every answer it wrote"
    return entry


# --- Loading and caching -----------------------------------------------------


def test_the_first_access_reads_from_disk(
    manifest: ArtifactManifest, loads: list[str]
) -> None:
    store = ArtifactStore(manifest)
    entry = _entry(manifest, fixture.ANSWER)

    artifact = store.get(entry)

    assert loads == [entry.artifact_id]
    assert artifact.artifact_id == entry.artifact_id


def test_a_second_access_to_the_same_answer_reads_nothing(
    manifest: ArtifactManifest, loads: list[str]
) -> None:
    store = ArtifactStore(manifest)
    entry = _entry(manifest, fixture.ANSWER)

    first = store.get(entry)
    second = store.get(entry)

    assert len(loads) == 1
    assert second is first, "a cache hit must return the loaded object, not a copy"


def test_nothing_is_loaded_before_an_answer_is_asked_for(
    manifest: ArtifactManifest, loads: list[str]
) -> None:
    """Construction is cheap; a root of thousands of answers must stay startable."""
    ArtifactStore(manifest)

    assert loads == []


def test_different_answers_are_cached_side_by_side(
    manifest: ArtifactManifest, loads: list[str]
) -> None:
    store = ArtifactStore(manifest, cache_size=len(ANSWERS))

    for answer in ANSWERS:
        store.get(_entry(manifest, answer))
    for answer in ANSWERS:
        store.get(_entry(manifest, answer))

    assert len(loads) == len(ANSWERS), "the second pass must be entirely cache hits"
    assert store.cached_artifact_count == len(ANSWERS)


# --- The bound ---------------------------------------------------------------


def test_the_cache_never_exceeds_its_size(manifest: ArtifactManifest) -> None:
    store = ArtifactStore(manifest, cache_size=2)

    for answer in ANSWERS:
        store.get(_entry(manifest, answer))
        assert store.cached_artifact_count <= 2

    assert store.cached_artifact_count == 2


def test_the_least_recently_used_answer_is_the_one_evicted(
    manifest: ArtifactManifest, loads: list[str]
) -> None:
    """Recency, not insertion order: a re-read answer must survive the next miss."""
    first, second, third = ANSWERS[0], ANSWERS[1], ANSWERS[2]
    store = ArtifactStore(manifest, cache_size=2)

    store.get(_entry(manifest, first))
    store.get(_entry(manifest, second))
    # Touching `first` makes `second` the least recently used one.
    store.get(_entry(manifest, first))
    store.get(_entry(manifest, third))
    loads.clear()

    store.get(_entry(manifest, first))
    assert loads == [], "the recently used answer must still be resident"

    store.get(_entry(manifest, second))
    assert loads == [_entry(manifest, second).artifact_id]


def test_an_evicted_answer_is_reloaded_on_the_next_access(
    manifest: ArtifactManifest, loads: list[str]
) -> None:
    evicted, keeper = ANSWERS[0], ANSWERS[1]
    store = ArtifactStore(manifest, cache_size=1)

    store.get(_entry(manifest, evicted))
    store.get(_entry(manifest, keeper))
    loads.clear()

    reloaded = store.get(_entry(manifest, evicted))

    assert loads == [_entry(manifest, evicted).artifact_id]
    assert reloaded.artifact_id == _entry(manifest, evicted).artifact_id


def test_a_cache_of_one_still_serves_repeat_access(
    manifest: ArtifactManifest, loads: list[str]
) -> None:
    """The smallest legal cache is a cache, not a passthrough."""
    store = ArtifactStore(manifest, cache_size=1)
    entry = _entry(manifest, fixture.ANSWER)

    store.get(entry)
    store.get(entry)
    store.get(entry)

    assert len(loads) == 1
    assert store.cached_artifact_count == 1


@pytest.mark.parametrize("size", [0, -1])
def test_a_cache_size_below_one_is_rejected(manifest: ArtifactManifest, size: int) -> None:
    with pytest.raises(ArtifactError, match="cache_size"):
        ArtifactStore(manifest, cache_size=size)


# --- What the store exposes to the rest of the app ---------------------------


def test_the_store_offers_the_manifests_answers_as_the_answer_source(
    manifest: ArtifactManifest,
) -> None:
    store = ArtifactStore(manifest)

    assert store.answers == manifest.answers
    assert set(store.answers) == set(ANSWERS)


# --- Concurrency -------------------------------------------------------------


def test_concurrent_access_stays_bounded_and_returns_intact_artifacts(
    manifest: ArtifactManifest,
) -> None:
    """Sync handlers run in a thread pool, so the cache is reached in parallel.

    Two threads missing on the same answer may both load it — that is the
    documented trade. What must not happen is a cache that outgrows its bound,
    or a caller that gets an artifact belonging to a different answer.
    """
    store = ArtifactStore(manifest, cache_size=2)
    barrier = threading.Barrier(len(ANSWERS) * 2)
    failures: list[str] = []

    def _worker(answer: str) -> None:
        entry = _entry(manifest, answer)
        barrier.wait()
        for _ in range(20):
            artifact = store.get(entry)
            if artifact.artifact_id != entry.artifact_id:
                failures.append(answer)
            if store.cached_artifact_count > 2:
                failures.append(f"{answer}: cache overflowed")

    threads = [
        threading.Thread(target=_worker, args=(answer,))
        for answer in ANSWERS * 2
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    assert store.cached_artifact_count == 2
