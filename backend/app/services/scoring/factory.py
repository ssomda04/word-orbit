"""Selects the active scoring provider, and caches what it owns.

One place decides where a guess's similarity and rank come from. Everything
downstream — the dependency wiring, the startup warm-up, the answer selector —
asks this module rather than reading ``SCORING_PROVIDER`` itself, so there is
exactly one spelling of the choice and exactly one place to widen it.

Settings are passed in, never fetched
--------------------------------------
``get_artifact_store`` takes a ``Settings`` and nothing here calls
``get_settings()``. That is deliberate and load-bearing: ``create_app`` accepts
an explicit settings object, and a function that re-reads the global one can
answer from a *different* configuration than the app that called it — selecting
artifact mode from one root while serving from another. Making the parameter
required rather than optional is the point; an optional one would let a call
site quietly fall back to the global and reopen the same gap.

Only the artifact provider has anything to cache. ``EmbeddingGuessScorer`` owns
no state and is composed per request from the embedding and ranking factories,
which already do their own caching; ``ArtifactStore`` owns a validated manifest
and a bounded array cache, so it is built once and shared under a lock. A plain
``lru_cache`` would not do: it stores one result but does not stop two threads
that miss simultaneously from both parsing and validating the root.

The cache is keyed by the configuration it was built from — the resolved root
and the cache size — not merely by "something was built". A process-global that
ignored the configuration would hand an app whatever the *previous* app was
configured with, which is a correctness bug rather than a stale-cache
inconvenience: the answers a game can have, and the numbers a guess scores,
would come from a root nobody asked for. Two spellings of one directory produce
two keys and therefore two stores; that costs a second load and is otherwise
harmless, so the key stays the path as configured rather than a resolved one.

One check the reader cannot make
---------------------------------
``load_manifest`` validates a root against the *format*: an answer key must be a
canonical vocabulary word. That is the vocabulary policy — NFKC, trimmed, in
``vocabulary.txt`` — and it deliberately permits words the *game* refuses, because
a vocabulary is a word list and a guess is an input. ``app.domain.vocabulary``
says so outright: a dictionary headword like ``"고유 명사"`` contains a space and
must stay in the vocabulary, or every rank below it would shift.

``app.domain.game.normalize_word`` applies the guess rules on top: no internal
whitespace, and a length cap. A root is therefore free to offer an answer this
server cannot set, and until it is checked the failure lands at the worst
possible moment — ``POST /api/games`` answering ``400 INVALID_WORD`` to a request
that contains no word at all, intermittently, only on the draws that happen to
pick it.

So the admissibility of the answer pool is checked here, once, at startup. Here
rather than in the artifact package because it is not a fact about the format:
it is this application asking whether it can run a game on this root, and the
reader stays free of game rules. Nothing is filtered — an unusable answer is a
root that should not be served at all, not one to quietly serve less of.
"""

import threading
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from app.core.config import Settings, get_settings
from app.core.errors import InvalidWordError
from app.domain.game import normalize_word
from app.services.scoring.artifact.errors import ArtifactError
from app.services.scoring.artifact.manifest import ArtifactManifest, load_manifest
from app.services.scoring.artifact.paths import artifact_id_for
from app.services.scoring.artifact.store import ArtifactStore


class ScoringProvider(StrEnum):
    """Where a guess's similarity and rank are read from."""

    EMBEDDING = "embedding"
    ARTIFACT = "artifact"


@dataclass(frozen=True, slots=True)
class _StoreConfig:
    """Everything about a ``Settings`` that changes which store it describes.

    Two settings objects agreeing on these two values describe the same store,
    whatever else differs; disagreeing on either means a different one must be
    built. This is the cache's identity, so it is frozen and compared by value.
    """

    root: Path
    cache_size: int


# The one store this process has built, next to the configuration it was built
# from. Single-entry rather than a dict: a server runs one configuration, and a
# process that legitimately sees several (the test suite) wants the previous one
# dropped rather than accumulated.
_cached: tuple[_StoreConfig, ArtifactStore] | None = None
_lock = threading.Lock()


def resolve_scoring_provider(settings: Settings | None = None) -> ScoringProvider:
    """Return the configured provider.

    ``settings`` is optional here, unlike ``get_artifact_store``: the answer is
    derived from one field and nothing is cached under it, so a caller with no
    settings in hand cannot cause a mismatch by omitting it.

    Raises:
        ValueError: ``SCORING_PROVIDER`` names a provider that does not exist.
            Deliberately not a silent fallback to the default: a typo would
            otherwise start a server that scores through the wrong data.
    """
    settings = settings or get_settings()
    raw = settings.scoring_provider.strip().lower()
    try:
        return ScoringProvider(raw)
    except ValueError:
        supported = ", ".join(provider.value for provider in ScoringProvider)
        raise ValueError(
            f"Unknown SCORING_PROVIDER={settings.scoring_provider!r}. "
            f"Expected one of: {supported}."
        ) from None


def get_artifact_store(settings: Settings) -> ArtifactStore:
    """Return the store for ``settings``, building it on first use.

    Callers passing equal configuration share one instance; a caller passing
    different configuration gets a store built from *its* configuration, and the
    previous one is dropped. No caller can be handed a store belonging to
    somebody else's root.

    Raises:
        ArtifactError: ``ARTIFACT_ROOT`` is unset or is not a directory, the root
            it names is missing, malformed, or outside the supported contract, or
            it offers an answer this server could not set a game on.
    """
    global _cached
    # Resolved before the lock and on every call: it is two field reads and two
    # filesystem-free checks, and it is what decides whether the cached store is
    # even the right one to return.
    config = _store_config(settings)

    # Fast path: an assignment to a module global is atomic, so reading the pair
    # once and comparing it needs no lock.
    cached = _cached
    if cached is not None and cached[0] == config:
        return cached[1]

    with _lock:
        if _cached is not None and _cached[0] == config:
            return _cached[1]
        store = _build_artifact_store(config)
        _cached = (config, store)
        return store


def reset_artifact_store() -> None:
    """Drop the cached store so the next call rebuilds it.

    For tests only, and only for the case the key cannot cover: forcing a rebuild
    from *identical* configuration. A test that changes the root or the cache
    size does not need this — that is a different key, and a different store.
    """
    global _cached
    with _lock:
        _cached = None


def _store_config(settings: Settings) -> _StoreConfig:
    """Resolve ``settings`` into the identity of the store it asks for.

    Raises:
        ArtifactError: ``ARTIFACT_ROOT`` is unset or does not name a directory.
    """
    raw_root = settings.artifact_root.strip()
    if not raw_root:
        raise ArtifactError(
            "SCORING_PROVIDER=artifact requires ARTIFACT_ROOT, which is unset. "
            "Set it to the absolute path of an artifact root produced by the ML "
            "area (see docs/ARTIFACT_FORMAT.md). Artifact roots are data and are "
            "never committed to this repository."
        )

    # `expanduser` for `~/artifacts`; relative paths resolve against the process
    # working directory, so the docs ask for an absolute path.
    root = Path(raw_root).expanduser()
    if not root.is_dir():
        raise ArtifactError(f"ARTIFACT_ROOT is not a directory: {root}")

    return _StoreConfig(root=root, cache_size=settings.artifact_cache_size)


def _build_artifact_store(config: _StoreConfig) -> ArtifactStore:
    # Full validation, every startup: the manifest, the canonical vocabulary,
    # every answer mapping, and the existence of every file they refer to. The
    # arrays themselves are read only when an answer is first scored.
    manifest = load_manifest(config.root)
    _validate_answers_are_playable(manifest)
    return ArtifactStore(manifest, cache_size=config.cache_size)


def _validate_answers_are_playable(manifest: ArtifactManifest) -> None:
    """Require every answer of ``manifest`` to be one this server can set.

    ``manifest.answers`` is the answer source of truth in artifact mode, so every
    entry has to survive the same rule a guess does — otherwise creating a game
    on it raises, or the player could never type it back.

    Only the rejection half of ``normalize_word`` is exercised. It also returns
    the normalized form, and that value is not compared: both normalizers apply
    NFKC-then-strip in that order (``app.domain.vocabulary``), and the reader has
    already required every key to be canonical under the vocabulary one, so a key
    that is accepted here is necessarily returned unchanged. Re-asserting it
    would be checking the reader's work twice; ``tests/test_scoring_factory.py``
    pins the property instead.

    Raises:
        ArtifactError: an answer is unusable as a game answer.
    """
    for answer in manifest.answers:
        try:
            normalize_word(answer)
        except InvalidWordError as exc:
            # `exc.message` is one of three fixed sentences in `normalize_word`,
            # none of which quotes its input — so the rule can be reported while
            # the answer stays out of the message, and out of the log that
            # message ends up in. This loop sees the whole answer pool, so the
            # cause is dropped rather than chained: nothing about a failure here
            # needs a frame from the domain layer, and not carrying one is one
            # fewer thing that has to keep being answer-free.
            raise ArtifactError(
                f"Artifact {artifact_id_for(answer)} is an answer this server "
                f"cannot set a game on: {exc.message} The key satisfies the "
                "vocabulary rules but not the guess rules "
                "(app.domain.game.normalize_word), so a game drawing it would "
                "fail to start. Rebuild the root without it."
            ) from None
