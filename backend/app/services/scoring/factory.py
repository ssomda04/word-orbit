"""Selects the active scoring provider, and caches what it owns.

One place decides where a guess's similarity and rank come from. Everything
downstream — the dependency wiring, the startup warm-up, the answer selector —
asks this module rather than reading ``SCORING_PROVIDER`` itself, so there is
exactly one spelling of the choice and exactly one place to widen it.

Only the artifact provider has anything to cache. ``EmbeddingGuessScorer`` owns
no state and is composed per request from the embedding and ranking factories,
which already do their own caching; ``ArtifactStore`` owns a validated manifest
and a bounded array cache, so it gets the same double-checked lock those two
use. ``lru_cache`` would not do: it stores one result but does not stop two
threads that miss simultaneously from both parsing and validating the root.

Configuration is validated here rather than on ``Settings`` for the reason
``FASTTEXT_MODEL_PATH`` is: ``ARTIFACT_ROOT`` is meaningless in embedding mode,
and a mock run must never fail over a variable it does not use. What *is* on
``Settings`` is ``ARTIFACT_CACHE_SIZE``, whose validity does not depend on the
mode at all.

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


_store: ArtifactStore | None = None
_lock = threading.Lock()


def resolve_scoring_provider(settings: Settings | None = None) -> ScoringProvider:
    """Return the configured provider.

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


def get_artifact_store() -> ArtifactStore:
    """Return the process-wide artifact store, building it on first use.

    Built at most once per process; concurrent callers share the same instance.

    Raises:
        ArtifactError: ``ARTIFACT_ROOT`` is unset or is not a directory, the root
            it names is missing, malformed, or outside the supported contract, or
            it offers an answer this server could not set a game on.
    """
    global _store
    # Fast path: an assignment to a module global is atomic, so an already-built
    # store needs no lock.
    if _store is not None:
        return _store
    with _lock:
        if _store is None:
            _store = _build_artifact_store()
        return _store


def reset_artifact_store() -> None:
    """Drop the cached store so the next call rebuilds it.

    For tests only: production wiring builds it once at startup
    (``app.main``'s lifespan) and never resets it.
    """
    global _store
    with _lock:
        _store = None


def _build_artifact_store() -> ArtifactStore:
    settings = get_settings()
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

    # Full validation, every startup: the manifest, the canonical vocabulary,
    # every answer mapping, and the existence of every file they refer to. The
    # arrays themselves are read only when an answer is first scored.
    manifest = load_manifest(root)
    _validate_answers_are_playable(manifest)
    return ArtifactStore(manifest, cache_size=settings.artifact_cache_size)


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
