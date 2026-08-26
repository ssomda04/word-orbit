"""Holding one validated artifact root, and the answers loaded out of it.

What is resident, and what is not
---------------------------------
The manifest — schema, vocabulary, and the answer→(artifact_id, index) map — is
read once, at startup, and stays. It is small and every request needs it.

The per-answer arrays are not. A root of a few thousand answers is hundreds of
megabytes, of which one game touches exactly one answer's worth, so they are
loaded when an answer is first scored and then kept in a bounded LRU cache.
Startup therefore stays fast and flat in the number of answers, while a warm
answer costs no disk read at all.

Keyed by ``artifact_id``, never by the answer
---------------------------------------------
The id is the sha256 of the answer and the on-disk directory name (see
``paths``), so it identifies a cache entry exactly as well as the word does. The
word is the thing that must not appear in a log, a metric, or a repr; the id
already appears in every path. Using it as the key means nothing this class
holds, counts, or evicts can name an answer.

Concurrency
-----------
Sync route handlers run in FastAPI's thread pool, so two guesses can reach the
cache at once. One lock guards the ``OrderedDict``; the load itself happens
*outside* it, so a slow read cannot block every other answer's cache hit. Two
threads missing on the same id therefore both load it — the same trade
``VectorRankProvider`` makes, and harmless here for the same reason: loading is
pure and validated, so the two results are interchangeable and the second insert
simply replaces the first.
"""

import threading
from collections import OrderedDict

from app.services.scoring.artifact.answer import AnswerArtifact, load_answer
from app.services.scoring.artifact.errors import ArtifactError
from app.services.scoring.artifact.manifest import AnswerEntry, ArtifactManifest

# Answers whose arrays stay resident. Mirrors ``Settings.artifact_cache_size``,
# which is what production passes; this default only covers a direct
# construction (tests, a script).
DEFAULT_CACHE_SIZE = 64


class ArtifactStore:
    """A validated artifact root, with its answers loaded on demand."""

    def __init__(
        self,
        manifest: ArtifactManifest,
        *,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        """Take ownership of an already-validated manifest.

        Args:
            manifest: The result of ``load_manifest``. Validation is not
                repeated here — the manifest type *is* the statement that the
                root was checked.
            cache_size: How many answers' arrays to keep resident.

        Raises:
            ArtifactError: ``cache_size`` is below one.
        """
        if cache_size < 1:
            raise ArtifactError("Artifact cache_size must be at least 1.")

        self._manifest = manifest
        self._cache_size = cache_size
        self._cache: OrderedDict[str, AnswerArtifact] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def manifest(self) -> ArtifactManifest:
        """The validated root this store serves from."""
        return self._manifest

    @property
    def answers(self) -> tuple[str, ...]:
        """Every answer this root can serve — the answer selector's word list.

        Plain answer words, like ``ArtifactManifest.answers``. Never log this.
        """
        return self._manifest.answers

    @property
    def cached_artifact_count(self) -> int:
        """Answers currently resident. Never exceeds the configured size."""
        with self._lock:
            return len(self._cache)

    def get(self, entry: AnswerEntry) -> AnswerArtifact:
        """Return the arrays for ``entry``, loading them on the first request.

        Raises:
            ArtifactError: the arrays are missing, unreadable, or contradict the
                manifest. Identified by ``artifact_id``, never by the answer.
        """
        with self._lock:
            cached = self._cache.get(entry.artifact_id)
            if cached is not None:
                self._cache.move_to_end(entry.artifact_id)
                return cached

        artifact = load_answer(self._manifest, entry)

        with self._lock:
            self._cache[entry.artifact_id] = artifact
            self._cache.move_to_end(entry.artifact_id)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return artifact
