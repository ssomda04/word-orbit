"""The scorer that reads both values out of a precomputed artifact root.

The counterpart of ``EmbeddingGuessScorer``, and deliberately as thin: it turns
one guess into one row lookup. Everything expensive — validating the root,
loading an answer's arrays, caching them — belongs to ``ArtifactStore``, and
everything about the *format* belongs to the reader. What is left here is the
one thing neither of those can decide: what a word outside the vocabulary means.

Out of vocabulary is ``None``, not an exception
-----------------------------------------------
A stored artifact holds a similarity for exactly the words in
``vocabulary.txt``. A guess outside it cannot be scored at all — unlike a live
model, which composes a vector for any string. That is a real gap between the
two providers, and it has to surface somewhere.

It surfaces as ``None``. Raising ``InvalidWordError`` here would be shorter by a
line and wrong by a layer: ``INVALID_WORD`` is a *game* rule with an HTTP status
attached, while this class only knows whether a row exists. ``GameService`` owns
the rule and turns ``None`` into the error, exactly as it already turns a
finished game into a conflict.

The answer, by contrast, is not allowed to be missing. An answer with no entry
in the manifest means the selector and the root disagree — a misconfigured
server, not an unlucky guess — so it raises rather than returning ``None``,
which would tell the player their perfectly good word was invalid. The failure
is identified by ``artifact_id``; see ``errors`` for why never by the answer.
"""

from app.services.scoring.artifact.errors import ArtifactError
from app.services.scoring.artifact.paths import artifact_id_for
from app.services.scoring.artifact.store import ArtifactStore
from app.services.scoring.base import GuessScore


class ArtifactGuessScorer:
    """Scores a guess by reading one row of one precomputed answer artifact."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store

    def score(self, answer: str, word: str) -> GuessScore | None:
        """Return the stored similarity and rank of ``word``, or ``None``.

        ``None`` means ``word`` is outside the artifact's canonical vocabulary,
        which is the whole of this class's own policy.

        Raises:
            ArtifactError: this root has no artifact for ``answer``, or that
                artifact cannot be read or trusted.
        """
        entry = self._store.manifest.entry_for(answer)
        if entry is None:
            # Not `answer!r`: the message is rendered into the traceback that
            # `app.main` logs. The id identifies the missing artifact just as
            # precisely, and is already the directory name on disk.
            raise ArtifactError(
                f"The artifact root has no entry for artifact {artifact_id_for(answer)}. "
                "The answer source and the artifact root disagree."
            )

        artifact = self._store.get(entry)
        found = artifact.lookup(word, self._store.manifest.vocabulary)
        if found is None:
            return None

        similarity, rank = found
        return GuessScore(similarity=similarity, rank=rank)
