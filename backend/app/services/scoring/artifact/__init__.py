"""Reading a rank artifact root produced offline by the ML area.

An artifact root holds, for each answer the game may choose, that answer's
similarity and rank against every word in one canonical vocabulary — computed
once, offline, by a model this server never loads. The layout is documented in
``docs/ARTIFACT_FORMAT.md``.

This package only *reads and validates* a root. Nothing here is wired into the
game yet: there is no provider selection, no cache, and no scorer. Those arrive
with the runtime integration, and keeping them out means the contract can be
reviewed on its own.

The backend never imports ``ml`` (AGENTS.md), so this is an independent
implementation of the same format. That is the point rather than a cost: two
implementations that must agree are what makes a format regression visible.
"""

from app.services.scoring.artifact.answer import AnswerArtifact, load_answer
from app.services.scoring.artifact.errors import ArtifactError
from app.services.scoring.artifact.manifest import (
    AnswerEntry,
    ArtifactManifest,
    load_manifest,
)
from app.services.scoring.artifact.paths import (
    ARTIFACT_ID_ALGORITHM,
    artifact_directory,
    artifact_id_for,
    rank_path,
    similarity_path,
)
from app.services.scoring.artifact.vocabulary import (
    CanonicalVocabulary,
    read_canonical_vocabulary,
)

__all__ = [
    "ARTIFACT_ID_ALGORITHM",
    "AnswerArtifact",
    "AnswerEntry",
    "ArtifactError",
    "ArtifactManifest",
    "CanonicalVocabulary",
    "artifact_directory",
    "artifact_id_for",
    "load_answer",
    "load_manifest",
    "rank_path",
    "read_canonical_vocabulary",
    "similarity_path",
]
