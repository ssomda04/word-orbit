"""How an answer maps to files inside an artifact root.

Every path in a root is *derived*, never chosen. ``manifest.json`` records the
two paths for each answer, but this module recomputes them and the manifest
loader compares — a manifest cannot point the server at an arbitrary file, and
`..`, absolute paths, and alternate spellings all fail that comparison without
needing a rule of their own.

The layout mirrors ``ml/src/contextle_eval/rank_artifact.py`` and is
re-implemented rather than imported: the backend never imports ``ml`` (AGENTS.md),
and two independent implementations of one contract are what make a format test
worth running.

    <root>/artifacts/<artifact_id[:2]>/<artifact_id>/similarity.npy
    <root>/artifacts/<artifact_id[:2]>/<artifact_id>/rank.npy

The two-character shard keeps a directory listing usable at a few thousand
answers; it carries no meaning beyond that.
"""

import hashlib
import unicodedata

# `manifest.json` names this algorithm so a root built by a different rule is
# rejected rather than silently mis-addressed.
ARTIFACT_ID_ALGORITHM = "sha256-nfkc-utf8"

SIMILARITY_FILENAME = "similarity.npy"
RANK_FILENAME = "rank.npy"

_ARTIFACTS_DIRECTORY = "artifacts"
_SHARD_LENGTH = 2


def artifact_id_for(answer: str) -> str:
    """Return the deterministic identifier for ``answer``.

    ``sha256`` of the NFKC-normalized, trimmed answer, encoded as UTF-8. This is
    an identifier, not a protection: it exists so a plaintext answer never
    appears in a filesystem path (see ``errors``).

    Raises:
        ValueError: ``answer`` is blank after normalization.
    """
    normalized = unicodedata.normalize("NFKC", answer).strip()
    if not normalized:
        raise ValueError("cannot derive an artifact id from a blank answer")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def artifact_directory(artifact_id: str) -> str:
    """Return the root-relative directory for ``artifact_id``, POSIX-separated."""
    return f"{_ARTIFACTS_DIRECTORY}/{artifact_id[:_SHARD_LENGTH]}/{artifact_id}"


def similarity_path(artifact_id: str) -> str:
    """Return the root-relative similarity array path, POSIX-separated."""
    return f"{artifact_directory(artifact_id)}/{SIMILARITY_FILENAME}"


def rank_path(artifact_id: str) -> str:
    """Return the root-relative rank array path, POSIX-separated."""
    return f"{artifact_directory(artifact_id)}/{RANK_FILENAME}"
