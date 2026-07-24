"""The embedding interface every implementation must satisfy.

This is the single seam between the game logic and whatever produces vectors.
Real models (sentence-transformers, ...) and the deterministic mock are
interchangeable as long as they conform to this Protocol.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingService(Protocol):
    """Turns text into vectors and compares them.

    Implementations MUST be deterministic for a given input during a process's
    lifetime so results are reproducible and testable.
    """

    def encode(self, text: str) -> list[float]:
        """Return the embedding vector for a single piece of text."""
        ...

    def encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Return embedding vectors for many texts (order preserved)."""
        ...

    def similarity(self, first: str, second: str) -> float:
        """Cosine similarity of two texts, in the range [-1.0, 1.0]."""
        ...

    def project_3d(self, texts: Sequence[str]) -> list[list[float]]:
        """Project texts to 3D coordinates for visualization.

        NOTE: A 3D projection is a lossy view, not the true semantic distance.
        Real implementations should use PCA/UMAP (see docs/MODEL_EVALUATION.md).
        """
        ...
