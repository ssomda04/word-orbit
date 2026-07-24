"""A deterministic, dependency-free stand-in for a real embedding model.

Why this exists
---------------
Frontend and backend need to build the game before a Korean embedding model is
selected and downloaded. `DeterministicEmbeddingService` produces stable vectors
purely from the input string (via hashing), so:

- the same word always maps to the same vector (reproducible tests),
- no model download, GPU, or heavy library is required,
- it satisfies the exact same `EmbeddingService` Protocol a real model will.

It is NOT semantically meaningful: two related Korean words are not "close".
Swap it for a real model later via `EMBEDDING_PROVIDER` — no caller changes.
"""

import hashlib
import math
from collections.abc import Sequence


def _hash_floats(text: str, dim: int) -> list[float]:
    """Deterministically derive ``dim`` floats in [-1, 1) from ``text``.

    We stream bytes from SHA-256 over ``"<i>:<text>"`` so the output is stable
    across processes and platforms (unlike Python's salted ``hash()``).
    """
    values: list[float] = []
    counter = 0
    while len(values) < dim:
        digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
        for i in range(0, len(digest), 4):
            if len(values) >= dim:
                break
            chunk = int.from_bytes(digest[i : i + 4], "big")
            # Map the 32-bit int into [-1, 1).
            values.append((chunk / 2**31) - 1.0)
        counter += 1
    return values


class DeterministicEmbeddingService:
    """Hash-based embedding service conforming to `EmbeddingService`."""

    def __init__(self, dim: int = 64) -> None:
        if dim < 3:
            raise ValueError("dim must be >= 3 (needed for 3D projection)")
        self.dim = dim
        # Fixed pseudo-random projection matrix (dim x 3) for project_3d.
        # Derived from a constant seed so projections are reproducible.
        self._projection = [_hash_floats(f"__axis_{axis}__", dim) for axis in range(3)]

    @staticmethod
    def _normalize(text: str) -> str:
        normalized = text.strip()
        if not normalized:
            raise ValueError("cannot embed empty or whitespace-only text")
        return normalized

    def encode(self, text: str) -> list[float]:
        raw = _hash_floats(self._normalize(text), self.dim)
        norm = math.sqrt(sum(x * x for x in raw))
        if norm == 0.0:
            return raw
        return [x / norm for x in raw]  # unit vector -> cosine == dot product

    def encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]

    def similarity(self, first: str, second: str) -> float:
        a = self.encode(first)
        b = self.encode(second)
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        # Clamp to guard against tiny floating-point overshoot past [-1, 1].
        return max(-1.0, min(1.0, dot))

    def project_3d(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.encode_many(texts)
        return [
            [sum(v * axis[i] for i, v in enumerate(vec)) for axis in self._projection]
            for vec in vectors
        ]
