"""Embedding service package.

The rest of the app depends only on the `EmbeddingService` Protocol, so the
concrete implementation (mock today, a real model later) can be swapped via the
`EMBEDDING_PROVIDER` env var without touching routers or domain logic.
"""

from app.services.embedding.base import EmbeddingService
from app.services.embedding.deterministic import DeterministicEmbeddingService
from app.services.embedding.factory import get_embedding_service

__all__ = [
    "EmbeddingService",
    "DeterministicEmbeddingService",
    "get_embedding_service",
]
