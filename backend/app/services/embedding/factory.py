"""Selects and caches the active embedding implementation.

The service is created once per process (a real model would be loaded here at
startup, exactly once) and shared via FastAPI's dependency injection.
"""

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.embedding.base import EmbeddingService
from app.services.embedding.deterministic import DeterministicEmbeddingService

# Providers that resolve to the dependency-free deterministic mock.
_MOCK_PROVIDERS = {"mock", "deterministic"}


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Return the process-wide embedding service, chosen by settings.

    Cached so the (potentially expensive) construction happens only once.
    """
    settings: Settings = get_settings()
    provider = settings.embedding_provider.strip().lower()

    if provider in _MOCK_PROVIDERS:
        return DeterministicEmbeddingService()

    if provider in {"sentence-transformers", "sentence_transformers", "st"}:
        # Intentionally not wired up in the skeleton — the model libraries live
        # in the optional `embeddings` extra and no model has been selected yet.
        raise NotImplementedError(
            "EMBEDDING_PROVIDER='sentence-transformers' is not implemented yet. "
            "Install the extra (`uv sync --extra embeddings`), pick a model in "
            "docs/MODEL_EVALUATION.md, then add a SentenceTransformerEmbeddingService."
        )

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={settings.embedding_provider!r}. "
        f"Expected one of: mock, deterministic, sentence-transformers."
    )
