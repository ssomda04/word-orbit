"""Selects and caches the active embedding implementation.

The service is created once per process — a real model is loaded here exactly
once — and shared via FastAPI's dependency injection.

Why an explicit lock instead of ``@lru_cache``: ``lru_cache`` only *stores* one
result, it does not stop two threads that miss simultaneously from both running
the factory body. With a mock that is harmless; with a multi-gigabyte FastText
model it means loading the weights twice and running out of memory. The
double-checked lock below makes construction happen once, full stop.
"""

import threading
from pathlib import Path

from app.core.config import Settings, get_settings
from app.services.embedding.base import EmbeddingService
from app.services.embedding.deterministic import DeterministicEmbeddingService
from app.services.embedding.fasttext_service import (
    FastTextConfigurationError,
    FastTextEmbeddingService,
)

# Providers that resolve to the dependency-free deterministic mock.
_MOCK_PROVIDERS = {"mock", "deterministic"}
# Providers that resolve to the local FastText baseline model.
_FASTTEXT_PROVIDERS = {"fasttext", "fast-text", "fast_text"}
_SENTENCE_TRANSFORMER_PROVIDERS = {"sentence-transformers", "sentence_transformers", "st"}

_instance: EmbeddingService | None = None
_lock = threading.Lock()


def get_embedding_service() -> EmbeddingService:
    """Return the process-wide embedding service, chosen by settings.

    Built at most once per process; concurrent callers share the same instance.
    """
    global _instance
    # Fast path: an assignment to a module global is atomic, so an already-built
    # service needs no lock.
    if _instance is not None:
        return _instance
    with _lock:
        if _instance is None:
            _instance = _build_embedding_service()
        return _instance


def reset_embedding_service() -> None:
    """Drop the cached service so the next call rebuilds it.

    For tests only: production wiring builds the service once at startup
    (``app.main.create_app``) and never resets it.
    """
    global _instance
    with _lock:
        _instance = None


def _build_embedding_service() -> EmbeddingService:
    settings: Settings = get_settings()
    provider = settings.embedding_provider.strip().lower()

    if provider in _MOCK_PROVIDERS:
        return DeterministicEmbeddingService()

    if provider in _FASTTEXT_PROVIDERS:
        return _build_fasttext_service(settings)

    if provider in _SENTENCE_TRANSFORMER_PROVIDERS:
        # Intentionally not wired up — no transformer model has been selected
        # yet; the comparison against the FastText baseline comes first
        # (docs/MODEL_EVALUATION.md).
        raise NotImplementedError(
            "EMBEDDING_PROVIDER='sentence-transformers' is not implemented yet. "
            "Install the extra (`uv sync --extra embeddings`), pick a model in "
            "docs/MODEL_EVALUATION.md, then add a SentenceTransformerEmbeddingService."
        )

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={settings.embedding_provider!r}. "
        f"Expected one of: mock, deterministic, fasttext, sentence-transformers."
    )


def _build_fasttext_service(settings: Settings) -> FastTextEmbeddingService:
    """Resolve ``FASTTEXT_MODEL_PATH`` and load the model from it."""
    raw_path = settings.fasttext_model_path.strip()
    if not raw_path:
        raise FastTextConfigurationError(
            "EMBEDDING_PROVIDER=fasttext requires FASTTEXT_MODEL_PATH, which is "
            "unset. Set it to the absolute path of a FastText .bin you have "
            "already downloaded yourself (e.g. cc.ko.300.bin), and install the "
            "loader with `uv sync --extra fasttext`. The server never downloads "
            "a model, and model files are never committed to this repository."
        )
    # `expanduser` for `~/models/...`; relative paths resolve against the process
    # working directory, so the docs ask for an absolute path.
    return FastTextEmbeddingService.load(Path(raw_path).expanduser())
