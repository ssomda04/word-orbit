"""FastAPI dependency providers.

Routers depend on these instead of constructing services directly, which keeps
handlers testable (dependencies can be overridden in tests).
"""

from typing import Annotated

from fastapi import Depends

from app.services.embedding import EmbeddingService, get_embedding_service


def embedding_service() -> EmbeddingService:
    return get_embedding_service()


# Reusable annotated dependency for route signatures.
EmbeddingServiceDep = Annotated[EmbeddingService, Depends(embedding_service)]
