"""Development-only endpoints for verifying the wiring end-to-end.

WARNING: This router is scaffolding, not a game API. `POST /api/dev/similarity`
lets the frontend confirm the embedding service is reachable before the real
`POST /api/games/{gameId}/guesses` endpoint exists. Remove it once the game API
lands (tracked in docs/ROADMAP.md).
"""

from fastapi import APIRouter

from app.api.deps import EmbeddingServiceDep
from app.core.config import get_settings
from app.schemas.dev import SimilarityRequest, SimilarityResponse

router = APIRouter(prefix="/api/dev", tags=["dev"])


@router.post("/similarity", response_model=SimilarityResponse)
def compute_similarity(
    payload: SimilarityRequest,
    embedder: EmbeddingServiceDep,
) -> SimilarityResponse:
    """Return the cosine similarity between two words (dev harness only)."""
    score = embedder.similarity(payload.first, payload.second)
    return SimilarityResponse(
        first=payload.first,
        second=payload.second,
        similarity=score,
        provider=get_settings().embedding_provider,
    )
