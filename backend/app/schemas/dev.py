"""Schemas for the development-only similarity endpoint.

NOTE: These back `POST /api/dev/similarity`, a scaffolding endpoint used to
verify the frontend<->backend<->embedding wiring. It is NOT a real game API and
is expected to be removed once `POST /api/games/{gameId}/guesses` exists.
See `docs/API_SPEC.md`.
"""

from pydantic import Field, field_validator

from app.schemas.base import CamelModel


class SimilarityRequest(CamelModel):
    first: str = Field(..., min_length=1, description="First word/phrase.")
    second: str = Field(..., min_length=1, description="Second word/phrase.")

    @field_validator("first", "second")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must not be empty or whitespace-only")
        return stripped


class SimilarityResponse(CamelModel):
    first: str
    second: str
    similarity: float
    provider: str
