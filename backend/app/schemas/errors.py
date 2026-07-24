"""Standard error envelope.

All handled API errors should return this shape (see `docs/API_SPEC.md`):

    {"code": "INVALID_WORD", "message": "...", "details": null}
"""

from typing import Any

from app.schemas.base import CamelModel


class ErrorResponse(CamelModel):
    code: str
    message: str
    details: Any | None = None
