"""Health-check schema."""

from typing import Literal

from app.schemas.base import CamelModel


class HealthResponse(CamelModel):
    status: Literal["ok"] = "ok"
