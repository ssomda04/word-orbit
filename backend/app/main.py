"""Contextle backend entry point.

`create_app()` wires configuration, CORS, error handling and routers together.
Business logic lives in `app/services` and `app/domain`; this file stays thin.

Run locally:
    uv run uvicorn app.main:app --reload
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import api_router
from app.core.config import Settings, get_settings
from app.schemas.errors import ErrorResponse


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Contextle API",
        version=__version__,
        description="Semantic word-guessing game backend (skeleton).",
    )

    # CORS: origins come from configuration, never hard-coded to "*".
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Return validation failures in the standard error envelope (docs/API_SPEC.md).
    @app.exception_handler(RequestValidationError)
    async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                code="INVALID_INPUT",
                message="Request validation failed.",
                details=exc.errors(),
            ).model_dump(mode="json"),
        )

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {"name": "Contextle API", "version": __version__, "docs": "/docs"}

    app.include_router(api_router)
    return app


app = create_app()
