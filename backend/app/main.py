"""Contextle backend entry point.

`create_app()` wires configuration, CORS, error handling and routers together.
Business logic lives in `app/services` and `app/domain`; this file stays thin.

Run locally:
    uv run uvicorn app.main:app --reload
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import api_router
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.schemas.errors import ErrorResponse
from app.services.embedding import get_embedding_service
from app.services.ranking import get_rank_provider


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
    """Build the embedding service and rank provider before serving anything.

    For the deterministic mock this is a sub-millisecond no-op. For a real model
    it is the difference between an 8-second first guess and a warm one, and it
    turns a bad `FASTTEXT_MODEL_PATH` into a loud startup failure instead of a
    500 that would not even match the documented error envelope.

    The rank provider is warmed for the same reasons: with `VOCABULARY_PATH` set
    it embeds the whole vocabulary, which must happen once at startup rather than
    inside whichever request arrives first, and a bad path should fail loudly
    here. Order matters — the provider embeds through the embedding service.
    """
    get_embedding_service()
    get_rank_provider()
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Contextle API",
        version=__version__,
        description="Semantic word-guessing game backend (skeleton).",
        lifespan=_lifespan,
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
                # A failing custom validator leaves the raised exception object in
                # `ctx`, which is not JSON-serializable; render it as its message
                # instead. Without this the handler itself raises a 500.
                details=jsonable_encoder(exc.errors(), custom_encoder={Exception: str}),
            ).model_dump(mode="json"),
        )

    # Domain/service failures share the same envelope. FastAPI's HTTPException
    # would emit {"detail": ...} instead, which is not the documented contract.
    @app.exception_handler(AppError)
    async def _app_error_handler(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                code=exc.code,
                message=exc.message,
                details=exc.details,
            ).model_dump(mode="json"),
        )

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {"name": "Contextle API", "version": __version__, "docs": "/docs"}

    app.include_router(api_router)
    return app


app = create_app()
