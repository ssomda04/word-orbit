"""Contextle backend entry point.

`create_app()` wires configuration, CORS, error handling and routers together.
Business logic lives in `app/services` and `app/domain`; this file stays thin.

Run locally:
    uv run uvicorn app.main:app --reload
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api import deps
from app.api.routes import api_router
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.schemas.errors import ErrorResponse
from app.services.embedding import get_embedding_service
from app.services.ranking import get_rank_provider
from app.services.scoring import (
    ScoringProvider,
    get_artifact_store,
    resolve_scoring_provider,
)

logger = logging.getLogger(__name__)

# Fixed and deliberately uninformative. An unhandled exception carries whatever
# its raiser put in the message — a filesystem path, a model error, the answer
# word (`app.services.ranking.vector` interpolates it) — none of which may reach
# a client. The traceback goes to the log; the client gets this sentence.
INTERNAL_ERROR_MESSAGE = "서버에 예기치 못한 오류가 발생했습니다."


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Build whatever the configured scoring provider reads through.

    Warming what a provider needs is the same decision in both modes: it must
    happen once, before the first request, and a misconfiguration must stop the
    process here rather than surface as a 500 that would not even match the
    documented error envelope.

    What differs is the *set*, and the difference is load-bearing rather than
    tidy. Only one branch runs, so in artifact mode no embedding service is
    constructed and no vocabulary is embedded — with `EMBEDDING_PROVIDER=fasttext`
    that is a multi-gigabyte model this process never has a use for.

    Embedding mode: for the deterministic mock the two calls are a
    sub-millisecond no-op; for a real model they are the difference between an
    8-second first guess and a warm one. Order matters — the rank provider
    embeds the whole vocabulary through the embedding service.

    Artifact mode: the manifest, the canonical vocabulary, every answer mapping,
    and the existence of every file they refer to are all validated here, so a
    wrong `ARTIFACT_ROOT` fails startup. The per-answer arrays are deliberately
    *not* read — a root is hundreds of megabytes, of which one game touches one
    answer's worth, so they load on first use (`ArtifactStore`).

    The guess scorer itself is absent from both branches on purpose: neither
    implementation holds anything that is not warmed above.
    """
    if app.state.scoring_provider is ScoringProvider.ARTIFACT:
        get_artifact_store()
    else:
        get_embedding_service()
        get_rank_provider()
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    # Resolved once, here, and read back by the lifespan through `app.state`, so
    # a `create_app(settings=...)` with an explicit object cannot start a server
    # whose startup warms a different provider than its requests score through.
    # An unknown SCORING_PROVIDER raises before the app object exists.
    provider = resolve_scoring_provider(settings)

    app = FastAPI(
        title="Contextle API",
        version=__version__,
        description="Semantic word-guessing game backend (skeleton).",
        lifespan=_lifespan,
    )
    app.state.scoring_provider = provider

    if provider is ScoringProvider.ARTIFACT:
        # Production wiring, not a test seam. `guess_scorer` declares the
        # embedding and ranking dependencies so that overriding either still
        # reaches a guess; FastAPI resolves declared dependencies whether or not
        # the body uses them, so leaving it in place here would build the
        # embedding stack to serve a scorer that never calls it. Swapping the
        # dependency function is what keeps that stack untouched — see
        # `app.api.deps`.
        app.dependency_overrides[deps.guess_scorer] = deps.artifact_guess_scorer

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

    # Last line of defence. Without it Starlette answers an unhandled exception
    # with `{"detail": "Internal Server Error"}`, which is not the documented
    # envelope, so a client parsing errors by `code` would break on exactly the
    # responses it most needs to understand. Registering a handler for
    # `Exception` installs it on Starlette's ServerErrorMiddleware, which sends
    # this response and then re-raises so the server still logs the failure.
    @app.exception_handler(Exception)
    async def _unhandled_error_handler(request: Request, _: Exception) -> JSONResponse:
        # Method and path only: the path holds a gameId, never a guess or the
        # answer. The exception itself is rendered by `exception()` into the log
        # (traceback included) and never into the response.
        logger.exception(
            "Unhandled exception while handling %s %s; returning INTERNAL_ERROR.",
            request.method,
            request.url.path,
        )
        return JSONResponse(
            # `AppError` already declares this code/status as its base defaults,
            # so the catch-all and an explicitly raised `AppError` cannot drift.
            status_code=AppError.status_code,
            content=ErrorResponse(
                code=AppError.code,
                message=INTERNAL_ERROR_MESSAGE,
                # Always null: `details` is context for the client, and there is
                # no context here that is safe to share.
                details=None,
            ).model_dump(mode="json"),
        )

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {"name": "Contextle API", "version": __version__, "docs": "/docs"}

    app.include_router(api_router)
    return app


app = create_app()
