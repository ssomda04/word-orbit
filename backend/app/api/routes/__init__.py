"""Route modules, aggregated into a single API router."""

from fastapi import APIRouter

from app.api.routes import dev, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(dev.router)
