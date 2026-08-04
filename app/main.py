"""FastAPI application factory.

Routers are mounted here; business logic lives in services (CLAUDE.md 3, 9).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import documents, health
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestIDMiddleware
from app.db.session import dispose_engines

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(
        "application starting",
        extra={"environment": settings.ENVIRONMENT, "storage_dir": str(settings.STORAGE_DIR)},
    )
    yield
    await dispose_engines()
    logger.info("application stopped")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="OpusCovIntel",
        description="Sukuk & bond covenant intelligence platform",
        version="0.1.0",
        lifespan=lifespan,
        # Interactive docs are a dev convenience, not a production surface.
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
    )
    app.state.settings = settings
    app.add_middleware(RequestIDMiddleware)
    app.include_router(health.router)
    app.include_router(documents.router)
    return app


app = create_app()
