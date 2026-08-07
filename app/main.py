"""FastAPI application factory.

Routers are mounted here; business logic lives in services (CLAUDE.md 3, 9).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from app.api.routes import audit, auth, catalog, documents, health, query, review
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestIDMiddleware
from app.db.session import dispose_engines
from app.web import routes as web_routes
from app.web.deps import RedirectToLogin, login_redirect
from app.web.templates import STATIC_DIR

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
    app.include_router(auth.router)
    app.include_router(documents.router)
    app.include_router(catalog.router)
    app.include_router(query.router)
    app.include_router(review.router)
    app.include_router(audit.router)

    # -- UI ---------------------------------------------------------------
    app.include_router(web_routes.router)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.exception_handler(RedirectToLogin)
    async def _to_login(request: Request, exc: RedirectToLogin) -> Response:
        """Send a browser to the login form instead of a 401 body.

        The JSON API still answers 401 -- a client that can read it should get
        it. This applies only to the HTML pages, whose dependency raises.
        """
        return login_redirect(exc.next_path)

    @app.get("/", include_in_schema=False)
    async def _root() -> Response:
        return RedirectResponse("/ui/ask")

    return app


app = create_app()
