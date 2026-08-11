"""FastAPI application factory.

Routers are mounted here; business logic lives in services (CLAUDE.md 3, 9).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Scope

from app.api.routes import audit, auth, catalog, documents, health, query, review
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestIDMiddleware
from app.db.session import dispose_engines
from app.web import routes as web_routes
from app.web.deps import RedirectToLogin, login_redirect
from app.web.templates import STATIC_DIR

logger = get_logger(__name__)

# The Angular build output (`make frontend`). Not committed and not required:
# the API and the server-rendered UI work without it.
CLIENT_APP_DIR = Path(__file__).resolve().parents[1] / "frontend" / "dist"


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
    _mount_client_app(app)

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


def _mount_client_app(app: FastAPI) -> None:
    """Serve the Angular build at `/app`, if it has been built.

    **Same origin, deliberately.** The client app shares this process's origin,
    so the session cookie keeps `HttpOnly` and `SameSite=lax` and the CSRF
    property that comes with them. Serving it from its own host would mean CORS
    plus `SameSite=none`, trading a real defence for a deployment convenience
    (docs/deploy.md 6).

    Absent when nobody has run `make frontend`, which is the normal state of a
    Python-only checkout and of the test suite. The API and the server-rendered
    UI do not depend on it, so a missing build is a missing screen rather than
    a broken application -- and saying so at startup beats a 404 nobody can
    explain.
    """
    # Recorded so the server-rendered nav can offer the screens the client app
    # owns without ever linking somewhere that is not there. A Python-only
    # checkout is a normal state, not a broken one.
    app.state.client_app_mounted = (CLIENT_APP_DIR / "index.html").is_file()

    if not app.state.client_app_mounted:
        logger.info(
            "client app not built; /app is unavailable", extra={"path": str(CLIENT_APP_DIR)}
        )
        return

    app.mount("/app", _ClientApp(directory=str(CLIENT_APP_DIR), html=True), name="client")


class _ClientApp(StaticFiles):
    """Static files, with the client router's paths falling back to `index.html`.

    `/app/documents` is a route in the browser, not a file on disk, so a plain
    static mount answers 404 to a reload or a shared link. Handled inside the
    mount rather than as a route in front of it: a route registered before the
    mount would shadow the bundles too, and one registered after would never be
    reached.

    A missing *asset* must still be a 404 -- a JavaScript request answered with
    HTML fails later and less legibly than one answered honestly.
    """

    async def get_response(self, path: str, scope: Scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or "." in Path(path).name:
                raise
            return await super().get_response("index.html", scope)


app = create_app()
