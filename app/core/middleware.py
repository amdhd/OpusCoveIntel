"""HTTP middleware.

Assigns a request id to every request, binds it into the logging context, echoes
it on the response, and emits one structured access log per request. Also sets
the security response headers (docs/review.md finding 5).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger, request_id_var

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

# Access logs for these are pure noise -- container probes hit them constantly.
_QUIET_PATHS: frozenset[str] = frozenset({"/health", "/ready", "/metrics"})


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a request id to the log context and the response headers."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        # Honour an upstream id when present so traces survive a proxy hop.
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        token = request_id_var.set(request_id)
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # Log with context before the exception handlers unwind it.
            logger.exception(
                "request failed",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )
            raise
        finally:
            request_id_var.reset(token)

        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        response.headers[REQUEST_ID_HEADER] = request_id

        if request.url.path not in _QUIET_PATHS:
            logger.info(
                "request completed",
                extra={
                    "http_method": request.method,
                    "http_path": request.url.path,
                    "http_status": response.status_code,
                    "duration_ms": duration_ms,
                },
            )

        return response


# -- security headers ---------------------------------------------------------

# The application policy. Everything the UI loads is same-origin, so the only
# concession is `data:` images -- the client app's `<link rel="icon"
# href="data:,">` placeholder.
#
# `script-src` and `style-src` inherit `'self'` from `default-src`, with **no**
# `'unsafe-inline'`. That is the property worth protecting: the UI renders
# clause text lifted verbatim out of third-party PDFs, and Jinja autoescaping is
# the only thing standing between that text and the page. CSP is the layer that
# holds when an escaping bug slips through, and `'unsafe-inline'` would remove
# it. Keeping it out is what forced `inlineCritical: false` in the client build
# -- see frontend/angular.json.
#
# `frame-ancestors 'none'` is the modern half of `X-Frame-Options: DENY`; both
# are sent because the header still covers older browsers.
APP_CSP = "; ".join(
    (
        "default-src 'self'",
        "img-src 'self' data:",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "object-src 'none'",
    )
)

# Swagger UI needs more than the application policy allows: it loads its bundle
# and stylesheet from a CDN, pulls its favicon from the FastAPI site, and
# **bootstraps itself from an inline `<script>`**. Under `APP_CSP` the page is
# blank -- verified in a browser, which is the only way this surfaces.
#
# `'unsafe-inline'` here is a deliberate, scoped exception, and the reasoning is
# the point: `/docs` is a development surface (`create_app` sets
# `docs_url=None` in production, so this policy is unreachable there), and what
# it renders is our own OpenAPI schema -- endpoint text we author. The threat
# this finding is really about is clause text lifted verbatim out of
# third-party PDFs, and none of that reaches this page. The pages that *do*
# render it keep the strict policy with no inline execution of any kind.
#
# If `/docs` is ever wanted in production, self-host the Swagger assets and
# drop this exception rather than widening `APP_CSP`.
_SWAGGER_CDN = "https://cdn.jsdelivr.net"
_FASTAPI_FAVICON = "https://fastapi.tiangolo.com"
DOCS_CSP = "; ".join(
    (
        "default-src 'self'",
        f"script-src 'self' 'unsafe-inline' {_SWAGGER_CDN}",
        f"style-src 'self' 'unsafe-inline' {_SWAGGER_CDN}",
        f"img-src 'self' data: {_SWAGGER_CDN} {_FASTAPI_FAVICON}",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "object-src 'none'",
    )
)

_DOCS_PATHS: frozenset[str] = frozenset({"/docs", "/docs/oauth2-redirect", "/redoc"})

# One year, the value browsers require before preloading is considered.
HSTS_VALUE = "max-age=31536000; includeSubDomains"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Set the security response headers on every response.

    docs/review.md finding 5. Applied as middleware rather than per-route so a
    route added later inherits it -- the failure mode of a decorator-based
    approach is a new page that silently has no policy.

    HSTS is gated on `https_only` (wired to `SESSION_COOKIE_SECURE`, the
    existing "this deployment is HTTPS" signal). Sending it over plain HTTP is
    pointless at best: browsers ignore it on an insecure origin, and if the host
    is ever reachable over HTTP on a shared domain a stale max-age is a
    self-inflicted outage.
    """

    def __init__(self, app: object, *, https_only: bool = False) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._https_only = https_only

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)

        # `setdefault`, not assignment: a handler that deliberately set its own
        # policy keeps it.
        response.headers.setdefault(
            "Content-Security-Policy",
            DOCS_CSP if request.url.path in _DOCS_PATHS else APP_CSP,
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        if self._https_only:
            response.headers.setdefault("Strict-Transport-Security", HSTS_VALUE)

        return response
