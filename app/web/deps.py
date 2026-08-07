"""Identity for HTML pages.

The API answers an unauthenticated request with 401, which is right for a
client that can act on it and wrong for a browser -- a person clicking a
bookmark should land on the login form, not on a JSON error body.

So the browser dependency redirects instead, carrying the path they wanted in
`?next=` so login returns them there. That parameter is attacker-controllable
by construction, hence `safe_next`: an open redirect on a login page is a
credible phishing primitive, and the check is cheap.
"""

from __future__ import annotations

from typing import Annotated
from urllib.parse import quote, urlsplit

from fastapi import Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import _demo_user
from app.auth.service import AuthService
from app.core.config import Settings, get_settings
from app.db.models.auth import User
from app.db.session import get_session

LOGIN_PATH = "/ui/login"


class RedirectToLogin(Exception):  # noqa: N818 -- a control-flow signal, not an error
    """Raised by the page dependency; turned into a 303 by an exception handler.

    A dependency cannot return a response in place of its declared type, and
    raising `HTTPException(303)` loses the Location header on some paths. An
    explicit exception with a handler keeps the redirect in one place.
    """

    def __init__(self, next_path: str) -> None:
        self.next_path = next_path
        super().__init__(next_path)


def safe_next(raw: str | None, *, fallback: str = "/ui/ask") -> str:
    """Reduce a `next` parameter to a local path, or the fallback.

    Rejects anything with a scheme or a host, and anything not starting with a
    single `/` -- `//evil.example` is a protocol-relative URL that a browser
    treats as absolute, which is the case a naive `startswith("/")` misses.
    """
    if not raw:
        return fallback
    split = urlsplit(raw)
    if split.scheme or split.netloc:
        return fallback
    if not raw.startswith("/") or raw.startswith("//"):
        return fallback
    return raw


async def page_user(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> User:
    """The signed-in user, or a redirect to the login page."""
    if not settings.AUTH_ENABLED:
        return _demo_user()

    token = request.cookies.get(settings.SESSION_COOKIE_NAME, "")
    user = await AuthService(session).resolve(token)
    if user is None:
        wanted = request.url.path
        if request.url.query:
            wanted = f"{wanted}?{request.url.query}"
        raise RedirectToLogin(wanted)
    return user


PageUser = Annotated[User, Depends(page_user)]


def login_redirect(next_path: str) -> RedirectResponse:
    return RedirectResponse(f"{LOGIN_PATH}?next={quote(next_path, safe='')}", status_code=303)
