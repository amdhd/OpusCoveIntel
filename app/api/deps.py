"""Request dependencies: who is calling, and may they do this.

`current_user` is the only place a request's identity is established, and
`require_reviewer` the only place a privilege is checked. Both are here rather
than in each router so that "which endpoints are protected?" is answerable by
reading one file plus the route signatures.

**`AUTH_ENABLED=false` returns a real, named user, not an anonymous bypass.**
The demo escape hatch must not create a second identity model where audit rows
suddenly carry `None`; it resolves to a fixed local account instead, so every
code path downstream sees the same shape and the audit trail says plainly that
authentication was off. `Settings` refuses to start with the flag off in
production.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.service import AuthService
from app.core.config import Settings, get_settings
from app.db.models.auth import User
from app.db.session import get_session
from app.domain.enums import UserRole

# The identity requests run as when AUTH_ENABLED=false. Not a real row -- it is
# never persisted -- but it carries a username so an audit entry written during
# a demo is still attributable to *something* and obviously not to a person.
DEMO_USERNAME = "auth-disabled"

# A fixed, obviously-synthetic id. The demo user is never persisted, so it has
# no database-assigned key -- and leaving it None made `GET /auth/me` raise a
# validation error instead of answering, which is a poor way to discover that
# authentication is switched off. All-zeros reads as "not a real account" to
# anyone who finds it in a log.
DEMO_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


def get_auth_service(
    session: Annotated[AsyncSession, Depends(get_session)],
) -> AuthService:
    return AuthService(session)


AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]


def _demo_user() -> User:
    return User(
        id=DEMO_USER_ID,
        username=DEMO_USERNAME,
        display_name="Authentication disabled",
        password_hash="scrypt$disabled",
        role=UserRole.REVIEWER,
        is_active=True,
    )


async def current_user(
    request: Request,
    service: AuthServiceDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> User:
    """The authenticated user, or 401.

    The cookie is read off the request rather than declared as a `Cookie(...)`
    parameter so the name stays configurable through settings; FastAPI would
    otherwise bake it into the signature at import time.
    """
    if not settings.AUTH_ENABLED:
        return _demo_user()

    token = request.cookies.get(settings.SESSION_COOKIE_NAME, "")
    user = await service.resolve(token)
    if user is None:
        # 401 with the header, so a browser client knows to send credentials
        # rather than concluding the resource is gone.
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="not authenticated",
            headers={"WWW-Authenticate": "Cookie"},
        )
    return user


CurrentUser = Annotated[User, Depends(current_user)]


async def require_reviewer(user: CurrentUser) -> User:
    """Gate the one privileged action: deciding a review-queue item.

    403 rather than 401 -- the caller is known, and telling them to
    re-authenticate would be a lie they could act on forever.
    """
    if not user.role.may_review:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail=f"role {user.role.value!r} may not decide review items",
        )
    return user


Reviewer = Annotated[User, Depends(require_reviewer)]

__all__ = [
    "DEMO_USERNAME",
    "DEMO_USER_ID",
    "AuthServiceDep",
    "CurrentUser",
    "Reviewer",
    "current_user",
    "get_auth_service",
    "require_reviewer",
]
