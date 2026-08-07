"""Login, logout, and "who am I".

Accounts are created from the CLI (`opuscovintel user-add`), not over HTTP.
There is no registration endpoint and no password-reset flow: this is a
single-tenant internal tool whose users are onboarded by an operator, and a
self-service reset needs an email channel the system does not have.

Endpoints:
    POST /auth/login    — exchange credentials for a session cookie
    POST /auth/logout   — revoke the current session
    GET  /auth/me       — the current user
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from app.api.deps import AuthServiceDep, CurrentUser
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.domain.enums import UserRole

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=1024)


class UserRead(BaseModel):
    """A user, minus everything that would be unwise to serve.

    No `password_hash`, and no way to add one by accident: the fields are
    listed rather than inherited from the ORM row.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    username: str
    display_name: str
    email: str | None = None
    role: UserRole
    is_active: bool
    last_login_at: dt.datetime | None = None


class LoginResponse(BaseModel):
    user: UserRead
    expires_at: dt.datetime


def _set_session_cookie(response: Response, token: str, settings: Settings) -> None:
    """Attach the session cookie.

    `httponly` so no script can read it, `samesite=lax` so a cross-site POST
    cannot ride it, and `secure` in any environment configured for HTTPS --
    `Settings` refuses to start a production without it.
    """
    response.set_cookie(
        settings.SESSION_COOKIE_NAME,
        token,
        max_age=settings.SESSION_TTL_HOURS * 3600,
        httponly=True,
        samesite="lax",
        secure=settings.SESSION_COOKIE_SECURE,
        path="/",
    )


@router.post("/login", response_model=LoginResponse, summary="Exchange credentials for a session")
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    service: AuthServiceDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> LoginResponse:
    """Authenticate and issue a session cookie.

    Every failure is the same 401 with the same message. Distinguishing "no
    such user" from "wrong password" would turn this endpoint into a directory
    of who works here.
    """
    user = await service.authenticate(body.username, body.password)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid username or password")

    issued = await service.start_session(
        user,
        ttl=settings.session_ttl,
        user_agent=request.headers.get("user-agent"),
        client_ip=request.client.host if request.client else None,
    )
    _set_session_cookie(response, issued.token, settings)

    return LoginResponse(
        user=UserRead.model_validate(user),
        expires_at=issued.session.expires_at,
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT, summary="Revoke this session")
async def logout(
    request: Request,
    response: Response,
    service: AuthServiceDep,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Response:
    """Revoke the session and clear the cookie.

    Deliberately 204 whether or not a live session was found. Logging out is
    idempotent, and reporting "you were not logged in" tells an unauthenticated
    caller something about a token they hold.
    """
    token = request.cookies.get(settings.SESSION_COOKIE_NAME, "")
    if token:
        await service.end_session(token)

    response.delete_cookie(settings.SESSION_COOKIE_NAME, path="/")
    response.status_code = status.HTTP_204_NO_CONTENT
    return response


@router.get("/me", response_model=UserRead, summary="The authenticated user")
async def me(user: CurrentUser) -> UserRead:
    return UserRead.model_validate(user)
