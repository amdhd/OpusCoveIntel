"""Authentication service: log in, resolve a session, log out.

Two rules shape this file.

**Login failures are indistinguishable.** A wrong username, a wrong password
and a deactivated account all produce the same `None` and the same elapsed
time. Anything else is a username oracle -- and for an internal tool at an
asset manager, "which of our analysts still has an account" is exactly the
enumeration an attacker wants.

**Only the fingerprint is stored.** The client holds an opaque token; the row
holds its SHA-256. A database dump does not yield live sessions.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.passwords import (
    hash_password,
    needs_rehash,
    new_session_token,
    token_fingerprint,
    verify_password,
)
from app.core.logging import get_logger
from app.db.models.auth import User, UserSession
from app.db.repositories.auth import UserRepository, UserSessionRepository
from app.domain.enums import UserRole

logger = get_logger(__name__)

# Cost of one scrypt hash, burned on a failed login so that a missing user and
# a wrong password take the same time. Computed once at import; the value is
# never used, only the work.
_DUMMY_HASH = hash_password("timing-equalisation-placeholder")


def normalize_username(raw: str) -> str:
    """Fold a username to its stored form.

    Case- and whitespace-insensitive, because "A.Rahman" and "a.rahman" are one
    person and a login screen should not make anyone care which they typed.
    """
    return raw.strip().lower()


@dataclass(frozen=True)
class IssuedSession:
    """A newly created session and the token to hand the client.

    The token appears here and nowhere else -- not in the row, not in a log
    line. Once this object is dropped it is unrecoverable, which is the point.
    """

    token: str
    session: UserSession
    user: User


class AuthService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self.users = UserRepository(session)
        self.sessions = UserSessionRepository(session)

    # -- accounts ------------------------------------------------------------

    async def create_user(
        self,
        *,
        username: str,
        password: str,
        display_name: str | None = None,
        email: str | None = None,
        role: UserRole = UserRole.ANALYST,
    ) -> User:
        """Create an account. Raises ValueError if the username is taken."""
        normalized = normalize_username(username)
        if not normalized:
            raise ValueError("username must not be empty")
        if await self.users.get_by_username(normalized) is not None:
            raise ValueError(f"username {normalized!r} already exists")

        user = User(
            username=normalized,
            display_name=display_name or normalized,
            email=email,
            password_hash=hash_password(password),
            role=role,
        )
        created = await self.users.add(user)
        logger.info("auth.user_created", extra={"username": normalized, "role": role.value})
        return created

    async def set_password(self, user: User, password: str) -> User:
        """Replace a password and revoke every session the user holds.

        Revoking is not optional politeness: a password change is usually a
        response to a suspected compromise, and leaving live sessions alone
        would defeat the reason for it.
        """
        user.password_hash = hash_password(password)
        await self._session.flush()
        revoked = await self.sessions.revoke_all_for_user(user.id, now=_now())
        logger.info(
            "auth.password_changed",
            extra={"username": user.username, "sessions_revoked": revoked},
        )
        return user

    # -- login / logout ------------------------------------------------------

    async def authenticate(self, username: str, password: str) -> User | None:
        """Verify credentials. Returns None for every kind of failure.

        The dummy verification on the miss path is load-bearing: without it a
        request for a non-existent user returns in microseconds while a real
        user with a wrong password takes ~170ms, and that difference alone
        enumerates the user table.
        """
        user = await self.users.get_by_username(normalize_username(username))

        if user is None:
            verify_password(password, _DUMMY_HASH)
            logger.info("auth.login_failed", extra={"reason": "unknown_user"})
            return None

        if not verify_password(password, user.password_hash):
            logger.info(
                "auth.login_failed",
                extra={"reason": "bad_password", "username": user.username},
            )
            return None

        if not user.is_active:
            logger.info(
                "auth.login_failed",
                extra={"reason": "inactive", "username": user.username},
            )
            return None

        # Transparent upgrade if the cost parameters have been raised since
        # this password was set. Only reachable with a correct password.
        if needs_rehash(user.password_hash):
            user.password_hash = hash_password(password)
            logger.info("auth.password_rehashed", extra={"username": user.username})

        return user

    async def start_session(
        self,
        user: User,
        *,
        ttl: dt.timedelta,
        user_agent: str | None = None,
        client_ip: str | None = None,
    ) -> IssuedSession:
        token = new_session_token()
        now = _now()
        session = UserSession(
            user_id=user.id,
            token_fingerprint=token_fingerprint(token),
            expires_at=now + ttl,
            user_agent=(user_agent or None) and user_agent[:512],
            client_ip=client_ip,
        )
        await self.sessions.add(session)
        user.last_login_at = now
        await self._session.flush()

        logger.info(
            "auth.login",
            extra={"username": user.username, "role": user.role.value},
        )
        return IssuedSession(token=token, session=session, user=user)

    async def resolve(self, token: str) -> User | None:
        """The authenticated user behind a token, or None.

        Also refuses a session whose user was deactivated after logging in --
        the session row is still live, but the account is not, and the account
        is what authorises.
        """
        if not token:
            return None

        session = await self.sessions.get_live_by_fingerprint(token_fingerprint(token), now=_now())
        if session is None:
            return None

        user = await self.users.get(session.user_id)
        if user is None or not user.is_active:
            return None
        return user

    async def end_session(self, token: str) -> bool:
        """Revoke one session. True if a live session was actually revoked."""
        session = await self.sessions.get_live_by_fingerprint(token_fingerprint(token), now=_now())
        if session is None:
            return False
        session.revoked_at = _now()
        await self._session.flush()
        logger.info("auth.logout", extra={"user_id": str(session.user_id)})
        return True

    async def revoke_all(self, user_id: uuid.UUID) -> int:
        return await self.sessions.revoke_all_for_user(user_id, now=_now())


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)
