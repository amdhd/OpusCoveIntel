"""User, session and login-attempt repositories."""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from sqlalchemy import ColumnElement, CursorResult, delete, func, select

from app.db.models.auth import LoginAttempt, User, UserSession
from app.db.repositories.base import BaseRepository

# Older than any row this system can hold, and older than any `since` a caller
# will pass. Used where SQL needs a "no successful login on record" sentinel:
# comparing against NULL yields NULL, which silently drops every row.
_BEFORE_EVERYTHING = dt.datetime(1970, 1, 1, tzinfo=dt.UTC)


class UserRepository(BaseRepository[User]):
    model = User

    async def get_by_username(self, username: str) -> User | None:
        """Look a user up by their already-normalised username.

        Normalisation belongs to the service, not here -- a repository that
        silently lower-cases its argument would hide the fact that the stored
        value is normalised, and the next caller would write an unnormalised
        row through `add()`.
        """
        result = await self.session.execute(select(User).where(User.username == username))
        return result.scalar_one_or_none()

    async def list_active(self, *, limit: int = 100) -> Sequence[User]:
        result = await self.session.execute(
            select(User).where(User.is_active.is_(True)).order_by(User.username).limit(limit)
        )
        return result.scalars().all()


class UserSessionRepository(BaseRepository[UserSession]):
    model = UserSession

    async def get_live_by_fingerprint(
        self, fingerprint: str, *, now: dt.datetime
    ) -> UserSession | None:
        """A session that exists, has not been revoked, and has not expired.

        All three conditions in SQL rather than in Python: an expired session
        must never be loaded into a request context in the first place, and a
        caller that has the object in hand is one `if` away from using it.
        """
        result = await self.session.execute(
            select(UserSession).where(
                UserSession.token_fingerprint == fingerprint,
                UserSession.revoked_at.is_(None),
                UserSession.expires_at > now,
            )
        )
        return result.scalar_one_or_none()

    async def revoke_all_for_user(self, user_id: uuid.UUID, *, now: dt.datetime) -> int:
        """Revoke every live session a user holds. Returns how many."""
        result = await self.session.execute(
            select(UserSession).where(
                UserSession.user_id == user_id,
                UserSession.revoked_at.is_(None),
            )
        )
        sessions = result.scalars().all()
        for session in sessions:
            session.revoked_at = now
        await self.session.flush()
        return len(sessions)


@dataclass(frozen=True)
class FailureWindow:
    """Consecutive failed attempts for one key, and when the last one was.

    "Consecutive" means since the most recent *successful* attempt for the same
    key: someone who mistypes four times and then gets in starts from zero, and
    does not spend the rest of the window one slip away from a lockout.
    """

    failures: int
    last_failure_at: dt.datetime | None


class LoginAttemptRepository(BaseRepository[LoginAttempt]):
    """The rate limiter's storage. Append-only, plus a bounded purge."""

    model = LoginAttempt

    async def record(
        self,
        *,
        username: str,
        client_ip: str | None,
        succeeded: bool,
        at: dt.datetime,
    ) -> LoginAttempt:
        """Stage one attempt. `at` is supplied, never defaulted.

        The `now()` server default would stamp every attempt in a transaction
        with that transaction's start time -- see the model docstring.
        """
        attempt = LoginAttempt(
            username=username,
            client_ip=client_ip,
            succeeded=succeeded,
            created_at=at,
        )
        self.session.add(attempt)
        await self.session.flush()
        return attempt

    async def failures_for_username(self, username: str, *, since: dt.datetime) -> FailureWindow:
        return await self._failures(LoginAttempt.username == username, since=since)

    async def failures_for_ip(self, client_ip: str, *, since: dt.datetime) -> FailureWindow:
        return await self._failures(LoginAttempt.client_ip == client_ip, since=since)

    async def _failures(self, key: ColumnElement[bool], *, since: dt.datetime) -> FailureWindow:
        """Count and time the failures that still count against `key`.

        One statement rather than two: the "since the last success" boundary is
        a subquery, so the count and the boundary cannot be read from different
        instants by a concurrent attempt.
        """
        last_success = (
            select(func.max(LoginAttempt.created_at))
            .where(key, LoginAttempt.succeeded.is_(True))
            .scalar_subquery()
        )
        result = await self.session.execute(
            select(func.count(), func.max(LoginAttempt.created_at)).where(
                key,
                LoginAttempt.succeeded.is_(False),
                LoginAttempt.created_at >= since,
                LoginAttempt.created_at > func.coalesce(last_success, _BEFORE_EVERYTHING),
            )
        )
        failures, last_failure_at = result.one()
        return FailureWindow(failures=int(failures), last_failure_at=last_failure_at)

    async def purge_for_username(self, username: str, *, before: dt.datetime) -> int:
        """Drop this username's attempts older than `before`. Returns how many.

        Scoped to one username so the DELETE rides the `(username, created_at)`
        index and stays cheap enough to run on every attempt. Rows for a
        username nobody ever tries again are left behind; a global sweep is the
        worker's job, if the table ever grows enough to need one.
        """
        # `execute()` is typed as returning the base Result, which does not
        # declare `rowcount` -- the same cast `BaseRepository.delete_by_id` makes.
        result = cast(
            "CursorResult[Any]",
            await self.session.execute(
                delete(LoginAttempt).where(
                    LoginAttempt.username == username,
                    LoginAttempt.created_at < before,
                )
            ),
        )
        return int(result.rowcount or 0)
