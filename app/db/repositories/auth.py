"""User and session repositories."""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence

from sqlalchemy import select

from app.db.models.auth import User, UserSession
from app.db.repositories.base import BaseRepository


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
