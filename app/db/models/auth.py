"""User and session models.

The reason these exist is the audit trail. Review decisions were recorded
against whatever `reviewer_id` the client sent, so "who approved this?" was
answerable only if the caller chose to be honest. A `users` row makes the
answer a foreign key.

Sessions are rows rather than signed cookies. A signed cookie needs no table
but cannot be revoked before it expires, and "revoke that person's access now"
is a question an asset manager's compliance function will ask. Postgres is
already here; a broker is not (CLAUDE.md 9).
"""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import CheckConstraint, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, enum_column
from app.domain.enums import UserRole


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "users"

    # Stored already-normalised (lower-cased, stripped) by the service, so the
    # unique constraint means what a reader expects: one account per person,
    # not one per capitalisation.
    username: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    email: Mapped[str | None] = mapped_column(String(320))

    # `scrypt$n$r$p$salt$hash` -- see app/auth/passwords.py. Never a plaintext
    # password, and never logged.
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)
    role: Mapped[UserRole] = enum_column(UserRole, default=UserRole.ANALYST, index=True)

    # Deactivation rather than deletion: an audit row naming a deleted user is
    # a dangling reference, and the whole point of the table is that it isn't.
    is_active: Mapped[bool] = mapped_column(default=True, nullable=False)
    last_login_at: Mapped[dt.datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    sessions: Mapped[list[UserSession]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("length(username) > 0", name="username_not_empty"),
        CheckConstraint("length(password_hash) > 0", name="password_hash_not_empty"),
        # Cheap structural guard against a plaintext password reaching the
        # column through some future code path that forgot to hash.
        CheckConstraint("password_hash LIKE 'scrypt$%'", name="password_hash_is_encoded"),
    )


class UserSession(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One logged-in session.

    The row stores the SHA-256 of the token, never the token. A database dump
    therefore does not hand an attacker live sessions -- the same reasoning that
    applies to passwords, for the same reason.
    """

    __tablename__ = "user_sessions"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Hex SHA-256 of the opaque token held by the client.
    token_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    expires_at: Mapped[dt.datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    revoked_at: Mapped[dt.datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    # Recorded for the audit trail, not for authentication -- neither is
    # trustworthy enough to bind a session to.
    user_agent: Mapped[str | None] = mapped_column(String(512))
    client_ip: Mapped[str | None] = mapped_column(String(64))

    user: Mapped[User] = relationship(back_populates="sessions")

    __table_args__ = (
        CheckConstraint("length(token_fingerprint) = 64", name="fingerprint_is_sha256_hex"),
        Index("ix_user_sessions_user_id_expires_at", "user_id", "expires_at"),
    )
