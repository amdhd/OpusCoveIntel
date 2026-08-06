"""Database engines and session factories.

Two engines, deliberately:

* `read_write` -- API writes, ingestion, extraction.
* `read_only`  -- the LangGraph query agent, and nothing else.

CLAUDE.md 1.6 makes the read-only role a hard invariant. Keeping the engines
separate here means a stray write from the agent path fails at the database,
not at a code review.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _build_engine(url: str, settings: Settings, *, readonly: bool) -> AsyncEngine:
    connect_args: dict[str, object] = {}
    if readonly:
        # Belt-and-braces alongside the role's own grants: bound every agent
        # statement in wall-clock time (PLAN.md 5).
        connect_args["server_settings"] = {
            "statement_timeout": str(settings.DB_STATEMENT_TIMEOUT_MS),
            "default_transaction_read_only": "on",
        }

    return create_async_engine(
        url,
        echo=settings.DB_ECHO,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_POOL_MAX_OVERFLOW,
        pool_pre_ping=True,
        connect_args=connect_args,
    )


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Read-write engine."""
    settings = get_settings()
    return _build_engine(str(settings.DATABASE_URL), settings, readonly=False)


@lru_cache(maxsize=1)
def get_readonly_engine() -> AsyncEngine:
    """Read-only engine. The query agent uses this and only this."""
    settings = get_settings()
    return _build_engine(str(settings.DATABASE_URL_RO), settings, readonly=True)


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)


@lru_cache(maxsize=1)
def get_readonly_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_readonly_engine(), expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a read-write session, committed on success.

    The commit belongs here rather than in each route. Without it a handler
    that flushes -- which is what the review queue does, so that it can read
    back the row it just changed -- returns 200 and then loses the write when
    the session closes. That shipped: approvals reported success and the row
    stayed `pending`.

    Services that manage their own transactions (ingestion, extraction) are
    unaffected; committing an already-committed session is a no-op.

    FastAPI throws a handler's exception into this generator, so the rollback
    covers the case a route raises after flushing -- a 409 on an already-decided
    review must leave nothing behind.
    """
    async with get_sessionmaker()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()


async def get_readonly_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a read-only session."""
    async with get_readonly_sessionmaker()() as session:
        yield session


async def check_database() -> bool:
    """Return True if the read-write database answers a trivial query."""
    try:
        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 -- readiness probe reports, never raises
        logger.warning("database readiness check failed", extra={"error": str(exc)})
        return False
    return True


async def dispose_engines() -> None:
    """Close pooled connections. Called on application shutdown."""
    for factory in (get_engine, get_readonly_engine):
        if factory.cache_info().currsize:
            await factory().dispose()
