"""Shared test fixtures.

CLAUDE.md 7: CI must never hit a paid API. `pytest_collection_modifyitems` below
enforces that structurally -- tests marked `live_llm` are skipped unless
RUN_LIVE_LLM_TESTS=1 is set explicitly.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

if TYPE_CHECKING:
    # Import-time only: the fixtures below import `app` lazily so the
    # environment is pinned before settings are ever constructed.
    from app.ingest.service import IngestionService
    from app.ingest.storage import LocalFileStore

# Pin the environment before anything imports settings.
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("LOG_LEVEL", "WARNING")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip billable tests unless explicitly opted in."""
    if os.getenv("RUN_LIVE_LLM_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="billable; set RUN_LIVE_LLM_TESTS=1 to run")
    for item in items:
        if "live_llm" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Drop cached settings/engines so env monkeypatching takes effect per test."""
    from app.core.config import get_settings
    from app.db.session import (
        get_engine,
        get_readonly_engine,
        get_readonly_sessionmaker,
        get_sessionmaker,
    )
    from app.ingest.storage import get_object_store

    caches = (
        get_settings,
        get_engine,
        get_readonly_engine,
        get_sessionmaker,
        get_readonly_sessionmaker,
        get_object_store,
    )
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A TestClient over the real app. No database required."""
    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


# --------------------------------------------------------------------------
# Database fixtures
#
# Repository tests run against a real Postgres, because the things worth
# testing here -- CHECK constraints, unique constraints, pgvector columns,
# ON DELETE CASCADE -- do not exist in SQLite. Faking the database would test
# the fake.
#
# Isolation is by transaction rollback rather than truncation: each test runs
# inside a transaction that is never committed, so tests cannot see each
# other's writes and the database is unchanged afterwards.
#
# Tests run against a DEDICATED database (`<db>_test`), created on first use.
# Rollback isolates tests from each other but not from rows that were already
# committed, so pointing the suite at the development database would make
# results depend on whether `make seed` had been run. Override with
# TEST_DATABASE_URL.
# --------------------------------------------------------------------------

_schema_ready = False


def _test_database_url() -> URL:
    from app.core.config import get_settings

    if override := os.getenv("TEST_DATABASE_URL"):
        return make_url(override)
    url = make_url(str(get_settings().DATABASE_URL))
    return url.set(database=f"{url.database}_test")


async def _ensure_test_database(url: URL) -> None:
    """Create the test database and schema once per session."""
    global _schema_ready
    if _schema_ready:
        return

    # CREATE DATABASE cannot run inside a transaction, hence AUTOCOMMIT, and it
    # must be issued from a different database -- use the default `postgres`.
    admin = create_async_engine(
        url.set(database="postgres").render_as_string(hide_password=False),
        poolclass=NullPool,
        isolation_level="AUTOCOMMIT",
    )
    try:
        async with admin.connect() as conn:
            exists = await conn.scalar(
                text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": url.database},
            )
            if not exists:
                await conn.execute(text(f'CREATE DATABASE "{url.database}"'))
    finally:
        await admin.dispose()

    from app.db.models import Base

    engine = create_async_engine(url.render_as_string(hide_password=False), poolclass=NullPool)
    try:
        async with engine.begin() as conn:
            # pgvector must exist before a Vector column can be created.
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS unaccent"))
            await conn.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()

    _schema_ready = True


@pytest_asyncio.fixture
async def db_engine() -> AsyncIterator[AsyncEngine]:
    """A dedicated engine per test, bound to the test database.

    Function-scoped deliberately: asyncpg connections bind to the event loop
    that created them, and pytest-asyncio gives each test a fresh loop. A
    session-scoped engine would hand out connections tied to a closed loop.
    """
    url = _test_database_url()
    try:
        await _ensure_test_database(url)
    except Exception as exc:  # noqa: BLE001 -- fixture reports, never fails the suite
        pytest.skip(f"postgres unavailable ({exc.__class__.__name__}); run `make up`")

    engine = create_async_engine(url.render_as_string(hide_password=False), poolclass=NullPool)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    """A session whose work is always rolled back."""
    async with db_engine.connect() as connection:
        transaction = await connection.begin()
        session = AsyncSession(
            bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
        )
        try:
            yield session
        finally:
            await session.close()
            if transaction.is_active:
                await transaction.rollback()


# --------------------------------------------------------------------------
# Ingestion fixtures
# --------------------------------------------------------------------------


@pytest.fixture
def storage_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point object storage at a temp directory for the duration of one test."""
    root = tmp_path / "storage"
    monkeypatch.setenv("STORAGE_DIR", str(root))
    return root


@pytest.fixture
def object_store(storage_root: Path) -> LocalFileStore:
    from app.ingest.storage import LocalFileStore

    return LocalFileStore(storage_root)


@pytest.fixture
def ingestion_service(db_session: AsyncSession, object_store: LocalFileStore) -> IngestionService:
    from app.ingest.service import IngestionService

    return IngestionService(db_session, object_store)
