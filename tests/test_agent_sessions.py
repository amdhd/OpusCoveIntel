"""The agent's two sessions, and why one will not do.

CLAUDE.md 1.6 puts the agent's read path on the read-only role, so a generated
statement that slipped past the SQL guardrail fails at the database rather than
at code review. The same agent writes `query_logs` and `audit_logs`, which that
role cannot do.

These tests pin both halves at once, because either one alone is satisfiable by
a broken implementation: running everything read-write makes the log work while
silently dropping the invariant, and running everything read-only keeps the
invariant while losing the audit trail.

The read-only session here is built the way `app/db/session.py` builds the real
one -- `default_transaction_read_only=on` on the connection -- rather than by
depending on the `opuscovintel_ro` role's grants existing in the test database.
That is the same enforcement mechanism, available without a second role.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select, text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.agent.service import AgentQueryService
from app.db.models.ops import AuditLog, QueryLog

pytestmark = pytest.mark.usefixtures("storage_root")

_QUESTION = "What is the cross-default threshold?"


@pytest_asyncio.fixture
async def readonly_session(db_engine: object) -> AsyncIterator[AsyncSession]:
    """A session that genuinely refuses writes, against the test database.

    Its own connection, not the rolled-back fixture session: read-only is a
    property of the transaction, so it cannot be layered onto a session the
    test is also writing through.
    """
    from tests.conftest import _test_database_url

    url: URL = _test_database_url()
    engine = create_async_engine(
        url.render_as_string(hide_password=False),
        poolclass=NullPool,
        connect_args={"server_settings": {"default_transaction_read_only": "on"}},
    )
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as session:
            yield session
    finally:
        await engine.dispose()


async def test_the_readonly_fixture_really_refuses_writes(
    readonly_session: AsyncSession,
) -> None:
    """Guard on the guard.

    If this session were quietly writable, every other test in this file would
    pass without proving anything at all.
    """
    with pytest.raises(DBAPIError):
        await readonly_session.execute(
            text("INSERT INTO query_logs (id, question) VALUES (gen_random_uuid(), 'nope')")
        )
    await readonly_session.rollback()


async def test_agent_answers_and_logs_with_a_read_only_read_path(
    readonly_session: AsyncSession,
    db_session: AsyncSession,
    indexed_corpus: list[object],
    seeded_universe: None,
) -> None:
    """The whole point: read-only reads *and* a durable log, together.

    With a single session this is unsatisfiable -- the log node's INSERT dies on
    the read-only connection, and the audit trail is lost precisely when the
    invariant is being honoured.
    """
    service = AgentQueryService(readonly_session, log_session=db_session)

    answer = await service.answer(_QUESTION, user_id="tester")

    assert answer.question == _QUESTION

    logged = (
        (await db_session.execute(select(QueryLog).where(QueryLog.question == _QUESTION)))
        .scalars()
        .all()
    )
    assert len(logged) == 1
    assert logged[0].user_id == "tester"

    audits = (
        (await db_session.execute(select(AuditLog).where(AuditLog.entity_id == logged[0].id)))
        .scalars()
        .all()
    )
    assert len(audits) == 1
    assert audits[0].action == "query"


async def test_reads_do_not_go_through_the_write_session(
    readonly_session: AsyncSession,
    db_session: AsyncSession,
    indexed_corpus: list[object],
    seeded_universe: None,
) -> None:
    """Only the log node may touch the read-write session.

    If retrieval quietly used the write session, the read-only role would be
    decorative -- present in configuration and bypassed in practice.
    """
    service = AgentQueryService(readonly_session, log_session=db_session)
    await service.answer(_QUESTION)

    # Everything the write session holds is a log row; nothing was read through it.
    written = {type(obj).__name__ for obj in db_session.identity_map.values()}
    assert written <= {"QueryLog", "AuditLog"}, written


async def test_single_session_caller_still_works(
    db_session: AsyncSession,
    indexed_corpus: list[object],
    seeded_universe: None,
) -> None:
    """The one-argument form stays valid for tests and read-write callers.

    It does not satisfy CLAUDE.md 1.6 on its own -- that is the caller's
    choice, and `open_agent_query_service()` is the form that does.
    """
    service = AgentQueryService(db_session)

    answer = await service.answer(_QUESTION)

    assert answer.question == _QUESTION
    logged = (
        (await db_session.execute(select(QueryLog).where(QueryLog.question == _QUESTION)))
        .scalars()
        .all()
    )
    assert len(logged) == 1


async def test_shared_session_is_not_committed_by_the_log_node(
    db_session: AsyncSession,
    indexed_corpus: list[object],
    seeded_universe: None,
) -> None:
    """A shared session keeps the caller's transaction boundary.

    The log node commits only the session it owns. Committing a shared one
    would end the caller's transaction underneath it -- and would break this
    suite's rollback isolation, which is how it would first be noticed.
    """
    await AgentQueryService(db_session).answer(_QUESTION)

    assert db_session.in_transaction()
