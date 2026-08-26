"""What the query agent's database role can actually read.

`app/agent/sql_guard.py` keeps six operational tables out of the agent's
allowlist, and `docker/postgres-init.sql` calls the Postgres grant "the actual
boundary" with that allowlist as "defence in depth". Those two statements were
not both true: the init script grants SELECT on every table, so for
`audit_logs`, `extraction_jobs`, `human_reviews`, `llm_cache`, `llm_calls` and
`query_logs` the boundary existed only in the SQL parser.

These tests run against a real role holding real grants, because that is the
only place the claim can be checked. Three things are pinned:

* the role really can read those tables *before* the migration runs -- without
  this, every denial below would pass in a database where the grant was simply
  never made, which is exactly the state of the test database,
* the migration's own SQL is what produces the denial (the statement is
  imported from the migration, not retyped here), and
* what the role can read afterwards is precisely `ALLOWED_TABLES`, so a table
  added later cannot quietly inherit SELECT from the init script's
  `ALTER DEFAULT PRIVILEGES` without this going red.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from app.agent.sql_guard import ALLOWED_TABLES
from app.db.models import Base

VERSIONS = Path(__file__).resolve().parents[1] / "migrations" / "versions"
REVOKE_MIGRATION = "20260810_0733_revoke_operational_tables_from_readonly.py"


def _load_migration(path: Path) -> ModuleType:
    """Import a revision module by path.

    `migrations/versions` is not a package, and Alembic loads revisions the
    same way. Importing one is safe: they define constants and functions and
    touch `op` only inside `upgrade()`/`downgrade()`.
    """
    spec = importlib.util.spec_from_file_location(f"_migration_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


migration = _load_migration(VERSIONS / REVOKE_MIGRATION)

# Every migration that narrows the read-only role publishes its statement as
# `revoke_sql()`. Collecting them means a new one is picked up by the sweep
# below without anybody remembering to add it here -- and a new table that
# arrives *without* a revoke is exactly what the sweep is for.
REVOKING_MIGRATIONS = [
    module
    for module in (_load_migration(path) for path in sorted(VERSIONS.glob("*.py")))
    if hasattr(module, "revoke_sql")
]

# Replays the Phase 9 migration (`20260807_0652_users_and_sessions.py`), which
# revoked these two the same way. Setup, not the subject: it is here so the
# end state of the probe database matches production, and so the sweep below
# compares against the whole boundary rather than half of it.
_PHASE_9_REVOKE = "REVOKE ALL ON TABLE users, user_sessions FROM {role};"


def _readonly_url(test_database: str) -> URL:
    """The configured read-only credentials, pointed at the test database."""
    from app.core.config import get_settings

    return make_url(str(get_settings().DATABASE_URL_RO)).set(database=test_database)


class _Probe:
    """Admin and read-only handles on one database, for one test."""

    def __init__(self, admin: AsyncEngine, readonly: AsyncEngine, role: str) -> None:
        self._admin = admin
        self._readonly = readonly
        self.role = role

    async def admin_execute(self, statement: str) -> None:
        async with self._admin.connect() as conn:
            await conn.execute(text(statement))

    async def select_as_readonly(self, table: str) -> None:
        """`SELECT 1 FROM <table>` as the read-only role. Raises if refused.

        A fresh connection per call: a refused statement poisons its
        transaction, and reusing one would make the second check fail for the
        wrong reason.
        """
        async with self._readonly.connect() as conn:
            await conn.execute(text(f"SELECT 1 FROM {table} LIMIT 1"))

    async def may_select(self, table: str) -> bool:
        """Ask the catalogue rather than the table. No connection as the role.

        `has_table_privilege` accounts for grants held directly and through
        PUBLIC, which is what "can this role read it" means.
        """
        async with self._admin.connect() as conn:
            granted = await conn.scalar(
                text("SELECT has_table_privilege(:role, :table, 'SELECT')"),
                {"role": self.role, "table": table},
            )
        return bool(granted)


@pytest_asyncio.fixture
async def probe(db_engine: AsyncEngine) -> AsyncIterator[_Probe]:
    """A read-only role holding the grants `docker/postgres-init.sql` would give it.

    The role is cluster-wide, but grants are per-database: everything here
    happens in the test database, so a developer's `opuscovintel` database is
    untouched. The role is created only if it is missing (CI has none) and
    dropped only if this fixture created it.
    """
    from tests.conftest import _test_database_url

    database = str(_test_database_url().database)
    url = _readonly_url(database)
    role = migration.READONLY_ROLE
    # Config and migration must name the same role; if they diverge, this
    # fixture would grant to one and revoke from the other and prove nothing.
    assert url.username == role, f"DATABASE_URL_RO uses {url.username!r}, migration uses {role!r}"

    # AUTOCOMMIT because grants have to be visible to the read-only role's own
    # connection, which cannot see an open transaction's work.
    admin = create_async_engine(
        _test_database_url().render_as_string(hide_password=False),
        poolclass=NullPool,
        isolation_level="AUTOCOMMIT",
    )

    created_role = False
    async with admin.connect() as conn:
        exists = await conn.scalar(
            text("SELECT 1 FROM pg_roles WHERE rolname = :role"), {"role": role}
        )
        if not exists:
            password = url.password or role
            await conn.execute(text(f"CREATE ROLE {role} LOGIN PASSWORD '{password}'"))
            created_role = True
        # The init script's grants, minus the parts that only matter to a
        # long-lived database (default privileges for future tables).
        await conn.execute(text(f'GRANT CONNECT ON DATABASE "{database}" TO {role}'))
        await conn.execute(text(f"GRANT USAGE ON SCHEMA public TO {role}"))
        await conn.execute(text(f"GRANT SELECT ON ALL TABLES IN SCHEMA public TO {role}"))

    readonly = create_async_engine(
        url.render_as_string(hide_password=False),
        poolclass=NullPool,
    )
    try:
        yield _Probe(admin, readonly, role)
    finally:
        await readonly.dispose()
        async with admin.connect() as conn:
            await conn.execute(text(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {role}"))
            await conn.execute(text(f"REVOKE ALL ON SCHEMA public FROM {role}"))
            await conn.execute(text(f'REVOKE ALL ON DATABASE "{database}" FROM {role}'))
            if created_role:
                await conn.execute(text(f"DROP ROLE {role}"))
        await admin.dispose()


async def test_the_grant_being_revoked_is_really_there(probe: _Probe) -> None:
    """Guard on the guard.

    The test database never received the init script's blanket grant, so a
    denial proves nothing unless the grant existed first. This is the assertion
    that makes the rest of the file meaningful.
    """
    for table in migration.OPERATIONAL_TABLES:
        await probe.select_as_readonly(table)


@pytest_asyncio.fixture
async def revoked(probe: _Probe) -> _Probe:
    """`probe`, with the migration's own REVOKE applied."""
    await probe.admin_execute(migration.revoke_sql())
    return probe


@pytest.mark.parametrize("table", migration.OPERATIONAL_TABLES)
async def test_the_agent_role_is_refused_each_operational_table(
    revoked: _Probe, table: str
) -> None:
    """The boundary, at the database, per table.

    Parametrised so a partially-applied revoke names which table it missed.
    """
    with pytest.raises(DBAPIError) as caught:
        await revoked.select_as_readonly(table)

    assert "permission denied" in str(caught.value).lower()


@pytest.mark.parametrize("table", sorted(ALLOWED_TABLES))
async def test_the_covenant_tables_stay_readable(revoked: _Probe, table: str) -> None:
    """The revoke is surgical. An agent that cannot read `covenants` is broken."""
    await revoked.select_as_readonly(table)


async def test_the_readable_set_is_exactly_the_guardrail_allowlist(revoked: _Probe) -> None:
    """The whole boundary, in one assertion.

    `ALTER DEFAULT PRIVILEGES` in the init script gives every future table
    SELECT, so a table added later arrives readable by the agent's role unless
    someone thinks about it. This fails when that happens -- which is the point.
    """
    for module in REVOKING_MIGRATIONS:
        await revoked.admin_execute(module.revoke_sql())
    await revoked.admin_execute(_PHASE_9_REVOKE.format(role=revoked.role))

    readable = {table for table in Base.metadata.tables if await revoked.may_select(table)}

    assert readable == set(ALLOWED_TABLES)
