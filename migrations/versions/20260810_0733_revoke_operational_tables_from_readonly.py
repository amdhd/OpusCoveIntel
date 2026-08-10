"""revoke operational tables from the read-only role

Revision ID: df62ce038a72
Revises: 1be910495383
Create Date: 2026-08-10 07:33:14.996535+00:00

Moves the boundary the SQL guardrail already assumes.

`app/agent/sql_guard.py` deliberately keeps six operational tables out of the
agent's allowlist, and says why: they hold each reviewer's identity and notes,
every other user's questions and answers, raw cached model output, and the
audit trail itself -- the record of what the agent did. None of it is needed to
answer a covenant question.

The grant never followed the allowlist. `docker/postgres/init` hands
`opuscovintel_ro` SELECT on every table, present and future, and calls that
grant "the actual boundary" with the guardrail as "defence in depth". For these
six tables only the defence in depth existed: anything bypassing the SQL parser
-- a parser bug, or a future code path that uses the read-only session directly
-- reached all of them.

This is the same revoke the Phase 9 migration applied to `users` and
`user_sessions`, extended to the rest of the operational schema.
"""
from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'df62ce038a72'
down_revision: Union[str, Sequence[str], None] = '1be910495383'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


READONLY_ROLE = "opuscovintel_ro"

# The six tables `ALLOWED_TABLES` in app/agent/sql_guard.py excludes. Kept in
# alphabetical order so a future addition is a one-line diff.
OPERATIONAL_TABLES: tuple[str, ...] = (
    "audit_logs",
    "extraction_jobs",
    "human_reviews",
    "llm_cache",
    "llm_calls",
    "query_logs",
)


def _role_guarded(statement: str, *, role: str = READONLY_ROLE) -> str:
    """Wrap a grant statement so it is a no-op where the role does not exist.

    The test database is built with `Base.metadata.create_all` and CI's
    Postgres has no read-only role, so an unguarded REVOKE would fail there for
    a reason that has nothing to do with the schema.
    """
    return f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{role}') THEN
                {statement}
            END IF;
        END
        $$;
    """


def revoke_sql(*, role: str = READONLY_ROLE) -> str:
    """The statement `upgrade()` runs. Exposed so a test can run the real thing.

    A test that wrote its own REVOKE would prove Postgres works, not that this
    migration does.
    """
    tables = ", ".join(OPERATIONAL_TABLES)
    return _role_guarded(f"REVOKE ALL ON TABLE {tables} FROM {role};", role=role)


def grant_sql(*, role: str = READONLY_ROLE) -> str:
    """The inverse, for `downgrade()`.

    SELECT only -- the state before this migration was SELECT from the init
    script's blanket grant, and a downgrade must not hand back more than it
    took away.
    """
    tables = ", ".join(OPERATIONAL_TABLES)
    return _role_guarded(f"GRANT SELECT ON TABLE {tables} TO {role};", role=role)


def upgrade() -> None:
    """Upgrade schema."""
    op.execute(revoke_sql())


def downgrade() -> None:
    """Downgrade schema."""
    op.execute(grant_sql())
