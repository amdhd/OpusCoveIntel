"""Reconcile query-role privileges without blanket future-table grants.

Revision ID: 8c2d5f0a91be
Revises: 70d74076b335
"""

from importlib.resources import files

from alembic import op

revision: str = "8c2d5f0a91be"
down_revision: str | None = "70d74076b335"
branch_labels: str | None = None
depends_on: str | None = None


def reconcile_sql() -> str:
    """The frozen v1 policy also consumed by PostgreSQL bootstrap."""
    return files("app.db").joinpath("readonly_v1.sql").read_text(encoding="utf-8")


def upgrade() -> None:
    op.execute(reconcile_sql())


def downgrade() -> None:
    # This revision changes no schema. Keep the security restriction on rollback
    # rather than restoring unrestricted reads or default privileges.
    pass
