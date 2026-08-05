"""Declarative base, naming conventions, and shared column mixins."""

from __future__ import annotations

import datetime as dt
import uuid
from enum import StrEnum
from typing import Any

from sqlalchemy import CheckConstraint, MetaData, func
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.domain.ids import uuid7

# Deterministic constraint names. Without this, Alembic autogenerate emits
# migrations that cannot drop the anonymous constraints Postgres invented, and
# downgrades break.
NAMING_CONVENTION: dict[str, str] = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


def enum_column(enum_cls: type[StrEnum], **kwargs: Any) -> Any:
    """A controlled-vocabulary column: VARCHAR + CHECK, not a native PG ENUM.

    Native `CREATE TYPE` enums store compactly, but adding a value requires
    `ALTER TYPE ... ADD VALUE`, which cannot run inside a transaction block and
    so cannot be cleanly rolled back in a migration. `clause_type` and
    `covenant_type` will keep growing as new deal structures appear, so we trade
    a few bytes for migrations that are ordinary transactional DDL.

    `values_callable` stores the enum *value* ("cross_default") rather than the
    member *name* ("CROSS_DEFAULT"), keeping the database readable.
    """
    return mapped_column(
        SAEnum(
            enum_cls,
            native_enum=False,
            length=64,
            values_callable=lambda e: [m.value for m in e],
            name=f"ck_{enum_cls.__name__.lower()}",
            create_constraint=True,
        ),
        **kwargs,
    )


class UUIDPrimaryKeyMixin:
    """UUIDv7 primary key (CLAUDE.md 7). Time-ordered, so index inserts stay local."""

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid7)


class TimestampMixin:
    """UTC created/updated stamps, maintained by the database clock.

    `server_default`/`onupdate` rather than Python defaults: an application
    clock skewed against the database makes audit ordering unreliable.
    """

    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


def type_bound_check_constraint_names(metadata: MetaData) -> frozenset[str]:
    """Names of the CHECK constraints `enum_column` causes SQLAlchemy to emit.

    SQLAlchemy marks these `_type_bound`, because they belong to the column's
    *type* rather than to the table. Alembic excludes type-bound constraints
    from the metadata side of an autogenerate comparison; Alembic 1.19 began
    reflecting CHECK constraints from the database while still excluding them
    from metadata, so every one of ours started to look database-only --
    36 spurious `remove_constraint` operations and a permanently red
    `alembic check`.

    `migrations/env.py` filters this set out of the comparison. It lives here,
    beside the helper that creates the constraints, so the two cannot drift
    apart -- and it is derived from the metadata rather than matched by name
    pattern, so a new enum column is covered automatically and a hand-written
    business CHECK never is.
    """
    return frozenset(
        # `name` is typed as `str | _NoneName`; a constraint SQLAlchemy
        # generated from a naming convention always has a real one.
        str(constraint.name)
        for table in metadata.tables.values()
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
        and getattr(constraint, "_type_bound", False)
        and constraint.name is not None
    )
