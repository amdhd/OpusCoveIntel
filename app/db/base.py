"""Declarative base, naming conventions, and shared column mixins."""

from __future__ import annotations

import datetime as dt
import uuid
from enum import StrEnum
from typing import Any

from sqlalchemy import Enum as SAEnum
from sqlalchemy import MetaData, func
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
