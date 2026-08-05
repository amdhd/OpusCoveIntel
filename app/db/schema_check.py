"""Comparing a live database's enum CHECK constraints against the models.

This exists because `alembic check` cannot do it, and never could.

`enum_column()` stores controlled vocabularies as VARCHAR + CHECK. SQLAlchemy
marks those constraints `_type_bound`, and Alembic excludes type-bound
constraints from autogenerate comparison -- so **adding a value to a StrEnum
without writing a migration has always been invisible to `alembic check`**,
on every version. The model accepts `DocumentStatus.ARCHIVED`, the database
still rejects `'archived'`, and nothing notices until an INSERT fails in
production.

`migrations/env.py` filters those constraints out of autogenerate to stop
Alembic 1.19 reporting them as phantom drops. That restores the comparison to
what it always was; it does not widen the blind spot, and it does not close
it. This closes it, by asking the database directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from sqlalchemy import Enum as SAEnum
from sqlalchemy import MetaData, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.base import type_bound_check_constraint_names
from app.db.models import Base

logger = get_logger(__name__)

# Postgres renders an enum CHECK as:
#   CHECK (((status)::text = ANY ((ARRAY['uploaded'::character varying, ...])::text[])))
_COLUMN_RE: Final[re.Pattern[str]] = re.compile(r"\(\(?(?P<column>\w+)\)?::text")
_LITERAL_RE: Final[re.Pattern[str]] = re.compile(r"'([^']*)'::")

_CONSTRAINT_QUERY: Final[str] = """
    SELECT rel.relname   AS table_name,
           con.conname   AS constraint_name,
           pg_get_constraintdef(con.oid) AS definition
    FROM pg_constraint con
    JOIN pg_class rel ON rel.oid = con.conrelid
    JOIN pg_namespace nsp ON nsp.oid = rel.relnamespace
    WHERE con.contype = 'c' AND nsp.nspname = 'public'
"""


@dataclass(frozen=True)
class EnumDrift:
    """One controlled vocabulary where database and models disagree."""

    table: str
    column: str
    constraint: str
    missing_in_database: tuple[str, ...]
    missing_in_models: tuple[str, ...]

    def describe(self) -> str:
        parts = [f"{self.table}.{self.column} ({self.constraint})"]
        if self.missing_in_database:
            parts.append(
                "models allow values the database rejects: "
                + ", ".join(sorted(self.missing_in_database))
            )
        if self.missing_in_models:
            parts.append(
                "database allows values no model declares: "
                + ", ".join(sorted(self.missing_in_models))
            )
        return " -- ".join(parts)


def model_enum_values(metadata: MetaData | None = None) -> dict[tuple[str, str], frozenset[str]]:
    """Allowed values per (table, column), as the models declare them."""
    target = metadata if metadata is not None else Base.metadata
    return {
        (table.name, column.name): frozenset(column.type.enums)
        for table in target.tables.values()
        for column in table.columns
        if isinstance(column.type, SAEnum)
    }


async def enum_constraint_drift(session: AsyncSession) -> list[EnumDrift]:
    """Every controlled vocabulary where the connected database and the models disagree.

    An empty list means the two agree. A non-empty one names the values and
    the direction, because "the model gained a value" and "the database gained
    one" call for opposite fixes.
    """
    expected = model_enum_values()
    # Only the constraints `enum_column()` generates. A business rule can
    # mention the same column and quote the same literals --
    # `status IN ('pending', 'not_required') OR reviewer_id IS NOT NULL` reads
    # exactly like a narrowed vocabulary to a regex, and reporting it as drift
    # would train people to ignore this check.
    vocabulary_constraints = type_bound_check_constraint_names(Base.metadata)
    rows = (await session.execute(text(_CONSTRAINT_QUERY))).all()

    drift: list[EnumDrift] = []
    for table_name, constraint_name, definition in rows:
        if constraint_name not in vocabulary_constraints:
            continue
        column_match = _COLUMN_RE.search(definition)
        literals = frozenset(_LITERAL_RE.findall(definition))
        if column_match is None or not literals:
            continue

        column = column_match.group("column")
        declared = expected.get((table_name, column))
        if declared is None:
            continue

        if declared != literals:
            drift.append(
                EnumDrift(
                    table=table_name,
                    column=column,
                    constraint=constraint_name,
                    missing_in_database=tuple(declared - literals),
                    missing_in_models=tuple(literals - declared),
                )
            )

    logger.info(
        "enum constraint check",
        extra={"constraints_examined": len(rows), "drift": len(drift)},
    )
    return drift
