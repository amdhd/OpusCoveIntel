"""The autogenerate filter that keeps the drift check honest.

`alembic check` is CI's guard against a model changing without a migration --
the failure that otherwise surfaces first in production, on someone else's
machine. Alembic 1.19 began reflecting CHECK constraints from the database
while still excluding type-bound ones from metadata, which made every
`Enum(create_constraint=True)` constraint look database-only and turned the
guard permanently red.

`include_object` filters those 36 out. These tests pin the filter's *scope*,
because an exclusion that quietly widened would disable drift detection
without anything going red to say so.
"""

from __future__ import annotations

from sqlalchemy import CheckConstraint, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import type_bound_check_constraint_names
from app.db.models import Base
from app.db.schema_check import enum_constraint_drift, model_enum_values
from app.domain.enums import DocumentStatus

TYPE_BOUND_CHECK_CONSTRAINTS = type_bound_check_constraint_names(Base.metadata)


def include_object(name: str, type_: str) -> bool:
    """The predicate `migrations/env.py` applies, without importing it.

    Importing `migrations.env` would *run the migrations*: the module ends by
    dispatching on `context.is_offline_mode()`.
    """
    if type_ == "table" and name in {"spatial_ref_sys"}:
        return False
    return not (type_ == "check_constraint" and name in TYPE_BOUND_CHECK_CONSTRAINTS)


def check_constraints() -> dict[str, CheckConstraint]:
    return {
        str(constraint.name): constraint
        for table in Base.metadata.tables.values()
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint) and constraint.name
    }


def test_every_excluded_constraint_is_enum_generated() -> None:
    constraints = check_constraints()

    for name in TYPE_BOUND_CHECK_CONSTRAINTS:
        assert name in constraints, f"{name} is excluded but not in the metadata"
        # `_type_bound` is SQLAlchemy's own marker for a constraint that
        # belongs to the column's type rather than to the table.
        assert getattr(constraints[name], "_type_bound", False), name


def test_business_check_constraints_are_never_excluded() -> None:
    business = {
        name
        for name, constraint in check_constraints().items()
        if not getattr(constraint, "_type_bound", False)
    }

    # confidence ranges, non-negative costs, ordered character spans: the rules
    # that make the schema self-defending. None may be filtered out.
    assert business
    assert business & TYPE_BOUND_CHECK_CONSTRAINTS == set()


def test_the_filter_excludes_only_named_check_constraints() -> None:
    sample = next(iter(TYPE_BOUND_CHECK_CONSTRAINTS))

    assert include_object(sample, "check_constraint") is False
    # The same name under any other object type is still compared.
    assert include_object(sample, "column") is True
    assert include_object(sample, "table") is True


def test_ordinary_objects_are_still_compared() -> None:
    assert include_object("documents", "table") is True
    assert include_object("ck_documents_parse_confidence_range", "check_constraint") is True
    assert include_object("ix_documents_status", "index") is True


def test_extension_owned_tables_stay_excluded() -> None:
    # The pre-existing exclusion must survive the new one.
    assert include_object("spatial_ref_sys", "table") is False


def test_the_exclusion_covers_every_enum_column() -> None:
    """Derived from metadata, so a new enum column is covered automatically.

    A name-pattern filter would silently miss one, and the drift check would go
    red on the next model that used `enum_column()`.
    """
    type_bound = {
        name
        for name, constraint in check_constraints().items()
        if getattr(constraint, "_type_bound", False)
    }

    assert TYPE_BOUND_CHECK_CONSTRAINTS == type_bound
    assert len(type_bound) >= 36


# --------------------------------------------------------------------------
# The gap the filter leaves open, and the check that closes it
# --------------------------------------------------------------------------


async def test_a_correct_database_reports_no_enum_drift(db_session: AsyncSession) -> None:
    assert await enum_constraint_drift(db_session) == []


async def test_a_narrowed_constraint_is_reported_as_drift(db_session: AsyncSession) -> None:
    """The failure `alembic check` cannot see, on any version.

    Adding a value to a StrEnum without a migration leaves the model accepting
    a value the database rejects. Alembic excludes type-bound constraints from
    autogenerate, so it reports nothing; the first symptom is an INSERT failing
    in production.

    Simulated from the database side -- dropping a value from the CHECK is
    indistinguishable from adding one to the enum, and does not require
    mutating the imported models mid-suite.
    """
    await db_session.execute(
        text("ALTER TABLE documents DROP CONSTRAINT ck_documents_ck_documentstatus")
    )
    await db_session.execute(
        text(
            "ALTER TABLE documents ADD CONSTRAINT ck_documents_ck_documentstatus "
            "CHECK (status::text = ANY (ARRAY['uploaded'::character varying]::text[]))"
        )
    )

    drift = await enum_constraint_drift(db_session)

    assert len(drift) == 1
    assert drift[0].table == "documents"
    assert drift[0].column == "status"
    assert "parsed" in drift[0].missing_in_database
    assert drift[0].missing_in_models == ()
    assert "database rejects" in drift[0].describe()


async def test_a_business_check_is_never_mistaken_for_a_vocabulary(
    db_session: AsyncSession,
) -> None:
    """`human_reviews` has a CHECK that quotes two status values.

    `status IN ('pending', 'not_required') OR reviewer_id IS NOT NULL` reads
    exactly like a narrowed vocabulary to a regex. Reporting it would train
    people to ignore this check, which is worse than not having it.
    """
    drift = await enum_constraint_drift(db_session)

    assert [item.constraint for item in drift] == []


def test_every_enum_column_is_covered() -> None:
    values = model_enum_values()

    assert ("documents", "status") in values
    assert ("covenants", "covenant_type") in values
    # One entry per enum-backed column, and the vocabulary matches the enum.
    assert values[("documents", "status")] == {status.value for status in DocumentStatus}
