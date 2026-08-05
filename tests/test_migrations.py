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

from sqlalchemy import CheckConstraint

from app.db.base import type_bound_check_constraint_names
from app.db.models import Base

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
