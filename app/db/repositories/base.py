"""Generic repository.

CLAUDE.md 3: the dependency direction is `api -> services -> repositories ->
models`. Route handlers never build SQL; repositories never import from `api/`.

Repositories deliberately do **not** commit. Transaction scope belongs to the
caller (a service or request handler), so several repository calls can succeed
or fail together -- e.g. writing a clause and its covenant atomically.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from typing import Any, cast

from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.base import Base


# PEP 695 type-parameter syntax (Python 3.12+), rather than a module-level
# TypeVar plus Generic[...].
class BaseRepository[ModelT: Base]:
    """CRUD operations shared by every entity."""

    model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, entity: ModelT) -> ModelT:
        """Stage an insert and flush so server defaults and the id are populated."""
        self.session.add(entity)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def add_all(self, entities: Sequence[ModelT]) -> Sequence[ModelT]:
        self.session.add_all(list(entities))
        await self.session.flush()
        return entities

    async def get(self, entity_id: uuid.UUID) -> ModelT | None:
        return await self.session.get(self.model, entity_id)

    async def get_or_raise(self, entity_id: uuid.UUID) -> ModelT:
        entity = await self.get(entity_id)
        if entity is None:
            raise LookupError(f"{self.model.__name__} {entity_id} not found")
        return entity

    async def list(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        **filters: Any,
    ) -> Sequence[ModelT]:
        stmt = select(self.model).limit(limit).offset(offset)
        for field, value in filters.items():
            stmt = stmt.where(getattr(self.model, field) == value)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def list_by_ids(self, entity_ids: Iterable[uuid.UUID]) -> Sequence[ModelT]:
        """Fetch many rows by primary key in one statement.

        Exists so a read model that joins a list of children to their parents
        does not issue one `get()` per row. Deduplicates, and returns only the
        ids that exist -- a caller must not assume the result is the same length
        as its input.
        """
        wanted = set(entity_ids)
        if not wanted:
            return []
        result = await self.session.execute(
            select(self.model).where(self.model.id.in_(wanted))  # type: ignore[attr-defined]
        )
        return result.scalars().all()

    async def count(self, **filters: Any) -> int:
        stmt = select(func.count()).select_from(self.model)
        for field, value in filters.items():
            stmt = stmt.where(getattr(self.model, field) == value)
        result = await self.session.execute(stmt)
        return int(result.scalar_one())

    async def update(self, entity: ModelT, **values: Any) -> ModelT:
        for field, value in values.items():
            setattr(entity, field, value)
        await self.session.flush()
        await self.session.refresh(entity)
        return entity

    async def delete(self, entity: ModelT) -> None:
        await self.session.delete(entity)
        await self.session.flush()

    async def delete_by_id(self, entity_id: uuid.UUID) -> int:
        # A DELETE always yields a CursorResult, but `execute()` is typed as
        # returning the base Result, which does not declare `rowcount`.
        result = cast(
            "CursorResult[Any]",
            await self.session.execute(
                delete(self.model).where(self.model.id == entity_id)  # type: ignore[attr-defined]
            ),
        )
        await self.session.flush()
        return int(result.rowcount or 0)
