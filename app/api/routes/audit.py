"""Audit log read endpoint.

docs/plan.md, Phase 7: "audit log, audit read endpoint."

The audit log is append-only — rows are written by the review queue API, the
extraction pipeline, and the query agent. This endpoint returns the trail for
a given entity, in reverse chronological order, so every mutation can be
reconstructed.

Endpoint:
    GET /audit/{entity_type}/{entity_id}
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Path, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser
from app.core.logging import get_logger
from app.db.repositories.ops import AuditLogRepository
from app.db.session import get_session

logger = get_logger(__name__)

router = APIRouter(prefix="/audit", tags=["audit"])

DEFAULT_LIMIT = 100


# -- response schema ---------------------------------------------------------


class AuditEntry(BaseModel):
    """One row from the audit log."""

    model_config = {"from_attributes": True}

    id: str
    created_at: dt.datetime
    actor_type: str
    actor_id: str | None = None
    action: str
    entity_type: str | None = None
    entity_id: str | None = None
    payload: dict[str, object] | None = None
    request_id: str | None = None


class AuditTrailResponse(BaseModel):
    entity_type: str
    entity_id: str
    entries: list[AuditEntry]
    count: int


# -- endpoint ----------------------------------------------------------------


@router.get("/{entity_type}/{entity_id}", response_model=AuditTrailResponse)
async def get_audit_trail(
    entity_type: Annotated[str, Path(min_length=1, max_length=64)],
    entity_id: uuid.UUID,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=500)] = DEFAULT_LIMIT,
    session: AsyncSession = Depends(get_session),
) -> AuditTrailResponse:
    """Return the full audit trail for an entity, newest first.

    Each entry records who did what and when, with a payload that captures the
    before/after state. This is the structural answer to "how did this value get
    here?" — the question every compliance review asks.
    """
    repo = AuditLogRepository(session)
    entries = await repo.list_for_entity(entity_type, entity_id, limit=limit)

    return AuditTrailResponse(
        entity_type=entity_type,
        entity_id=str(entity_id),
        entries=[
            AuditEntry(
                id=str(entry.id),
                created_at=entry.created_at,
                actor_type=entry.actor_type.value,
                actor_id=entry.actor_id,
                action=entry.action,
                entity_type=entry.entity_type,
                entity_id=str(entry.entity_id) if entry.entity_id else None,
                payload=entry.payload_json,
                request_id=entry.request_id,
            )
            for entry in entries
        ],
        count=len(entries),
    )
