"""Review queue API — approve, correct, or reject items flagged for human review.

PLAN.md, Phase 7: "Review queue API (approve/reject/edit with value history),
audit log, audit read endpoint."

Every mutation here writes an AuditLog row. The review queue is the structured
escape hatch for extraction disagreements, low-confidence fields, and citation
failures — everything the machine could not decide on its own.

Endpoints:
    GET  /review/pending        — list items awaiting review
    POST /review/{id}/approve   — accept the machine's value as-is
    POST /review/{id}/correct   — replace with a human-corrected value
    POST /review/{id}/reject    — dismiss the item (false positive, etc.)
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models.ops import AuditLog
from app.db.repositories.ops import HumanReviewRepository
from app.db.session import get_session
from app.domain.enums import ActorType, ReviewStatus

logger = get_logger(__name__)

router = APIRouter(prefix="/review", tags=["review"])

DEFAULT_LIMIT = 100


# -- request / response schemas ----------------------------------------------


class ReviewItem(BaseModel):
    """One item in the review queue."""

    model_config = {"from_attributes": True}

    id: str
    entity_type: str
    entity_id: str
    field_name: str
    old_value: str | None = None
    new_value: str | None = None
    source_quote: str | None = None
    page_number: int | None = None
    confidence: float | None = None
    trigger_reason: str
    status: str
    reviewer_id: str | None = None
    review_notes: str | None = None
    reviewed_at: dt.datetime | None = None
    created_at: dt.datetime


class PendingResponse(BaseModel):
    items: list[ReviewItem]
    count: int
    total_pending: int


class ApproveRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=255)
    notes: str | None = None


class CorrectRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=255)
    new_value: str = Field(min_length=1)
    notes: str | None = None


class RejectRequest(BaseModel):
    reviewer_id: str = Field(min_length=1, max_length=255)
    reason: str = Field(min_length=1)
    notes: str | None = None


class ReviewActionResponse(BaseModel):
    id: str
    status: str
    action: str
    entity_type: str
    entity_id: str
    field_name: str


# -- endpoints ---------------------------------------------------------------


@router.get("/pending", response_model=PendingResponse)
async def list_pending(
    limit: Annotated[int, Query(ge=1, le=500)] = DEFAULT_LIMIT,
    session: AsyncSession = Depends(get_session),
) -> PendingResponse:
    """List items awaiting human review."""
    repo = HumanReviewRepository(session)
    items = await repo.list_pending(limit=limit)
    total = await repo.count_pending()

    return PendingResponse(
        items=[
            ReviewItem(
                id=str(item.id),
                entity_type=item.entity_type,
                entity_id=str(item.entity_id),
                field_name=item.field_name,
                old_value=item.old_value,
                new_value=item.new_value,
                source_quote=item.source_quote,
                page_number=item.page_number,
                confidence=item.confidence,
                trigger_reason=item.trigger_reason.value,
                status=item.status.value,
                reviewer_id=item.reviewer_id,
                review_notes=item.review_notes,
                reviewed_at=item.reviewed_at,
                created_at=item.created_at,
            )
            for item in items
        ],
        count=len(items),
        total_pending=total,
    )


@router.post("/{review_id}/approve", response_model=ReviewActionResponse)
async def approve_review(
    review_id: uuid.UUID,
    body: ApproveRequest,
    session: AsyncSession = Depends(get_session),
) -> ReviewActionResponse:
    """Accept the machine-extracted value.

    The old_value stays as-is; the review is marked APPROVED. An audit log
    entry records who approved it and when.
    """
    repo = HumanReviewRepository(session)
    review = await repo.get(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail=f"review {review_id} not found")

    if review.status is not ReviewStatus.PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"review {review_id} is already {review.status.value}",
        )

    # Update the review
    review.status = ReviewStatus.APPROVED
    review.reviewer_id = body.reviewer_id
    review.review_notes = body.notes
    review.reviewed_at = dt.datetime.now(dt.UTC)
    await session.flush()

    # Write audit entry
    audit = AuditLog(
        actor_type=ActorType.USER,
        actor_id=body.reviewer_id,
        action="review_approved",
        entity_type=review.entity_type,
        entity_id=review.entity_id,
        payload_json={
            "field_name": review.field_name,
            "old_value": review.old_value,
            "trigger_reason": review.trigger_reason.value,
            "notes": body.notes,
        },
    )
    session.add(audit)
    await session.flush()

    logger.info(
        "review.approved",
        extra={
            "review_id": str(review_id),
            "reviewer": body.reviewer_id,
            "entity_type": review.entity_type,
        },
    )

    return ReviewActionResponse(
        id=str(review.id),
        status=review.status.value,
        action="approved",
        entity_type=review.entity_type,
        entity_id=str(review.entity_id),
        field_name=review.field_name,
    )


@router.post("/{review_id}/correct", response_model=ReviewActionResponse)
async def correct_review(
    review_id: uuid.UUID,
    body: CorrectRequest,
    session: AsyncSession = Depends(get_session),
) -> ReviewActionResponse:
    """Replace the machine-extracted value with a human-corrected one.

    The original `old_value` is preserved so the audit trail can reconstruct
    what the machine said. `new_value` records the human correction. An audit
    log entry captures all three states (old, new, corrected).
    """
    repo = HumanReviewRepository(session)
    review = await repo.get(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail=f"review {review_id} not found")

    if review.status is not ReviewStatus.PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"review {review_id} is already {review.status.value}",
        )

    # Preserve what the machine said
    machine_value = review.old_value

    # Update with correction
    review.new_value = body.new_value
    review.status = ReviewStatus.CORRECTED
    review.reviewer_id = body.reviewer_id
    review.review_notes = body.notes
    review.reviewed_at = dt.datetime.now(dt.UTC)
    await session.flush()

    # Write audit entry preserving the full history
    audit = AuditLog(
        actor_type=ActorType.USER,
        actor_id=body.reviewer_id,
        action="review_corrected",
        entity_type=review.entity_type,
        entity_id=review.entity_id,
        payload_json={
            "field_name": review.field_name,
            "old_value": machine_value,
            "new_value": body.new_value,
            "trigger_reason": review.trigger_reason.value,
            "notes": body.notes,
        },
    )
    session.add(audit)
    await session.flush()

    logger.info(
        "review.corrected",
        extra={
            "review_id": str(review_id),
            "reviewer": body.reviewer_id,
            "field": review.field_name,
        },
    )

    return ReviewActionResponse(
        id=str(review.id),
        status=review.status.value,
        action="corrected",
        entity_type=review.entity_type,
        entity_id=str(review.entity_id),
        field_name=review.field_name,
    )


@router.post("/{review_id}/reject", response_model=ReviewActionResponse)
async def reject_review(
    review_id: uuid.UUID,
    body: RejectRequest,
    session: AsyncSession = Depends(get_session),
) -> ReviewActionResponse:
    """Reject a review item (false positive, not applicable, etc.).

    The original value is preserved in the audit trail. A reason is required
    so the rejection can be audited and the extractor can be improved.
    """
    repo = HumanReviewRepository(session)
    review = await repo.get(review_id)
    if review is None:
        raise HTTPException(status_code=404, detail=f"review {review_id} not found")

    if review.status is not ReviewStatus.PENDING:
        raise HTTPException(
            status_code=409,
            detail=f"review {review_id} is already {review.status.value}",
        )

    review.status = ReviewStatus.REJECTED
    review.reviewer_id = body.reviewer_id
    review.review_notes = body.notes or body.reason
    review.reviewed_at = dt.datetime.now(dt.UTC)
    await session.flush()

    # Write audit entry
    audit = AuditLog(
        actor_type=ActorType.USER,
        actor_id=body.reviewer_id,
        action="review_rejected",
        entity_type=review.entity_type,
        entity_id=review.entity_id,
        payload_json={
            "field_name": review.field_name,
            "old_value": review.old_value,
            "rejection_reason": body.reason,
            "trigger_reason": review.trigger_reason.value,
            "notes": body.notes,
        },
    )
    session.add(audit)
    await session.flush()

    logger.info(
        "review.rejected",
        extra={
            "review_id": str(review_id),
            "reviewer": body.reviewer_id,
            "reason": body.reason[:120],
        },
    )

    return ReviewActionResponse(
        id=str(review.id),
        status=review.status.value,
        action="rejected",
        entity_type=review.entity_type,
        entity_id=str(review.entity_id),
        field_name=review.field_name,
    )
