"""Operational models: jobs, LLM spend, cache, review queue, audit, query log.

This is where cost governance (PLAN.md 2) and auditability (CLAUDE.md 1) become
queryable rather than aspirational.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, enum_column
from app.domain.enums import (
    ActorType,
    JobStatus,
    JobType,
    LLMStage,
    QueryIntent,
    ReviewStatus,
    ReviewTrigger,
)

# USD, to the micro-dollar. Individual calls cost fractions of a cent and are
# summed across thousands of rows, so precision matters more than range.
COST_USD = Numeric(12, 6)


class ExtractionJob(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One unit of pipeline work against one document.

    CLAUDE.md 1.7: `(document_sha256, job_type, prompt_version, model_id,
    extractor_version)` is the extraction identity. The unique constraint makes
    idempotency a database guarantee -- re-running an unchanged pipeline cannot
    duplicate work or spend, even if two workers race.
    """

    __tablename__ = "extraction_jobs"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Denormalised from documents.sha256 so the identity key is self-contained
    # and survives a document row being re-created.
    document_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    job_type: Mapped[JobType] = enum_column(JobType, index=True)
    status: Mapped[JobStatus] = enum_column(JobStatus, default=JobStatus.QUEUED, index=True)

    provider: Mapped[str | None] = mapped_column(String(64))
    model_id: Mapped[str] = mapped_column(String(128), nullable=False, default="none")
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False, default="v0")
    extractor_version: Mapped[str] = mapped_column(String(64), nullable=False, default="v0")

    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_cost_usd: Mapped[Decimal] = mapped_column(
        COST_USD, default=Decimal("0"), nullable=False
    )

    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[dt.datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    finished_at: Mapped[dt.datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    __table_args__ = (
        UniqueConstraint(
            "document_sha256",
            "job_type",
            "prompt_version",
            "model_id",
            "extractor_version",
            name="uq_extraction_jobs_identity",
        ),
        CheckConstraint("estimated_cost_usd >= 0", name="cost_non_negative"),
        CheckConstraint(
            "prompt_tokens >= 0 AND completion_tokens >= 0", name="tokens_non_negative"
        ),
        Index("ix_extraction_jobs_status_job_type", "status", "job_type"),
    )


class LLMCall(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One provider request. The ledger `make cost-report` reads."""

    __tablename__ = "llm_calls"

    job_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("extraction_jobs.id", ondelete="SET NULL"), index=True
    )
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), index=True
    )

    stage: Mapped[LLMStage] = enum_column(LLMStage, index=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)

    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # PLAN.md 2: a cache_read_tokens of zero across repeated extractions means a
    # silent prompt-cache invalidator. Recorded so that is detectable.
    cache_read_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cache_write_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    estimated_cost_usd: Mapped[Decimal] = mapped_column(
        COST_USD, default=Decimal("0"), nullable=False
    )
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    cache_hit: Mapped[bool] = mapped_column(default=False, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        CheckConstraint("estimated_cost_usd >= 0", name="cost_non_negative"),
        CheckConstraint("cache_hit = false OR estimated_cost_usd = 0", name="cache_hit_is_free"),
        Index("ix_llm_calls_document_id_stage", "document_id", "stage"),
    )


class LLMCache(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Response cache keyed on prompt version + model + content hash (PLAN.md 2)."""

    __tablename__ = "llm_cache"

    cache_key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(64), nullable=False)

    response_json: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    # Cost avoided by this entry, for reporting cache savings.
    estimated_cost_usd: Mapped[Decimal] = mapped_column(
        COST_USD, default=Decimal("0"), nullable=False
    )
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    __table_args__ = (CheckConstraint("hit_count >= 0", name="hit_count_non_negative"),)


class HumanReview(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Review queue entry.

    `old_value` is retained on correction so an audit can reconstruct what the
    machine originally said (PLAN.md, Phase 7 acceptance).
    """

    __tablename__ = "human_reviews"

    entity_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    entity_id: Mapped[uuid.UUID] = mapped_column(nullable=False, index=True)
    field_name: Mapped[str] = mapped_column(String(128), nullable=False)

    old_value: Mapped[str | None] = mapped_column(Text)
    new_value: Mapped[str | None] = mapped_column(Text)
    source_quote: Mapped[str | None] = mapped_column(Text)
    page_number: Mapped[int | None] = mapped_column(Integer)
    confidence: Mapped[float | None] = mapped_column(Float)

    trigger_reason: Mapped[ReviewTrigger] = enum_column(ReviewTrigger, index=True)
    status: Mapped[ReviewStatus] = enum_column(
        ReviewStatus, default=ReviewStatus.PENDING, index=True
    )
    reviewer_id: Mapped[str | None] = mapped_column(String(255))
    review_notes: Mapped[str | None] = mapped_column(Text)
    reviewed_at: Mapped[dt.datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
        # A resolved review must say who resolved it and when.
        CheckConstraint(
            "status IN ('pending', 'not_required') "
            "OR (reviewer_id IS NOT NULL AND reviewed_at IS NOT NULL)",
            name="resolved_review_has_reviewer",
        ),
        Index("ix_human_reviews_status_trigger_reason", "status", "trigger_reason"),
    )


class AuditLog(UUIDPrimaryKeyMixin, Base):
    """Append-only record of every consequential action.

    No `TimestampMixin`: audit rows are immutable, so an `updated_at` would be
    a lie. `created_at` is defined directly.
    """

    __tablename__ = "audit_logs"

    created_at: Mapped[dt.datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    actor_type: Mapped[ActorType] = enum_column(ActorType, default=ActorType.SYSTEM, index=True)
    actor_id: Mapped[str | None] = mapped_column(String(255))
    action: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    entity_type: Mapped[str | None] = mapped_column(String(64), index=True)
    entity_id: Mapped[uuid.UUID | None] = mapped_column(index=True)
    payload_json: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)

    __table_args__ = (Index("ix_audit_logs_entity_type_entity_id", "entity_type", "entity_id"),)


class QueryLog(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One /query invocation, with everything needed to reconstruct the answer."""

    __tablename__ = "query_logs"

    user_id: Mapped[str | None] = mapped_column(String(255), index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[QueryIntent] = enum_column(
        QueryIntent, default=QueryIntent.UNSUPPORTED, index=True
    )

    retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    tools_called: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    # Every generated statement is logged, whether or not it executed (PLAN.md 5).
    sql_generated: Mapped[str | None] = mapped_column(Text)

    answer: Mapped[str | None] = mapped_column(Text)
    citations_json: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, default=list, nullable=False
    )
    confidence: Mapped[float | None] = mapped_column(Float)
    refused: Mapped[bool] = mapped_column(default=False, nullable=False)

    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    estimated_cost_usd: Mapped[Decimal] = mapped_column(
        COST_USD, default=Decimal("0"), nullable=False
    )
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)

    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
        # CLAUDE.md 1.5: a refusal is a correct outcome, but it must not also
        # assert confidence in an answer.
        CheckConstraint(
            "NOT refused OR confidence IS NULL OR confidence = 0",
            name="refusal_has_no_confidence",
        ),
    )
