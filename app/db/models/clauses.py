"""Clause, covenant, call schedule and rating trigger models.

The citation chain lives here. A `Clause` is the cited evidence -- it names the
chunk, the page, and the verbatim quote. Everything derived from it
(`Covenant`, `CallSchedule`, `RatingTrigger`) points back at a clause, so no
structured fact exists without a source (CLAUDE.md 1.2).
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    Date,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, enum_column
from app.db.models.instruments import MONEY
from app.domain.enums import (
    CallType,
    ClauseType,
    CovenantType,
    ExtractionMethod,
    ExtractionStatus,
    RatingAgency,
    ReviewStatus,
    Severity,
    TriggerDirection,
)


class Clause(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A cited span of contractual language."""

    __tablename__ = "clauses"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    instrument_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("instruments.id", ondelete="SET NULL"), index=True
    )
    # The chunk this was read from -- the anchor citation verification checks against.
    source_chunk_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="SET NULL"), index=True
    )

    clause_type: Mapped[ClauseType] = enum_column(ClauseType, index=True)
    clause_text: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    section_title: Mapped[str | None] = mapped_column(String(512))

    # Verbatim quote as returned by the extractor, plus its verified offsets.
    # CLAUDE.md 1.3: this is checked against the chunk before persistence.
    source_quote: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    citation_verified: Mapped[bool] = mapped_column(default=False, nullable=False)
    citation_match_score: Mapped[float | None] = mapped_column(Float)

    normalized_json: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    method: Mapped[ExtractionMethod] = enum_column(ExtractionMethod, default=ExtractionMethod.LLM)
    confidence: Mapped[float | None] = mapped_column(Float)
    extraction_status: Mapped[ExtractionStatus] = enum_column(
        ExtractionStatus, default=ExtractionStatus.PENDING, index=True
    )
    review_status: Mapped[ReviewStatus] = enum_column(
        ReviewStatus, default=ReviewStatus.NOT_REQUIRED, index=True
    )

    covenants: Mapped[list[Covenant]] = relationship(
        back_populates="clause", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("page_number >= 1", name="page_number_positive"),
        CheckConstraint("length(source_quote) > 0", name="source_quote_not_empty"),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
        # A verified citation must record what it scored. Prevents a code path
        # from flipping the flag without running the check.
        CheckConstraint(
            "NOT citation_verified OR citation_match_score IS NOT NULL",
            name="verified_citation_has_score",
        ),
        Index("ix_clauses_document_id_clause_type", "document_id", "clause_type"),
    )


class Covenant(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A structured obligation derived from exactly one clause."""

    __tablename__ = "covenants"

    clause_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("clauses.id", ondelete="CASCADE"), nullable=False, index=True
    )
    instrument_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("instruments.id", ondelete="SET NULL"), index=True
    )

    covenant_type: Mapped[CovenantType] = enum_column(CovenantType, index=True)
    summary: Mapped[str | None] = mapped_column(Text)

    # Machine-evaluable structure consumed by the Phase 4 rules engine.
    conditions_json: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    thresholds_json: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    # Denormalised primary monetary threshold, so "cross-default below RM50m"
    # is an indexed numeric predicate rather than a JSONB probe.
    threshold_amount: Mapped[Decimal | None] = mapped_column(MONEY, index=True)
    threshold_currency: Mapped[str | None] = mapped_column(String(3))

    effective_date: Mapped[dt.date | None] = mapped_column(Date)
    trigger_event: Mapped[str | None] = mapped_column(Text)
    severity: Mapped[Severity] = enum_column(Severity, default=Severity.MEDIUM)
    method: Mapped[ExtractionMethod] = enum_column(ExtractionMethod, default=ExtractionMethod.LLM)
    confidence: Mapped[float | None] = mapped_column(Float)
    review_status: Mapped[ReviewStatus] = enum_column(
        ReviewStatus, default=ReviewStatus.NOT_REQUIRED, index=True
    )

    clause: Mapped[Clause] = relationship(back_populates="covenants")

    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
        CheckConstraint(
            "threshold_amount IS NULL OR threshold_currency IS NOT NULL",
            name="threshold_amount_requires_currency",
        ),
        Index("ix_covenants_instrument_id_covenant_type", "instrument_id", "covenant_type"),
    )


class CallSchedule(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "call_schedules"

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_clause_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("clauses.id", ondelete="SET NULL"), index=True
    )

    call_date: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)
    # Percentage of par, e.g. 101.50.
    call_price: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    call_type: Mapped[CallType] = enum_column(CallType, default=CallType.OPTIONAL)
    conditions_json: Mapped[dict[str, object]] = mapped_column(JSONB, default=dict, nullable=False)
    method: Mapped[ExtractionMethod] = enum_column(ExtractionMethod, default=ExtractionMethod.LLM)
    confidence: Mapped[float | None] = mapped_column(Float)
    review_status: Mapped[ReviewStatus] = enum_column(
        ReviewStatus, default=ReviewStatus.NOT_REQUIRED, index=True
    )

    __table_args__ = (
        CheckConstraint("call_price IS NULL OR call_price > 0", name="call_price_positive"),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
    )


class RatingTrigger(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A rating-linked consequence.

    `trigger_rank` is the ordinal notch index of `trigger_rating`
    (see `app/rules/ratings.py`). Storing it makes "which holdings trip on a
    downgrade below A" an integer comparison in SQL. Neither source plan had
    this, and without it the flagship query is wrong: `'AA-' > 'A+'` is false
    lexically and true ordinally.
    """

    __tablename__ = "rating_triggers"

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    source_clause_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("clauses.id", ondelete="SET NULL"), index=True
    )

    rating_agency: Mapped[RatingAgency] = enum_column(RatingAgency, default=RatingAgency.UNKNOWN)
    trigger_rating: Mapped[str] = mapped_column(String(16), nullable=False)
    trigger_rank: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    trigger_direction: Mapped[TriggerDirection] = enum_column(
        TriggerDirection, default=TriggerDirection.DOWNGRADE_BELOW
    )
    consequence: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[Severity] = enum_column(Severity, default=Severity.HIGH)

    method: Mapped[ExtractionMethod] = enum_column(ExtractionMethod, default=ExtractionMethod.LLM)
    confidence: Mapped[float | None] = mapped_column(Float)
    review_status: Mapped[ReviewStatus] = enum_column(
        ReviewStatus, default=ReviewStatus.NOT_REQUIRED, index=True
    )

    __table_args__ = (
        CheckConstraint("trigger_rank >= 0", name="trigger_rank_non_negative"),
        CheckConstraint("length(trigger_rating) > 0", name="trigger_rating_not_empty"),
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
    )
