"""Instrument and sukuk structure models."""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    Date,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, enum_column
from app.domain.enums import (
    InstrumentType,
    RatingAgency,
    ReviewStatus,
    SukukStructureType,
)

# MYR issue sizes reach into the billions; 4 dp covers price/rate precision.
# CLAUDE.md 6: Decimal, never float, for anything monetary.
MONEY = Numeric(20, 4)


class Instrument(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "instruments"

    issuer_name: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    instrument_name: Mapped[str] = mapped_column(String(512), nullable=False)
    instrument_type: Mapped[InstrumentType] = enum_column(
        InstrumentType, default=InstrumentType.UNKNOWN
    )
    # ISO 4217. Malaysian issuance is overwhelmingly MYR but not exclusively.
    currency: Mapped[str] = mapped_column(String(3), default="MYR", nullable=False)
    isin: Mapped[str | None] = mapped_column(String(12), unique=True)
    ticker: Mapped[str | None] = mapped_column(String(64))

    sukuk_structure: Mapped[SukukStructureType] = enum_column(
        SukukStructureType, default=SukukStructureType.UNKNOWN
    )
    issue_size: Mapped[Decimal | None] = mapped_column(MONEY)
    maturity_date: Mapped[dt.date | None] = mapped_column(Date, index=True)

    current_rating: Mapped[str | None] = mapped_column(String(16))
    # Denormalised ordinal rank of `current_rating` (CLAUDE.md 6). Lets
    # "holdings rated below A" be an integer comparison in SQL rather than
    # application-side string logic.
    current_rating_rank: Mapped[int | None] = mapped_column(Integer, index=True)
    rating_agency: Mapped[RatingAgency] = enum_column(RatingAgency, default=RatingAgency.UNKNOWN)

    review_status: Mapped[ReviewStatus] = enum_column(
        ReviewStatus, default=ReviewStatus.NOT_REQUIRED, index=True
    )

    sukuk_structure_detail: Mapped[SukukStructure | None] = relationship(
        back_populates="instrument", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (
        CheckConstraint("char_length(currency) = 3", name="currency_is_iso4217"),
        CheckConstraint("issue_size IS NULL OR issue_size >= 0", name="issue_size_non_negative"),
        CheckConstraint(
            "current_rating_rank IS NULL OR current_rating_rank >= 0",
            name="rating_rank_non_negative",
        ),
    )


class SukukStructure(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Shariah-specific structural detail, kept separate from the instrument.

    Dissolution events and Shariah non-compliance events are distinct concepts
    (CLAUDE.md 6) and are stored as structured JSON lists rather than prose, so
    the rules engine can evaluate them.
    """

    __tablename__ = "sukuk_structures"

    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    structure_type: Mapped[SukukStructureType] = enum_column(
        SukukStructureType, default=SukukStructureType.UNKNOWN
    )
    spv_name: Mapped[str | None] = mapped_column(String(512))
    originator: Mapped[str | None] = mapped_column(String(512))
    underlying_asset: Mapped[str | None] = mapped_column(Text)

    profit_rate: Mapped[Decimal | None] = mapped_column(Numeric(9, 6))
    profit_payment_frequency: Mapped[str | None] = mapped_column(String(64))
    purchase_undertaking: Mapped[bool | None] = mapped_column()

    dissolution_events_json: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, default=list, nullable=False
    )
    shariah_compliance_events_json: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, default=list, nullable=False
    )

    confidence: Mapped[float | None] = mapped_column(Float)
    review_status: Mapped[ReviewStatus] = enum_column(
        ReviewStatus, default=ReviewStatus.NOT_REQUIRED, index=True
    )

    instrument: Mapped[Instrument] = relationship(back_populates="sukuk_structure_detail")

    __table_args__ = (
        CheckConstraint(
            "confidence IS NULL OR (confidence >= 0 AND confidence <= 1)",
            name="confidence_range",
        ),
    )
