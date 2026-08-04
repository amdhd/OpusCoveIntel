"""Portfolio and holding models.

Kept minimal (two tables) but not dropped: the queries that justify the whole
system -- "which *holdings* trip on a downgrade below A" -- are portfolio-level.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal

from sqlalchemy import (
    CheckConstraint,
    Date,
    ForeignKey,
    Index,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin
from app.db.models.instruments import MONEY


class Portfolio(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "portfolios"

    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    owner: Mapped[str | None] = mapped_column(String(255))
    mandate_type: Mapped[str | None] = mapped_column(String(128))
    base_currency: Mapped[str] = mapped_column(String(3), default="MYR", nullable=False)

    holdings: Mapped[list[PortfolioHolding]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("char_length(base_currency) = 3", name="base_currency_is_iso4217"),
    )


class PortfolioHolding(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A position, as of a date.

    Holdings are a time series: the same instrument appears once per
    `as_of_date`, so historical exposure stays queryable rather than being
    overwritten on each import.
    """

    __tablename__ = "portfolio_holdings"

    portfolio_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True
    )
    instrument_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("instruments.id", ondelete="CASCADE"), nullable=False, index=True
    )

    quantity: Mapped[Decimal | None] = mapped_column(MONEY)
    market_value: Mapped[Decimal | None] = mapped_column(MONEY)
    # Fraction of NAV in [0, 1] -- not a percentage. 6 dp resolves a basis point.
    nav_weight: Mapped[Decimal | None] = mapped_column(Numeric(9, 6))
    as_of_date: Mapped[dt.date] = mapped_column(Date, nullable=False, index=True)
    source: Mapped[str | None] = mapped_column(String(128))

    portfolio: Mapped[Portfolio] = relationship(back_populates="holdings")

    __table_args__ = (
        UniqueConstraint(
            "portfolio_id",
            "instrument_id",
            "as_of_date",
            name="uq_portfolio_holdings_portfolio_id_instrument_id_as_of_date",
        ),
        CheckConstraint(
            "nav_weight IS NULL OR (nav_weight >= 0 AND nav_weight <= 1)",
            name="nav_weight_is_fraction",
        ),
        CheckConstraint(
            "market_value IS NULL OR market_value >= 0", name="market_value_non_negative"
        ),
        Index("ix_portfolio_holdings_as_of_date_portfolio_id", "as_of_date", "portfolio_id"),
    )
