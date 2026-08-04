"""Portfolio and holding repositories."""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence

from sqlalchemy import func, select

from app.db.models.instruments import Instrument
from app.db.models.portfolio import Portfolio, PortfolioHolding
from app.db.repositories.base import BaseRepository


class PortfolioRepository(BaseRepository[Portfolio]):
    model = Portfolio

    async def get_by_name(self, name: str) -> Portfolio | None:
        result = await self.session.execute(select(Portfolio).where(Portfolio.name == name))
        return result.scalar_one_or_none()


class PortfolioHoldingRepository(BaseRepository[PortfolioHolding]):
    model = PortfolioHolding

    async def latest_as_of_date(self, portfolio_id: uuid.UUID) -> dt.date | None:
        """Most recent position date. Holdings are a time series, not a snapshot."""
        result = await self.session.execute(
            select(func.max(PortfolioHolding.as_of_date)).where(
                PortfolioHolding.portfolio_id == portfolio_id
            )
        )
        return result.scalar_one_or_none()

    async def list_holdings(
        self, portfolio_id: uuid.UUID, *, as_of: dt.date | None = None
    ) -> Sequence[PortfolioHolding]:
        """Holdings for a date, defaulting to the latest available."""
        if as_of is None:
            as_of = await self.latest_as_of_date(portfolio_id)
            if as_of is None:
                return []
        result = await self.session.execute(
            select(PortfolioHolding).where(
                PortfolioHolding.portfolio_id == portfolio_id,
                PortfolioHolding.as_of_date == as_of,
            )
        )
        return result.scalars().all()

    async def list_holdings_rated_at_or_below(
        self,
        rank_threshold: int,
        *,
        portfolio_id: uuid.UUID | None = None,
        as_of: dt.date | None = None,
    ) -> Sequence[tuple[PortfolioHolding, Instrument]]:
        """Exposure to instruments at or worse than a rating notch.

        Joins holdings to instruments on the denormalised rank, which is the
        shape the Phase 7 agent's `get_portfolio_holdings` tool needs.
        """
        stmt = (
            select(PortfolioHolding, Instrument)
            .join(Instrument, PortfolioHolding.instrument_id == Instrument.id)
            .where(
                Instrument.current_rating_rank.is_not(None),
                Instrument.current_rating_rank >= rank_threshold,
            )
        )
        if portfolio_id is not None:
            stmt = stmt.where(PortfolioHolding.portfolio_id == portfolio_id)
        if as_of is not None:
            stmt = stmt.where(PortfolioHolding.as_of_date == as_of)

        result = await self.session.execute(stmt.order_by(Instrument.current_rating_rank))
        return [(row[0], row[1]) for row in result.all()]
