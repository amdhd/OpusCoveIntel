"""Instrument and sukuk structure repositories."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select

from app.db.models.instruments import Instrument, SukukStructure
from app.db.repositories.base import BaseRepository
from app.rules.ratings import try_rank


class InstrumentRepository(BaseRepository[Instrument]):
    model = Instrument

    async def get_by_isin(self, isin: str) -> Instrument | None:
        result = await self.session.execute(select(Instrument).where(Instrument.isin == isin))
        return result.scalar_one_or_none()

    async def get_by_name(self, instrument_name: str) -> Instrument | None:
        result = await self.session.execute(
            select(Instrument).where(Instrument.instrument_name == instrument_name)
        )
        return result.scalar_one_or_none()

    async def set_rating(self, instrument: Instrument, rating: str | None) -> Instrument:
        """Set the rating and keep its ordinal rank in sync.

        The denormalised rank is what makes "rated below A" an indexed integer
        comparison (CLAUDE.md 6). Writing `current_rating` directly, without
        going through here, would leave the two fields inconsistent -- so this
        is the only supported way to change a rating.
        """
        instrument.current_rating = rating
        instrument.current_rating_rank = try_rank(rating, instrument.rating_agency)
        await self.session.flush()
        await self.session.refresh(instrument)
        return instrument

    async def list_rated_at_or_below(
        self, rank_threshold: int, *, limit: int = 200
    ) -> Sequence[Instrument]:
        """Instruments whose rating is at or worse than a notch rank.

        Lower rank is better, so "at or below A" is `rank >= rank_of('A')`.
        """
        result = await self.session.execute(
            select(Instrument)
            .where(
                Instrument.current_rating_rank.is_not(None),
                Instrument.current_rating_rank >= rank_threshold,
            )
            .order_by(Instrument.current_rating_rank)
            .limit(limit)
        )
        return result.scalars().all()


class SukukStructureRepository(BaseRepository[SukukStructure]):
    model = SukukStructure

    async def get_for_instrument(self, instrument_id: uuid.UUID) -> SukukStructure | None:
        result = await self.session.execute(
            select(SukukStructure).where(SukukStructure.instrument_id == instrument_id)
        )
        return result.scalar_one_or_none()
