"""Clause, covenant, call schedule and rating trigger repositories."""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from decimal import Decimal

from sqlalchemy import func, select

from app.db.models.clauses import CallSchedule, Clause, Covenant, RatingTrigger
from app.db.repositories.base import BaseRepository
from app.domain.enums import ClauseType, CovenantType, ExtractionStatus


class ClauseRepository(BaseRepository[Clause]):
    model = Clause

    async def add_verified(self, clause: Clause) -> Clause:
        """Persist a clause, refusing any whose citation was not verified.

        CLAUDE.md 1.3 is enforced here rather than left to the caller: an
        unverified quote must never reach the database, so the repository is
        the last place that can stop it.
        """
        if not clause.citation_verified:
            raise ValueError(
                f"refusing to persist clause with unverified citation "
                f"(status={clause.extraction_status}); route it to human review instead"
            )
        return await self.add(clause)

    async def list_for_document(
        self, document_id: uuid.UUID, *, clause_type: ClauseType | None = None
    ) -> Sequence[Clause]:
        stmt = select(Clause).where(Clause.document_id == document_id)
        if clause_type is not None:
            stmt = stmt.where(Clause.clause_type == clause_type)
        result = await self.session.execute(stmt.order_by(Clause.page_number))
        return result.scalars().all()

    async def list_unverified(self, *, limit: int = 100) -> Sequence[Clause]:
        """Clauses whose citation check failed -- the review queue's inbox."""
        result = await self.session.execute(
            select(Clause)
            .where(Clause.extraction_status == ExtractionStatus.CITATION_FAILED)
            .limit(limit)
        )
        return result.scalars().all()


class CovenantRepository(BaseRepository[Covenant]):
    model = Covenant

    async def list_for_document(self, document_id: uuid.UUID) -> Sequence[Covenant]:
        """Covenants reached through their clause -- a covenant has no document
        of its own, because it exists only as a consequence of cited text."""
        result = await self.session.execute(
            select(Covenant)
            .join(Clause, Covenant.clause_id == Clause.id)
            .where(Clause.document_id == document_id)
            .order_by(Clause.page_number)
        )
        return result.scalars().all()

    async def list_with_clause_for_document(
        self, document_id: uuid.UUID
    ) -> Sequence[tuple[Covenant, Clause]]:
        """Covenants paired with the clause that evidences them.

        The eval harness reads extraction output through this rather than
        through raw SQL, so it sees the same rows every other reader does and
        stays usable against the read-only role.
        """
        result = await self.session.execute(
            select(Covenant, Clause)
            .join(Clause, Covenant.clause_id == Clause.id)
            .where(Clause.document_id == document_id)
            .order_by(Clause.page_number)
        )
        return [(covenant, clause) for covenant, clause in result.all()]

    async def list_for_clause(self, clause_id: uuid.UUID) -> Sequence[Covenant]:
        """Covenants derived from one clause.

        Needed to clear the review-queue entries that name them: those rows
        carry a polymorphic `entity_id` with no foreign key, so deleting a
        covenant leaves its queue entry behind pointing at nothing.
        """
        result = await self.session.execute(select(Covenant).where(Covenant.clause_id == clause_id))
        return result.scalars().all()

    async def count_for_document(self, document_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(Covenant)
            .join(Clause, Covenant.clause_id == Clause.id)
            .where(Clause.document_id == document_id)
        )
        return int(result.scalar_one())

    async def list_for_instrument(
        self, instrument_id: uuid.UUID, *, covenant_type: CovenantType | None = None
    ) -> Sequence[Covenant]:
        stmt = select(Covenant).where(Covenant.instrument_id == instrument_id)
        if covenant_type is not None:
            stmt = stmt.where(Covenant.covenant_type == covenant_type)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def list_below_threshold(
        self,
        covenant_type: CovenantType,
        amount: Decimal,
        *,
        currency: str = "MYR",
        limit: int = 200,
    ) -> Sequence[Covenant]:
        """e.g. "cross-default thresholds below RM50m".

        Reads the denormalised `threshold_amount` column, so this is an indexed
        numeric comparison rather than a JSONB probe.
        """
        result = await self.session.execute(
            select(Covenant)
            .where(
                Covenant.covenant_type == covenant_type,
                Covenant.threshold_currency == currency,
                Covenant.threshold_amount.is_not(None),
                Covenant.threshold_amount < amount,
            )
            .order_by(Covenant.threshold_amount)
            .limit(limit)
        )
        return result.scalars().all()


class CallScheduleRepository(BaseRepository[CallSchedule]):
    model = CallSchedule

    async def list_for_instrument(self, instrument_id: uuid.UUID) -> Sequence[CallSchedule]:
        result = await self.session.execute(
            select(CallSchedule)
            .where(CallSchedule.instrument_id == instrument_id)
            .order_by(CallSchedule.call_date)
        )
        return result.scalars().all()

    async def list_between(
        self, start: dt.date, end: dt.date, *, limit: int = 500
    ) -> Sequence[CallSchedule]:
        result = await self.session.execute(
            select(CallSchedule)
            .where(CallSchedule.call_date >= start, CallSchedule.call_date <= end)
            .order_by(CallSchedule.call_date)
            .limit(limit)
        )
        return result.scalars().all()


class RatingTriggerRepository(BaseRepository[RatingTrigger]):
    model = RatingTrigger

    async def list_triggered_at_rank(
        self, rank_threshold: int, *, limit: int = 200
    ) -> Sequence[RatingTrigger]:
        """Triggers that fire at or before a given notch rank.

        This is the flagship query -- "which holdings trip on a downgrade below
        A" -- and it works only because `trigger_rank` is an integer. Comparing
        the rating strings would put 'AA-' above 'A+', which is backwards.
        """
        result = await self.session.execute(
            select(RatingTrigger)
            .where(RatingTrigger.trigger_rank <= rank_threshold)
            .order_by(RatingTrigger.trigger_rank)
            .limit(limit)
        )
        return result.scalars().all()

    async def list_for_instrument(self, instrument_id: uuid.UUID) -> Sequence[RatingTrigger]:
        result = await self.session.execute(
            select(RatingTrigger).where(RatingTrigger.instrument_id == instrument_id)
        )
        return result.scalars().all()
