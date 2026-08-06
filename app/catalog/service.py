"""Assemble catalogue read models from the stored rows.

The one non-obvious job here is joining every covenant to the clause that
evidences it. `CovenantRead` requires a `source`, which means this service is
where CLAUDE.md 1.2 stops being an aspiration and becomes a type error: a
covenant whose clause is missing cannot be built, and is dropped with a warning
rather than served without provenance.

That is deliberate and worth stating plainly. A covenant row with no reachable
clause is a data defect -- `covenants.clause_id` is NOT NULL with an
`ON DELETE CASCADE`, so it should be unreachable -- and the honest failure is
to omit the row and say so in the log, not to invent a citation-shaped blank
for the UI to render.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models.clauses import CallSchedule, Clause, Covenant, RatingTrigger
from app.db.repositories.clauses import (
    CallScheduleRepository,
    ClauseRepository,
    CovenantRepository,
    RatingTriggerRepository,
)
from app.db.repositories.documents import DocumentChunkRepository, DocumentRepository
from app.db.repositories.instruments import InstrumentRepository, SukukStructureRepository
from app.db.repositories.portfolio import PortfolioHoldingRepository, PortfolioRepository
from app.domain.catalog import (
    CallScheduleRead,
    ClauseProvenance,
    ClauseSource,
    CovenantRead,
    HoldingRead,
    InstrumentDetail,
    InstrumentRead,
    PortfolioHoldings,
    PortfolioRead,
    RatingTriggerRead,
    SukukStructureRead,
)
from app.domain.enums import CovenantType

logger = get_logger(__name__)


def _source(clause: Clause) -> ClauseSource:
    return ClauseSource(
        clause_id=clause.id,
        document_id=clause.document_id,
        chunk_id=clause.source_chunk_id,
        clause_type=clause.clause_type,
        page_number=clause.page_number,
        section_title=clause.section_title,
        source_quote=clause.source_quote,
        char_start=clause.char_start,
        char_end=clause.char_end,
        citation_verified=clause.citation_verified,
        citation_match_score=clause.citation_match_score,
    )


def _covenant(covenant: Covenant, clause: Clause) -> CovenantRead:
    return CovenantRead(
        id=covenant.id,
        instrument_id=covenant.instrument_id,
        covenant_type=covenant.covenant_type,
        summary=covenant.summary,
        conditions=covenant.conditions_json,
        thresholds=covenant.thresholds_json,
        threshold_amount=covenant.threshold_amount,
        threshold_currency=covenant.threshold_currency,
        effective_date=covenant.effective_date,
        trigger_event=covenant.trigger_event,
        severity=covenant.severity,
        method=covenant.method,
        confidence=covenant.confidence,
        review_status=covenant.review_status,
        source=_source(clause),
    )


class CatalogService:
    """Read-only assembly. Opens no transactions and writes nothing."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self.clauses = ClauseRepository(session)
        self.covenants = CovenantRepository(session)
        self.call_schedules = CallScheduleRepository(session)
        self.rating_triggers = RatingTriggerRepository(session)
        self.instruments = InstrumentRepository(session)
        self.sukuk_structures = SukukStructureRepository(session)
        self.portfolios = PortfolioRepository(session)
        self.holdings = PortfolioHoldingRepository(session)
        self.documents = DocumentRepository(session)
        self.chunks = DocumentChunkRepository(session)

    # -- covenants -----------------------------------------------------------

    async def _attach_clauses(self, covenants: Sequence[Covenant]) -> list[CovenantRead]:
        """Pair each covenant with its clause, dropping any that has none.

        One query for the clauses rather than one per covenant -- a covenant
        list for an instrument is otherwise N+1 against the table that every
        covenant necessarily joins.
        """
        if not covenants:
            return []

        clauses = {
            clause.id: clause
            for clause in await self.clauses.list_by_ids(
                covenant.clause_id for covenant in covenants
            )
        }

        assembled: list[CovenantRead] = []
        for covenant in covenants:
            clause = clauses.get(covenant.clause_id)
            if clause is None:
                logger.warning(
                    "catalog.covenant_without_clause",
                    extra={"covenant_id": str(covenant.id), "clause_id": str(covenant.clause_id)},
                )
                continue
            assembled.append(_covenant(covenant, clause))
        return assembled

    async def covenants_for_instrument(
        self, instrument_id: uuid.UUID, *, covenant_type: CovenantType | None = None
    ) -> list[CovenantRead]:
        rows = await self.covenants.list_for_instrument(instrument_id, covenant_type=covenant_type)
        return await self._attach_clauses(rows)

    async def covenants_for_document(self, document_id: uuid.UUID) -> list[CovenantRead]:
        pairs = await self.covenants.list_with_clause_for_document(document_id)
        return [_covenant(covenant, clause) for covenant, clause in pairs]

    # -- instruments ---------------------------------------------------------

    async def list_instruments(self, *, limit: int = 100, offset: int = 0) -> list[InstrumentRead]:
        rows = await self.instruments.list(limit=limit, offset=offset)
        return [InstrumentRead.model_validate(row) for row in rows]

    async def get_instrument(self, instrument_id: uuid.UUID) -> InstrumentDetail | None:
        instrument = await self.instruments.get(instrument_id)
        if instrument is None:
            return None

        structure = await self.sukuk_structures.get_for_instrument(instrument_id)
        calls = await self.call_schedules.list_for_instrument(instrument_id)
        triggers = await self.rating_triggers.list_for_instrument(instrument_id)

        return InstrumentDetail(
            **InstrumentRead.model_validate(instrument).model_dump(),
            sukuk_structure_detail=(
                SukukStructureRead(
                    id=structure.id,
                    structure_type=structure.structure_type,
                    spv_name=structure.spv_name,
                    originator=structure.originator,
                    underlying_asset=structure.underlying_asset,
                    profit_rate=structure.profit_rate,
                    profit_payment_frequency=structure.profit_payment_frequency,
                    purchase_undertaking=structure.purchase_undertaking,
                    dissolution_events=structure.dissolution_events_json,
                    shariah_compliance_events=structure.shariah_compliance_events_json,
                    confidence=structure.confidence,
                    review_status=structure.review_status,
                )
                if structure is not None
                else None
            ),
            covenants=await self.covenants_for_instrument(instrument_id),
            call_schedules=[_call_schedule(call) for call in calls],
            # Ordinal, not lexical (CLAUDE.md 6): the tightest trigger first.
            rating_triggers=[
                _rating_trigger(trigger)
                for trigger in sorted(triggers, key=lambda row: row.trigger_rank)
            ],
        )

    # -- provenance ----------------------------------------------------------

    async def clause_provenance(self, clause_id: uuid.UUID) -> ClauseProvenance | None:
        """A clause, its chunk, and where the quote sits inside that chunk.

        The clause's `char_start`/`char_end` are document-relative, so they are
        not usable as offsets into the chunk text. The quote is located inside
        the chunk here instead; when it cannot be found the offsets are left
        null rather than guessed, and the caller renders the quote without a
        highlight. A wrong highlight is worse than none -- it would point an
        auditor at text the extractor never read.
        """
        clause = await self.clauses.get(clause_id)
        if clause is None:
            return None

        chunk_text: str | None = None
        quote_start: int | None = None
        quote_end: int | None = None
        if clause.source_chunk_id is not None:
            chunk = await self.chunks.get(clause.source_chunk_id)
            if chunk is not None:
                chunk_text = chunk.chunk_text
                found = chunk_text.find(clause.source_quote)
                if found >= 0:
                    quote_start = found
                    quote_end = found + len(clause.source_quote)

        document = await self.documents.get(clause.document_id)
        covenants = await self.covenants.list_for_clause(clause_id)

        return ClauseProvenance(
            source=_source(clause),
            clause_text=clause.clause_text,
            method=clause.method,
            confidence=clause.confidence,
            review_status=clause.review_status,
            document_filename=document.filename if document is not None else None,
            chunk_text=chunk_text,
            quote_start=quote_start,
            quote_end=quote_end,
            covenants=[_covenant(covenant, clause) for covenant in covenants],
        )

    # -- portfolios ----------------------------------------------------------

    async def list_portfolios(self, *, limit: int = 100) -> list[PortfolioRead]:
        rows = await self.portfolios.list(limit=limit)
        return [PortfolioRead.model_validate(row) for row in rows]

    async def portfolio_holdings(
        self, portfolio_id: uuid.UUID, *, as_of: dt.date | None = None
    ) -> PortfolioHoldings | None:
        portfolio = await self.portfolios.get(portfolio_id)
        if portfolio is None:
            return None

        # Resolved rather than left to the repository's own default, because
        # the response echoes the date it actually used.
        as_of_date = as_of or await self.holdings.latest_as_of_date(portfolio_id)
        rows = await self.holdings.list_holdings(portfolio_id, as_of=as_of_date)

        instruments = {
            instrument.id: instrument
            for instrument in await self.instruments.list_by_ids(row.instrument_id for row in rows)
        }

        holdings: list[HoldingRead] = []
        for row in rows:
            instrument = instruments.get(row.instrument_id)
            if instrument is None:
                logger.warning(
                    "catalog.holding_without_instrument",
                    extra={"holding_id": str(row.id), "instrument_id": str(row.instrument_id)},
                )
                continue
            holdings.append(
                HoldingRead(
                    id=row.id,
                    instrument=InstrumentRead.model_validate(instrument),
                    quantity=row.quantity,
                    market_value=row.market_value,
                    nav_weight=row.nav_weight,
                    as_of_date=row.as_of_date,
                    source=row.source,
                )
            )

        # Decimal, never float (CLAUDE.md 6). Summing None-free values only;
        # a missing market value must not silently become zero.
        priced = [holding.market_value for holding in holdings if holding.market_value is not None]
        total: Decimal | None = sum(priced, Decimal(0)) if priced else None

        return PortfolioHoldings(
            portfolio=PortfolioRead.model_validate(portfolio),
            as_of_date=as_of_date,
            holdings=holdings,
            total_market_value=total,
            count=len(holdings),
        )


def _call_schedule(call: CallSchedule) -> CallScheduleRead:
    return CallScheduleRead(
        id=call.id,
        call_date=call.call_date,
        call_price=call.call_price,
        call_type=call.call_type,
        method=call.method,
        confidence=call.confidence,
        review_status=call.review_status,
        source_clause_id=call.source_clause_id,
    )


def _rating_trigger(trigger: RatingTrigger) -> RatingTriggerRead:
    return RatingTriggerRead(
        id=trigger.id,
        rating_agency=trigger.rating_agency,
        trigger_rating=trigger.trigger_rating,
        trigger_rank=trigger.trigger_rank,
        trigger_direction=trigger.trigger_direction,
        consequence=trigger.consequence,
        severity=trigger.severity,
        method=trigger.method,
        confidence=trigger.confidence,
        review_status=trigger.review_status,
        source_clause_id=trigger.source_clause_id,
    )
