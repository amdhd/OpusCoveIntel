"""Deterministic tools the agent can call. No LLM in the tools themselves.

PLAN.md 5: all tools are deterministic. The model *plans* which tools to call;
the tools *execute* against the database and the rules engine. This boundary is
what makes the verify node possible — every tool return is traceable to a row or
a computation.

Each tool takes an `AsyncSession` and returns a typed result. Tools own no state
and import nothing from `app/api/`.
"""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.sql_guard import SQLGuardError, validate_sql
from app.core.logging import get_logger
from app.db.models.clauses import CallSchedule, Clause, Covenant, RatingTrigger
from app.db.models.documents import Document
from app.db.models.instruments import Instrument
from app.db.models.portfolio import Portfolio, PortfolioHolding
from app.domain.enums import CovenantType, RatingAgency
from app.domain.rules import (
    Citation,
    CovenantTerms,
    ObservedFacts,
)
from app.retrieval.hybrid import HybridSearcher
from app.rules.covenants import evaluate
from app.rules.ratings import UnknownRatingError, rank

logger = get_logger(__name__)


# -- tool result types -------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    """Every tool returns one of these, so the agent can inspect what happened."""

    tool_name: str
    ok: bool
    data: Any = None
    error: str = ""
    citations: list[Citation] = field(default_factory=list)


# -- tool implementations ----------------------------------------------------


async def search_clauses(
    session: AsyncSession,
    query: str,
    *,
    limit: int = 5,
    document_id: uuid.UUID | None = None,
) -> ToolResult:
    """Hybrid search (vector + FTS) over the chunk corpus.

    Returns a list of SearchHit carrying chunks with page numbers, offsets, and
    the retrieval path (which legs found them).
    """
    searcher = HybridSearcher(session)
    hits = await searcher.search(query, limit=limit, document_id=document_id)
    citations = [
        Citation(
            document_id=str(hit.chunk.document_id),
            chunk_id=str(hit.chunk.id),
            page_number=hit.chunk.page_number,
            quote=hit.chunk.chunk_text[:600],
            char_start=hit.chunk.char_start,
            char_end=hit.chunk.char_end,
            section_title=hit.chunk.section_title,
        )
        for hit in hits
    ]
    return ToolResult(
        tool_name="search_clauses",
        ok=True,
        data={"hits": hits, "count": len(hits)},
        citations=citations,
    )


async def get_instrument(
    session: AsyncSession,
    *,
    instrument_id: uuid.UUID | None = None,
    isin: str | None = None,
    name: str | None = None,
) -> ToolResult:
    """Look up an instrument by id, ISIN, or name."""
    if instrument_id is not None:
        result = await session.execute(select(Instrument).where(Instrument.id == instrument_id))
    elif isin is not None:
        result = await session.execute(select(Instrument).where(Instrument.isin == isin))
    elif name is not None:
        result = await session.execute(
            select(Instrument).where(Instrument.instrument_name.ilike(f"%{name}%"))
        )
    else:
        # No filter — return all instruments (used by breach check and instrument lookup)
        result = await session.execute(select(Instrument))

    instruments = list(result.scalars().all())
    return ToolResult(
        tool_name="get_instrument",
        ok=True,
        data={"instruments": instruments, "count": len(instruments)},
    )


async def list_documents(session: AsyncSession, *, limit: int = 500) -> ToolResult:
    """Every document's id and filename, for matching against a question.

    Names, not contents: this is what lets "the Dubai prospectus" be resolved
    to a document before anything is retrieved on its behalf.
    """
    result = await session.execute(select(Document.id, Document.filename).limit(limit))
    documents = [{"id": row[0], "filename": row[1]} for row in result.all()]
    return ToolResult(
        tool_name="list_documents",
        ok=True,
        data={"documents": documents, "count": len(documents)},
    )


async def get_covenants(
    session: AsyncSession,
    *,
    instrument_id: uuid.UUID | None = None,
    instrument_ids: Sequence[uuid.UUID] | None = None,
    document_ids: Sequence[uuid.UUID] | None = None,
    covenant_type: CovenantType | None = None,
) -> ToolResult:
    """Retrieve covenants, optionally filtered.

    `document_ids` filters on the *clause's* document, which is where a
    covenant's provenance lives. Without it, a question naming one document was
    answered with every covenant in the corpus (docs/review.md, finding 15).
    """
    stmt = select(Covenant, Clause).join(Clause, Covenant.clause_id == Clause.id)
    if instrument_id is not None:
        stmt = stmt.where(Covenant.instrument_id == instrument_id)
    if instrument_ids:
        stmt = stmt.where(Covenant.instrument_id.in_(instrument_ids))
    if document_ids:
        stmt = stmt.where(Clause.document_id.in_(document_ids))
    if covenant_type is not None:
        stmt = stmt.where(Covenant.covenant_type == covenant_type)
    result = await session.execute(stmt.order_by(Clause.page_number))
    rows = [(row[0], row[1]) for row in result.all()]
    citations = [
        Citation(
            document_id=str(clause.document_id),
            chunk_id=str(clause.source_chunk_id) if clause.source_chunk_id else None,
            clause_id=str(clause.id),
            page_number=clause.page_number,
            quote=clause.source_quote[:600],
            char_start=clause.char_start,
            char_end=clause.char_end,
            section_title=clause.section_title,
        )
        for _, clause in rows
    ]
    return ToolResult(
        tool_name="get_covenants",
        ok=True,
        data={"covenants": rows, "count": len(rows)},
        citations=citations,
    )


async def get_call_schedules(
    session: AsyncSession,
    *,
    instrument_id: uuid.UUID | None = None,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> ToolResult:
    """Retrieve call/redemption schedules."""
    stmt = select(CallSchedule).order_by(CallSchedule.call_date)
    if instrument_id is not None:
        stmt = stmt.where(CallSchedule.instrument_id == instrument_id)
    if start is not None:
        stmt = stmt.where(CallSchedule.call_date >= start)
    if end is not None:
        stmt = stmt.where(CallSchedule.call_date <= end)
    result = await session.execute(stmt)
    calls = list(result.scalars().all())
    return ToolResult(
        tool_name="get_call_schedules",
        ok=True,
        data={"calls": calls, "count": len(calls)},
    )


async def get_rating_triggers(
    session: AsyncSession,
    *,
    instrument_id: uuid.UUID | None = None,
) -> ToolResult:
    """Retrieve rating triggers with their source clauses."""
    stmt = (
        select(RatingTrigger, Clause)
        .outerjoin(Clause, RatingTrigger.source_clause_id == Clause.id)
        .order_by(RatingTrigger.trigger_rank)
    )
    if instrument_id is not None:
        stmt = stmt.where(RatingTrigger.instrument_id == instrument_id)
    result = await session.execute(stmt)
    rows = [(row[0], row[1]) for row in result.all()]

    # Deduplicate — same trigger extracted by rule+LLM is one economic fact.
    distinct: dict[tuple[str, str, str], tuple[RatingTrigger, Clause | None]] = {}
    for trigger, clause in rows:
        key = (trigger.rating_agency.value, trigger.trigger_rating, trigger.trigger_direction.value)
        existing = distinct.get(key)
        if existing is None or (existing[1] is None and clause is not None):
            distinct[key] = (trigger, clause)

    citations = [
        Citation(
            document_id=str(clause.document_id) if clause else "",
            clause_id=str(clause.id) if clause else None,
            page_number=clause.page_number if clause else 1,
            quote=clause.source_quote[:600] if clause else trigger.consequence,
            char_start=clause.char_start if clause else None,
            char_end=clause.char_end if clause else None,
            section_title=clause.section_title if clause else None,
        )
        for trigger, clause in distinct.values()
    ]
    return ToolResult(
        tool_name="get_rating_triggers",
        ok=True,
        data={"triggers": list(distinct.values()), "count": len(distinct)},
        citations=citations,
    )


async def get_portfolio_holdings(
    session: AsyncSession,
    *,
    portfolio_id: uuid.UUID | None = None,
    as_of: dt.date | None = None,
    rating_below: str | None = None,
) -> ToolResult:
    """Retrieve portfolio holdings, optionally filtered by rating."""
    stmt = (
        select(PortfolioHolding, Instrument, Portfolio)
        .join(Instrument, PortfolioHolding.instrument_id == Instrument.id)
        .join(Portfolio, PortfolioHolding.portfolio_id == Portfolio.id)
    )
    if portfolio_id is not None:
        stmt = stmt.where(PortfolioHolding.portfolio_id == portfolio_id)
    if as_of is not None:
        stmt = stmt.where(PortfolioHolding.as_of_date == as_of)
    if rating_below is not None:
        try:
            threshold_rank = rank(rating_below)
            stmt = stmt.where(
                Instrument.current_rating_rank.is_not(None),
                Instrument.current_rating_rank > threshold_rank,
            )
        except UnknownRatingError:
            return ToolResult(
                tool_name="get_portfolio_holdings",
                ok=False,
                error=f"unknown rating threshold: {rating_below}",
            )

    result = await session.execute(stmt.order_by(Instrument.current_rating_rank))
    holdings_data = [
        {
            "holding_id": str(row[0].id),
            "quantity": row[0].quantity,
            "market_value": row[0].market_value,
            "nav_weight": row[0].nav_weight,
            "as_of_date": row[0].as_of_date,
            "instrument_name": row[1].instrument_name,
            "issuer_name": row[1].issuer_name,
            "instrument_type": row[1].instrument_type.value,
            "current_rating": row[1].current_rating,
            "rating_agency": row[1].rating_agency.value,
            "portfolio_name": row[2].name,
        }
        for row in result.all()
    ]
    return ToolResult(
        tool_name="get_portfolio_holdings",
        ok=True,
        data={"holdings": holdings_data, "count": len(holdings_data)},
    )


async def run_read_only_sql(
    session: AsyncSession,
    sql: str,
) -> ToolResult:
    """Execute a read-only SQL statement through the guardrail.

    The statement is parsed, validated against the table+column allowlist, and
    a forced LIMIT is appended if one is not already present. Only SELECT is
    permitted. The guardrail rejects non-SELECT, disallowed tables/columns, and
    syntactically invalid SQL before it reaches Postgres.
    """
    try:
        guard = validate_sql(sql)
        if not guard.allowed:
            return ToolResult(
                tool_name="run_read_only_sql",
                ok=False,
                error=f"SQL rejected by guardrail: {guard.reason}",
            )
    except SQLGuardError as exc:
        return ToolResult(
            tool_name="run_read_only_sql",
            ok=False,
            error=str(exc),
        )

    try:
        result = await session.execute(text(guard.rewritten))
        rows = result.all()
        columns = list(result.keys())
        rows_data = [dict(zip(columns, row, strict=True)) for row in rows]
        return ToolResult(
            tool_name="run_read_only_sql",
            ok=True,
            data={
                "columns": columns,
                "rows": rows_data,
                "row_count": len(rows_data),
                "sql_executed": guard.rewritten,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "sql_tool.execution_error",
            extra={"error": str(exc), "sql": guard.rewritten[:200]},
        )
        return ToolResult(
            tool_name="run_read_only_sql",
            ok=False,
            error=f"SQL execution failed: {exc}",
        )


async def evaluate_covenant_rule(
    session: AsyncSession,
    *,
    instrument_id: uuid.UUID,
    as_of: dt.date | None = None,
) -> ToolResult:
    """Run the deterministic rules engine over one instrument's covenants.

    CLAUDE.md 1.1: the LLM never computes a breach. This tool calls the
    deterministic rules engine and returns the structured evaluations.
    """
    instrument = await session.get(Instrument, instrument_id)
    if instrument is None:
        return ToolResult(
            tool_name="evaluate_covenant_rule",
            ok=False,
            error=f"instrument {instrument_id} not found",
        )

    as_of_date = as_of or dt.date.today()
    facts = ObservedFacts(
        as_of=as_of_date,
        current_rating=instrument.current_rating,
        rating_agency=instrument.rating_agency,
    )

    # Rating triggers
    trigger_result = await session.execute(
        select(RatingTrigger)
        .where(RatingTrigger.instrument_id == instrument_id)
        .order_by(RatingTrigger.trigger_rank)
    )
    triggers = list(trigger_result.scalars().all())

    evaluations: list[dict[str, Any]] = []
    for trigger in triggers:
        terms = CovenantTerms(
            covenant_type=CovenantType.RATING_TRIGGER,
            trigger_rating=trigger.trigger_rating,
            rating_agency=trigger.rating_agency,
            severity=trigger.severity,
        )
        ev = evaluate(terms, facts)
        evaluations.append(
            {
                "covenant_type": ev.covenant_type.value,
                "status": ev.status.value,
                "severity": ev.severity.value,
                "explanation": ev.explanation,
                "trigger_rating": trigger.trigger_rating,
                "trigger_rank": trigger.trigger_rank,
            }
        )

    # Financial covenants
    covenant_result = await session.execute(
        select(Covenant).where(Covenant.instrument_id == instrument_id)
    )
    for covenant in covenant_result.scalars().all():
        terms_or_none = _terms_from_row(covenant)
        if terms_or_none is None:
            continue
        ev = evaluate(terms_or_none, facts)
        evaluations.append(
            {
                "covenant_type": ev.covenant_type.value,
                "status": ev.status.value,
                "severity": ev.severity.value,
                "explanation": ev.explanation,
                "threshold_amount": (
                    str(covenant.threshold_amount) if covenant.threshold_amount else None
                ),
                "threshold_currency": covenant.threshold_currency,
            }
        )

    breaching = sum(1 for e in evaluations if e["status"] == "breach")
    at_risk = sum(1 for e in evaluations if e["status"] == "at_risk")
    insufficient = sum(1 for e in evaluations if e["status"] == "insufficient_data")

    return ToolResult(
        tool_name="evaluate_covenant_rule",
        ok=True,
        data={
            "instrument_name": instrument.instrument_name,
            "issuer_name": instrument.issuer_name,
            "current_rating": instrument.current_rating,
            "as_of": as_of_date.isoformat(),
            "evaluations": evaluations,
            "breach_count": breaching,
            "at_risk_count": at_risk,
            "insufficient_data_count": insufficient,
        },
    )


async def cite_sources(
    session: AsyncSession,
    *,
    clause_ids: list[uuid.UUID] | None = None,
    chunk_ids: list[uuid.UUID] | None = None,
) -> ToolResult:
    """Format citations for the answer.

    Given clause or chunk IDs, returns structured citations with page numbers,
    verbatim quotes, and offsets — everything the verify node needs to check
    that a claim is actually supported.
    """
    citations: list[Citation] = []

    if clause_ids:
        for cid in clause_ids:
            clause = await session.get(Clause, cid)
            if clause is not None:
                citations.append(
                    Citation(
                        document_id=str(clause.document_id),
                        chunk_id=str(clause.source_chunk_id) if clause.source_chunk_id else None,
                        clause_id=str(clause.id),
                        page_number=clause.page_number,
                        quote=clause.source_quote[:600],
                        char_start=clause.char_start,
                        char_end=clause.char_end,
                        section_title=clause.section_title,
                    )
                )

    if chunk_ids:
        from app.db.models.documents import DocumentChunk

        for cid in chunk_ids:
            chunk = await session.get(DocumentChunk, cid)
            if chunk is not None:
                citations.append(
                    Citation(
                        document_id=str(chunk.document_id),
                        chunk_id=str(chunk.id),
                        page_number=chunk.page_number,
                        quote=chunk.chunk_text[:600],
                        char_start=chunk.char_start,
                        char_end=chunk.char_end,
                        section_title=chunk.section_title,
                    )
                )

    return ToolResult(
        tool_name="cite_sources",
        ok=True,
        data={"citations": citations, "count": len(citations)},
        citations=citations,
    )


# -- helpers -----------------------------------------------------------------


def _terms_from_row(covenant: Covenant) -> CovenantTerms | None:
    """Rebuild evaluable terms from a stored covenant row."""
    from app.domain.rules import ComparisonOperator

    thresholds = covenant.thresholds_json or {}
    operator_raw = (covenant.conditions_json or {}).get("operator")
    operator = ComparisonOperator(str(operator_raw)) if operator_raw else None
    ratio = thresholds.get("threshold_ratio")
    trigger = thresholds.get("trigger_rating")

    try:
        return CovenantTerms(
            covenant_type=covenant.covenant_type,
            operator=operator,
            threshold_ratio=Decimal(str(ratio)) if ratio else None,
            threshold_amount=covenant.threshold_amount,
            threshold_currency=covenant.threshold_currency,
            trigger_rating=str(trigger) if trigger else None,
            rating_agency=RatingAgency.UNKNOWN,
            severity=covenant.severity,
        )
    except (ValueError, ArithmeticError):
        return None
