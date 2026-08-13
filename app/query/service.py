"""Answering questions without a model.

Every answer here is assembled from structured rows, deterministic rules and
retrieved spans. Three properties are non-negotiable and hold in every branch:

**Refusal is a correct outcome** (CLAUDE.md 1.5). When retrieval and SQL find
nothing, the answer is "No supporting evidence in the corpus" with confidence
0.0. There is no branch that fills a gap with plausible prose, because there is
no model here to do the filling -- which is precisely why building this before
Phase 5 is worth the effort.

**Breaches are computed, never inferred** (CLAUDE.md 1.1). Rating triggers are
evaluated by `app/rules/covenants.py` against ordinal ranks. A covenant whose
observed facts we do not hold is reported as INSUFFICIENT_DATA and *named as
such* in the answer, rather than quietly omitted.

**Every factual claim carries a citation** (CLAUDE.md 1.2), traced to a clause
or a chunk with a page number and a verbatim quote.
"""

from __future__ import annotations

import datetime as dt
import re
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Final

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models.clauses import CallSchedule, Clause, Covenant, RatingTrigger
from app.db.models.documents import Document, DocumentChunk
from app.db.models.instruments import Instrument
from app.db.models.portfolio import Portfolio, PortfolioHolding
from app.domain.enums import CovenantType, QueryIntent, RatingAgency
from app.domain.rules import (
    Citation,
    ComparisonOperator,
    CovenantEvaluation,
    CovenantTerms,
    ObservedFacts,
    RuleStatus,
)
from app.extract.patterns import RATING_TOKEN
from app.query.answerable import STRUCTURED_INTENTS, refusal_for, unsupported_terms
from app.query.intent import (
    classify,
    mentioned_documents,
    mentioned_entities,
    name_words,
)
from app.retrieval.hybrid import HybridSearcher
from app.rules.covenants import evaluate
from app.rules.money import format_myr
from app.rules.ratings import UnknownRatingError, normalise, rank

logger = get_logger(__name__)

NO_EVIDENCE: Final[str] = "No supporting evidence in the corpus."


def no_covenants_for(filename: str) -> str:
    """Refusal for a document the corpus holds but has never extracted.

    Naming the document and the reason matters. A bare "no supporting evidence"
    reads as "your document does not mention that", when the truth is that
    nothing has been extracted from it yet -- a different fact, with a different
    remedy, and the difference is the whole of finding 15. The alternative the
    system used to choose was worse than either: answer with some other
    document's covenants.
    """
    return (
        f"No covenants have been extracted from {filename}. The document is in the corpus and "
        "its text is searchable, but the extraction pipeline has not run against it, so there "
        "is nothing structured to answer from. Ask about its text instead, or extract it first."
    )


UNSUPPORTED_MESSAGE: Final[str] = (
    "This system answers questions evidenced by the document corpus. "
    "It does not forecast markets, value instruments or give investment advice."
)

# Retrieval alone is weaker evidence than an extracted, cited covenant, and the
# confidence reported should say so.
_RETRIEVAL_CONFIDENCE: Final[float] = 0.55
_STRUCTURED_CONFIDENCE: Final[float] = 0.95

_COVENANT_KEYWORDS: Final[dict[str, CovenantType]] = {
    "negative pledge": CovenantType.NEGATIVE_PLEDGE,
    "cross default": CovenantType.CROSS_DEFAULT,
    "cross-default": CovenantType.CROSS_DEFAULT,
    "gearing": CovenantType.GEARING_RATIO,
    "nisbah gearan": CovenantType.GEARING_RATIO,
    "interest cover": CovenantType.INTEREST_COVER,
    "finance service cover": CovenantType.FINANCE_SERVICE_COVER,
    "net worth": CovenantType.MINIMUM_NET_WORTH,
    "rating trigger": CovenantType.RATING_TRIGGER,
    "shariah": CovenantType.SHARIAH_NON_COMPLIANCE,
    "change of control": CovenantType.CHANGE_OF_CONTROL,
}

_BELOW_RATING_RE: Final[re.Pattern[str]] = re.compile(
    rf"\b(?:below|under|worse\s+than)\s+(?P<rating>{RATING_TOKEN})\b", re.IGNORECASE
)


@dataclass(frozen=True)
class Answer:
    question: str
    intent: QueryIntent
    text: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    tools_used: list[str] = field(default_factory=list)

    @property
    def chunk_ids(self) -> list[str]:
        return [item.chunk_id for item in self.citations if item.chunk_id]


class DeterministicQueryService:
    """The whole read path, with no LLM anywhere in it."""

    def __init__(
        self,
        session: AsyncSession,
        searcher: HybridSearcher | None = None,
        *,
        as_of: dt.date | None = None,
    ) -> None:
        self._session = session
        self._searcher = searcher or HybridSearcher(session)
        self._as_of = as_of or dt.date.today()

    async def answer(self, question: str) -> Answer:
        intent = classify(question)
        logger.info("query", extra={"intent": intent.value, "chars": len(question)})

        # The structured intents answer from rows, so nothing further down can
        # notice that the question was about something no row holds. Checked
        # once, here, rather than in each branch (app/query/answerable.py).
        if intent in STRUCTURED_INTENTS:
            unsupported = await self._unsupported_terms(question)
            if unsupported:
                logger.info(
                    "query.unsupported_terms",
                    extra={"intent": intent.value, "terms": list(unsupported)},
                )
                return Answer(
                    question=question,
                    intent=intent,
                    text=refusal_for(unsupported),
                    confidence=0.0,
                    refused=True,
                    tools_used=["classify_intent"],
                )

        match intent:
            case QueryIntent.UNSUPPORTED:
                return Answer(
                    question=question,
                    intent=intent,
                    text=UNSUPPORTED_MESSAGE,
                    confidence=0.0,
                    refused=True,
                    tools_used=["classify_intent"],
                )
            case QueryIntent.COVENANT_BREACH_CHECK:
                return await self._breach_check(question)
            case QueryIntent.PORTFOLIO_QUERY:
                return await self._portfolio(question)
            case QueryIntent.COVENANT_LOOKUP:
                return await self._covenant_lookup(question)
            case QueryIntent.INSTRUMENT_LOOKUP:
                return await self._instrument_lookup(question)
            case QueryIntent.DOCUMENT_SEARCH:
                return await self._document_search(question)

    # -- breach checking ---------------------------------------------------

    async def _breach_check(self, question: str) -> Answer:
        """Evaluate covenants against the facts we hold.

        Today that means rating triggers, because `instruments.current_rating`
        is a fact the system owns. Financial ratios have no observed-facts
        source yet, so they evaluate to INSUFFICIENT_DATA and the answer says
        so by name -- an unevaluated covenant silently dropped from a breach
        report is the single most dangerous output this system could produce.
        """
        instruments = await self._instruments_for(question)
        if not instruments:
            return self._no_evidence(question, QueryIntent.COVENANT_BREACH_CHECK)

        lines: list[str] = []
        citations: list[Citation] = []
        breaching = 0
        unevaluated: list[str] = []

        for instrument in instruments:
            facts = ObservedFacts(
                as_of=self._as_of,
                current_rating=instrument.current_rating,
                rating_agency=instrument.rating_agency,
            )
            for trigger, clause in await self._rating_triggers(instrument.id):
                trigger_terms = CovenantTerms(
                    covenant_type=CovenantType.RATING_TRIGGER,
                    trigger_rating=trigger.trigger_rating,
                    rating_agency=trigger.rating_agency,
                    severity=trigger.severity,
                )
                result = evaluate(trigger_terms, facts)
                lines.append(_verdict_line(instrument, result))
                if result.is_breach:
                    breaching += 1
                if clause is not None:
                    citations.append(_clause_citation(clause))

            for covenant, clause in await self._covenants_with_clause(instrument.id):
                covenant_terms = _terms_from_row(covenant)
                if covenant_terms is None:
                    continue
                result = evaluate(covenant_terms, facts)
                if result.status is RuleStatus.INSUFFICIENT_DATA:
                    unevaluated.append(
                        f"{instrument.instrument_name}: {covenant.covenant_type.value} "
                        f"({result.explanation})"
                    )
                    continue
                lines.append(_verdict_line(instrument, result))
                if result.is_breach:
                    breaching += 1
                if clause is not None:
                    citations.append(_clause_citation(clause))

        if not lines and not unevaluated:
            return self._no_evidence(question, QueryIntent.COVENANT_BREACH_CHECK)

        headline = (
            f"{breaching} covenant breach(es) found across {len(instruments)} instrument(s)."
            if breaching
            else f"No covenant breaches found across {len(instruments)} instrument(s)."
        )
        body = "\n".join(f"  - {line}" for line in lines)
        text = f"{headline}\n{body}" if body else headline
        if unevaluated:
            # Named, not hidden. "Could not evaluate" and "compliant" are
            # different answers and must never be conflated.
            text += "\n\nNot evaluated for lack of reported facts:\n" + "\n".join(
                f"  - {item}" for item in unevaluated
            )

        return Answer(
            question=question,
            intent=QueryIntent.COVENANT_BREACH_CHECK,
            text=text,
            citations=citations,
            confidence=_STRUCTURED_CONFIDENCE if lines else 0.4,
            tools_used=["get_instrument", "get_rating_triggers", "evaluate_covenant_rule"],
        )

    # -- covenant lookup ---------------------------------------------------

    async def _covenant_lookup(self, question: str) -> Answer:
        wanted = covenant_type_in(question)
        instruments = await self._instruments_for(question)
        instrument_ids = [item.id for item in instruments] or None

        # A document the question names, and whether anything has been
        # extracted from it. Without this, a question about one document was
        # answered from every document (docs/review.md, finding 15).
        document_ids, unextracted = await self._documents_for(question)
        if unextracted is not None:
            return Answer(
                question=question,
                intent=QueryIntent.COVENANT_LOOKUP,
                text=no_covenants_for(unextracted),
                confidence=0.0,
                refused=True,
                tools_used=["get_covenants"],
            )

        rows = await self._covenants_matching(wanted, instrument_ids, document_ids)
        if not rows:
            # Structured extraction found nothing; the corpus may still say it.
            return await self._document_search(question)

        lines: list[str] = []
        citations: list[Citation] = []
        confidences: list[float] = []
        for covenant, clause in rows:
            lines.append(_covenant_line(covenant, clause))
            citations.append(_clause_citation(clause))
            confidences.append(covenant.confidence or 0.5)

        calls = await self._call_schedules(instrument_ids) if _asks_about_calls(question) else []
        if calls:
            lines.append(
                "Call schedule: "
                + "; ".join(
                    f"{item.call_date.isoformat()} at {item.call_price}% ({item.call_type.value})"
                    for item in calls
                )
            )

        return Answer(
            question=question,
            intent=QueryIntent.COVENANT_LOOKUP,
            text="\n".join(f"- {line}" for line in lines),
            citations=citations,
            confidence=min(confidences) if confidences else _RETRIEVAL_CONFIDENCE,
            tools_used=["get_covenants", "cite_sources"],
        )

    # -- instrument lookup -------------------------------------------------

    async def _instrument_lookup(self, question: str) -> Answer:
        threshold = _rating_threshold_in(question)
        if threshold is not None:
            # Ordinal, not lexical: "below A" is rank > rank("A"), which is why
            # the rank column exists (CLAUDE.md 6).
            result = await self._session.execute(
                select(Instrument)
                .where(
                    Instrument.current_rating_rank.is_not(None),
                    Instrument.current_rating_rank > rank(threshold),
                )
                .order_by(Instrument.current_rating_rank)
            )
            instruments = list(result.scalars().all())
            headline = f"{len(instruments)} instrument(s) rated below {threshold}."
        else:
            instruments = await self._instruments_for(question)
            headline = f"{len(instruments)} instrument(s) matched."

        if not instruments:
            return self._no_evidence(question, QueryIntent.INSTRUMENT_LOOKUP)

        lines = [_instrument_line(item) for item in instruments]
        return Answer(
            question=question,
            intent=QueryIntent.INSTRUMENT_LOOKUP,
            text=headline + "\n" + "\n".join(f"  - {line}" for line in lines),
            confidence=_STRUCTURED_CONFIDENCE,
            tools_used=["get_instrument"],
        )

    # -- portfolio ---------------------------------------------------------

    async def _portfolio(self, question: str) -> Answer:
        threshold = _rating_threshold_in(question)
        portfolios = await self._portfolios_for(question)
        portfolio_ids = [item.id for item in portfolios] or None

        stmt = (
            select(
                Portfolio.name,
                func.count(PortfolioHolding.id),
                func.coalesce(func.sum(PortfolioHolding.market_value), 0),
            )
            .join(Portfolio, Portfolio.id == PortfolioHolding.portfolio_id)
            .join(Instrument, Instrument.id == PortfolioHolding.instrument_id)
            .group_by(Portfolio.name)
            .order_by(Portfolio.name)
        )
        if portfolio_ids:
            stmt = stmt.where(PortfolioHolding.portfolio_id.in_(portfolio_ids))
        if threshold is not None:
            stmt = stmt.where(
                Instrument.current_rating_rank.is_not(None),
                Instrument.current_rating_rank > rank(threshold),
            )

        rows = (await self._session.execute(stmt)).all()
        if not rows:
            return self._no_evidence(question, QueryIntent.PORTFOLIO_QUERY)

        qualifier = f" rated below {threshold}" if threshold else ""
        lines = [
            f"{name}: {int(count)} holding(s){qualifier}, market value {format_myr(Decimal(total))}"
            for name, count, total in rows
        ]
        total_value = sum((Decimal(row[2]) for row in rows), Decimal(0))
        return Answer(
            question=question,
            intent=QueryIntent.PORTFOLIO_QUERY,
            text="\n".join(f"- {line}" for line in lines)
            + f"\nTotal{qualifier}: {format_myr(total_value)}",
            confidence=_STRUCTURED_CONFIDENCE,
            tools_used=["get_portfolio_holdings", "run_read_only_sql"],
        )

    # -- document search ---------------------------------------------------

    async def _document_search(self, question: str) -> Answer:
        hits = await self._searcher.search(question, limit=3)
        if not hits:
            return self._no_evidence(question, QueryIntent.DOCUMENT_SEARCH)

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
        best = hits[0]
        text = (
            f"Found {len(hits)} supporting passage(s). "
            f"Best match (page {best.chunk.page_number}"
            f"{f', {best.chunk.section_title}' if best.chunk.section_title else ''}):\n"
            f"{best.chunk.chunk_text[:600]}"
        )
        return Answer(
            question=question,
            intent=QueryIntent.DOCUMENT_SEARCH,
            text=text,
            citations=citations,
            confidence=_RETRIEVAL_CONFIDENCE,
            tools_used=["search_clauses", "cite_sources"],
        )

    # -- data access -------------------------------------------------------

    async def _unsupported_terms(self, question: str) -> tuple[str, ...]:
        """Words in the question that no row and no vocabulary accounts for.

        Two `SELECT name` queries so that a question naming a real instrument,
        issuer or portfolio is understood. They run before any answer is built,
        which is the point: the alternative is discovering the question was
        unanswerable after formatting rows into a reply to it.
        """
        instruments = (
            await self._session.execute(select(Instrument.instrument_name, Instrument.issuer_name))
        ).all()
        portfolios = (await self._session.execute(select(Portfolio.name))).scalars().all()
        names = [str(row[0]) for row in instruments]
        names += [str(row[1]) for row in instruments]
        names += [str(name) for name in portfolios]
        return unsupported_terms(question, known_names=names)

    async def _instruments_for(self, question: str) -> list[Instrument]:
        result = await self._session.execute(select(Instrument))
        instruments = list(result.scalars().all())
        names = [item.instrument_name for item in instruments] + [
            item.issuer_name for item in instruments
        ]
        mentioned = mentioned_entities(question, names)
        if not mentioned:
            return instruments
        return [
            item
            for item in instruments
            if item.instrument_name in mentioned or item.issuer_name in mentioned
        ]

    async def _portfolios_for(self, question: str) -> list[Portfolio]:
        result = await self._session.execute(select(Portfolio))
        portfolios = list(result.scalars().all())
        mentioned = mentioned_entities(question, [item.name for item in portfolios])
        return [item for item in portfolios if item.name in mentioned] if mentioned else []

    async def _covenants_matching(
        self,
        covenant_type: CovenantType | None,
        instrument_ids: list[uuid.UUID] | None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[tuple[Covenant, Clause]]:
        stmt = select(Covenant, Clause).join(Clause, Covenant.clause_id == Clause.id)
        if covenant_type is not None:
            stmt = stmt.where(Covenant.covenant_type == covenant_type)
        if instrument_ids:
            stmt = stmt.where(Covenant.instrument_id.in_(instrument_ids))
        if document_ids:
            stmt = stmt.where(Clause.document_id.in_(document_ids))
        result = await self._session.execute(stmt.order_by(Clause.page_number))
        return [(row[0], row[1]) for row in result.all()]

    async def _documents_for(self, question: str) -> tuple[list[uuid.UUID] | None, str | None]:
        """The documents a question names, and the one with nothing extracted.

        The second value is set only when the question named exactly one
        document and that document holds no covenants -- the case that must
        refuse and say so, rather than widening to the rest of the corpus.
        """
        rows = (await self._session.execute(select(Document.id, Document.filename))).all()
        by_filename = {str(row[1]): row[0] for row in rows}
        # Words that name an instrument are reserved: they identify an
        # instrument, which is a different lookup with its own answer. Without
        # this, "the RM300m Green Ijarah Sukuk" resolved to whichever filename
        # happened to be the only one containing "sukuk".
        instruments = (await self._session.execute(select(Instrument))).scalars().all()
        reserved = set(
            name_words(
                " ".join(
                    [item.instrument_name for item in instruments]
                    + [item.issuer_name for item in instruments]
                )
            )
        )
        mentioned = mentioned_documents(question, list(by_filename), reserved=reserved)
        if not mentioned:
            return None, None

        document_ids = [by_filename[filename] for filename in mentioned]
        if len(document_ids) == 1 and not await self._covenants_matching(None, None, document_ids):
            return document_ids, mentioned[0]
        return document_ids, None

    async def _covenants_with_clause(
        self, instrument_id: uuid.UUID
    ) -> list[tuple[Covenant, Clause]]:
        result = await self._session.execute(
            select(Covenant, Clause)
            .join(Clause, Covenant.clause_id == Clause.id)
            .where(Covenant.instrument_id == instrument_id)
        )
        return [(row[0], row[1]) for row in result.all()]

    async def _rating_triggers(
        self, instrument_id: uuid.UUID
    ) -> list[tuple[RatingTrigger, Clause | None]]:
        """Distinct rating triggers for an instrument.

        The same trigger can be recorded twice -- once from the rule extractor
        and once from another source -- and they are one economic fact. Left
        undeduplicated, a single breach would be *counted* twice in a breach
        report, which is a wrong number rather than a cosmetic repeat. The
        cited copy wins, so the surviving row is the one that can be defended.
        """
        result = await self._session.execute(
            select(RatingTrigger, Clause)
            .outerjoin(Clause, RatingTrigger.source_clause_id == Clause.id)
            .where(RatingTrigger.instrument_id == instrument_id)
            .order_by(RatingTrigger.trigger_rank)
        )
        distinct: dict[tuple[str, str, str], tuple[RatingTrigger, Clause | None]] = {}
        for trigger, clause in result.all():
            key = (
                trigger.rating_agency.value,
                trigger.trigger_rating,
                trigger.trigger_direction.value,
            )
            existing = distinct.get(key)
            if existing is None or (existing[1] is None and clause is not None):
                distinct[key] = (trigger, clause)
        return list(distinct.values())

    async def _call_schedules(self, instrument_ids: list[uuid.UUID] | None) -> list[CallSchedule]:
        stmt = select(CallSchedule).order_by(CallSchedule.call_date)
        if instrument_ids:
            stmt = stmt.where(CallSchedule.instrument_id.in_(instrument_ids))
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    def _no_evidence(self, question: str, intent: QueryIntent) -> Answer:
        return Answer(
            question=question,
            intent=intent,
            text=NO_EVIDENCE,
            confidence=0.0,
            refused=True,
            tools_used=["classify_intent"],
        )


# -- formatting ------------------------------------------------------------


def _verdict_line(instrument: Instrument, result: CovenantEvaluation) -> str:
    return (
        f"{instrument.instrument_name} [{result.status.value.upper()}] "
        f"{result.covenant_type.value}: {result.explanation}"
    )


def _covenant_line(covenant: Covenant, clause: Clause) -> str:
    parts = [covenant.covenant_type.value]
    if covenant.threshold_amount is not None:
        parts.append(f"threshold {format_myr(covenant.threshold_amount)}")
    ratio = covenant.thresholds_json.get("threshold_ratio")
    if ratio:
        operator = covenant.conditions_json.get("operator")
        phrase = "not more than" if operator == ComparisonOperator.LTE.value else "not less than"
        parts.append(f"{phrase} {ratio} times")
    trigger = covenant.thresholds_json.get("trigger_rating")
    if trigger:
        parts.append(f"trigger rating {trigger}")
    parts.append(f"(page {clause.page_number}, {covenant.method.value}-extracted)")
    return " · ".join(str(part) for part in parts)


def _instrument_line(instrument: Instrument) -> str:
    bits = [instrument.instrument_name, instrument.issuer_name]
    if instrument.current_rating:
        bits.append(f"{instrument.current_rating} ({instrument.rating_agency.value})")
    if instrument.issue_size is not None:
        bits.append(format_myr(instrument.issue_size))
    if instrument.maturity_date is not None:
        bits.append(f"matures {instrument.maturity_date.isoformat()}")
    return " · ".join(bits)


def _clause_citation(clause: Clause) -> Citation:
    return Citation(
        document_id=str(clause.document_id),
        chunk_id=str(clause.source_chunk_id) if clause.source_chunk_id else None,
        clause_id=str(clause.id),
        page_number=clause.page_number,
        quote=clause.source_quote[:600],
        char_start=clause.char_start,
        char_end=clause.char_end,
        section_title=clause.section_title,
    )


def _terms_from_row(covenant: Covenant) -> CovenantTerms | None:
    """Rebuild evaluable terms from a stored covenant row."""
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


def covenant_type_in(question: str) -> CovenantType | None:
    """The covenant type a question names, if any.

    Public because the Phase 7 agent needs the same reading of a question that
    the Phase 4 service has. While this was private the agent retrieved *every*
    covenant for a question that named one, so "what is the cross-default
    threshold?" came back as a list of thirteen unrelated covenants that never
    mentioned the threshold -- a worse answer than the path the agent wraps.
    """
    text = question.lower()
    for keyword, covenant_type in _COVENANT_KEYWORDS.items():
        if keyword in text:
            return covenant_type
    return None


def _rating_threshold_in(question: str) -> str | None:
    match = _BELOW_RATING_RE.search(question)
    if match is None:
        return None
    try:
        return normalise(match.group("rating"))
    except UnknownRatingError:
        return None


def _asks_about_calls(question: str) -> bool:
    text = question.lower()
    return any(word in text for word in ("call", "redeem", "redemption"))


def _chunk_page(chunk: DocumentChunk) -> int:
    return chunk.page_number
