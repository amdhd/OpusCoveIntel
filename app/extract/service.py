"""Persisting rule extractions as cited clauses and covenants.

This is what makes the system answer questions at $0 (PLAN.md 3). It is also
where the review-queue policy from CLAUDE.md 5 first becomes real code:

* confidence below the threshold  -> review, trigger LOW_CONFIDENCE
* a monetary threshold over RM100m -> review, trigger HIGH_VALUE_THRESHOLD
* a citation that will not verify  -> **not persisted at all** (CLAUDE.md 1.3)

Every clause is written through `ClauseRepository.add_verified`, which refuses
an unverified citation. A regex extractor cannot produce one -- its quotes are
literal slices -- but routing it through the same gate means the gate is
exercised now rather than first tested in Phase 6 against output that can lie.

Re-running is idempotent: rule-authored clauses are cleared and rebuilt, except
any a human has already approved or corrected. Machine output is disposable;
human judgement is not.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.models.clauses import CallSchedule, Clause, Covenant, RatingTrigger
from app.db.models.documents import Document, DocumentChunk
from app.db.models.instruments import Instrument
from app.db.models.ops import ExtractionJob, HumanReview
from app.db.repositories.clauses import (
    CallScheduleRepository,
    ClauseRepository,
    CovenantRepository,
    RatingTriggerRepository,
)
from app.db.repositories.documents import DocumentChunkRepository, DocumentRepository
from app.db.repositories.ops import ExtractionJobRepository, HumanReviewRepository
from app.domain.enums import (
    DocumentStatus,
    ExtractionMethod,
    ExtractionStatus,
    JobStatus,
    JobType,
    ReviewStatus,
    ReviewTrigger,
)
from app.domain.extraction import RuleExtraction
from app.extract.citations import verify_quote
from app.extract.rule_extractor import EXTRACTOR_VERSION, extract, extract_call_schedule
from app.rules.ratings import UnknownRatingError, rank

logger = get_logger(__name__)

EXTRACT_PROMPT_VERSION = "v0"
EXTRACT_MODEL_ID = "none"

# CLAUDE.md 5: any monetary threshold above this goes to a human regardless of
# how confident the extractor was. A misread cross-default threshold at this
# size is a portfolio-level error.
HIGH_VALUE_THRESHOLD = Decimal("100000000")

# Review outcomes that represent human work and must survive a re-extraction.
_HUMAN_TOUCHED = (ReviewStatus.APPROVED, ReviewStatus.CORRECTED, ReviewStatus.REJECTED)


@dataclass(frozen=True)
class ExtractionOutcome:
    document_id: uuid.UUID
    instrument_id: uuid.UUID | None
    clauses: int
    covenants: int
    call_schedules: int
    rating_triggers: int
    queued_for_review: int
    skipped: bool


class RuleExtractionService:
    def __init__(self, session: AsyncSession, settings: Settings | None = None) -> None:
        self._session = session
        self._settings = settings or get_settings()
        self._documents = DocumentRepository(session)
        self._chunks = DocumentChunkRepository(session)
        self._clauses = ClauseRepository(session)
        self._covenants = CovenantRepository(session)
        self._calls = CallScheduleRepository(session)
        self._triggers = RatingTriggerRepository(session)
        self._reviews = HumanReviewRepository(session)
        self._jobs = ExtractionJobRepository(session)

    async def extract_document(
        self, document_id: uuid.UUID, *, instrument_id: uuid.UUID | None = None
    ) -> ExtractionOutcome:
        document = await self._documents.get(document_id)
        if document is None:
            raise LookupError(f"document {document_id} not found")

        job = await self._ensure_job(document)
        if job.status is JobStatus.SUCCEEDED:
            logger.info(
                "rule extraction skipped; identity already satisfied",
                extra={"document_id": str(document_id), "extractor_version": EXTRACTOR_VERSION},
            )
            return await self._existing_outcome(document_id, instrument_id, skipped=True)

        job.status = JobStatus.RUNNING
        job.started_at = _now()
        await self._session.flush()

        try:
            resolved = instrument_id or await self._resolve_instrument(document_id)
            await self._clear_previous(document_id)

            counts = _Counts()
            chunks = await self._chunks.list_for_document(document_id, limit=100_000)
            for chunk in chunks:
                await self._extract_chunk(document, chunk, resolved, counts)

            document.status = DocumentStatus.EXTRACTED
            job.status = JobStatus.SUCCEEDED
            job.finished_at = _now()
            await self._session.commit()
        except Exception as exc:
            await self._session.rollback()
            await self._fail(document_id, exc)
            raise

        logger.info(
            "rule extraction complete",
            extra={
                "document_id": str(document_id),
                "instrument_id": str(resolved) if resolved else None,
                "clauses": counts.clauses,
                "covenants": counts.covenants,
                "queued_for_review": counts.reviews,
            },
        )
        return ExtractionOutcome(
            document_id=document_id,
            instrument_id=resolved,
            clauses=counts.clauses,
            covenants=counts.covenants,
            call_schedules=counts.calls,
            rating_triggers=counts.triggers,
            queued_for_review=counts.reviews,
            skipped=False,
        )

    # -- per chunk ---------------------------------------------------------

    async def _extract_chunk(
        self,
        document: Document,
        chunk: DocumentChunk,
        instrument_id: uuid.UUID | None,
        counts: _Counts,
    ) -> None:
        for extraction in extract(chunk.chunk_text):
            clause = await self._persist_clause(document, chunk, extraction, instrument_id)
            if clause is None:
                continue
            counts.clauses += 1

            if extraction.covenant_type is not None and extraction.terms is not None:
                await self._persist_covenant(clause, extraction, instrument_id, counts)

            if (
                instrument_id is not None
                and extraction.terms is not None
                and extraction.terms.trigger_rating
            ):
                await self._persist_rating_trigger(clause, extraction, instrument_id, counts)

        if instrument_id is not None:
            await self._persist_call_schedule(chunk, instrument_id, counts)

    async def _persist_clause(
        self,
        document: Document,
        chunk: DocumentChunk,
        extraction: RuleExtraction,
        instrument_id: uuid.UUID | None,
    ) -> Clause | None:
        check = verify_quote(extraction.quote, chunk.chunk_text)
        if not check.verified:
            # CLAUDE.md 1.3 / 9: never persist an unverified quote. For a regex
            # extractor this should be unreachable, which is exactly why it is
            # worth logging loudly rather than passing silently.
            logger.warning(
                "citation verification failed; clause dropped",
                extra={
                    "chunk_id": str(chunk.id),
                    "pattern_id": extraction.pattern_id,
                    "clause_type": extraction.clause_type.value,
                },
            )
            return None

        clause = Clause(
            document_id=document.id,
            instrument_id=instrument_id,
            source_chunk_id=chunk.id,
            clause_type=extraction.clause_type,
            clause_text=extraction.quote,
            page_number=chunk.page_number,
            section_title=chunk.section_title,
            source_quote=extraction.quote,
            # Offsets are into the chunk, which is what verification checked.
            char_start=check.char_start,
            char_end=check.char_end,
            citation_verified=True,
            citation_match_score=check.score,
            normalized_json=dict(extraction.normalized),
            method=ExtractionMethod.RULE,
            confidence=extraction.confidence,
            extraction_status=ExtractionStatus.EXTRACTED,
            review_status=self._review_status(extraction.confidence),
        )
        return await self._clauses.add_verified(clause)

    async def _persist_covenant(
        self,
        clause: Clause,
        extraction: RuleExtraction,
        instrument_id: uuid.UUID | None,
        counts: _Counts,
    ) -> None:
        terms = extraction.terms
        assert terms is not None
        covenant = Covenant(
            clause_id=clause.id,
            instrument_id=instrument_id,
            covenant_type=terms.covenant_type,
            summary=terms.description,
            conditions_json={
                "operator": terms.operator.value if terms.operator else None,
                "pattern_id": extraction.pattern_id,
            },
            thresholds_json=dict(extraction.normalized),
            threshold_amount=terms.threshold_amount,
            threshold_currency=terms.threshold_currency,
            severity=terms.severity,
            method=ExtractionMethod.RULE,
            confidence=extraction.confidence,
            review_status=self._review_status(extraction.confidence, terms.threshold_amount),
        )
        await self._covenants.add(covenant)
        counts.covenants += 1

        trigger = self._review_trigger(extraction.confidence, terms.threshold_amount)
        if trigger is not None:
            await self._queue_review(clause, covenant, extraction, trigger)
            counts.reviews += 1

    async def _persist_rating_trigger(
        self,
        clause: Clause,
        extraction: RuleExtraction,
        instrument_id: uuid.UUID,
        counts: _Counts,
    ) -> None:
        terms = extraction.terms
        assert terms is not None and terms.trigger_rating
        try:
            trigger_rank = rank(terms.trigger_rating, terms.rating_agency)
        except UnknownRatingError:
            return
        await self._triggers.add(
            RatingTrigger(
                instrument_id=instrument_id,
                source_clause_id=clause.id,
                rating_agency=terms.rating_agency,
                trigger_rating=terms.trigger_rating,
                # Stored so "which holdings trip on a downgrade below A" is an
                # integer comparison in SQL rather than string logic.
                trigger_rank=trigger_rank,
                consequence=extraction.quote[:1000],
                severity=terms.severity,
                method=ExtractionMethod.RULE,
                confidence=extraction.confidence,
                review_status=self._review_status(extraction.confidence),
            )
        )
        counts.triggers += 1

    async def _persist_call_schedule(
        self, chunk: DocumentChunk, instrument_id: uuid.UUID, counts: _Counts
    ) -> None:
        for call_date, price, call_type, _start, _end in extract_call_schedule(chunk.chunk_text):
            existing = await self._session.execute(
                select(CallSchedule).where(
                    CallSchedule.instrument_id == instrument_id,
                    CallSchedule.call_date == call_date,
                )
            )
            if existing.scalar_one_or_none() is not None:
                continue
            await self._calls.add(
                CallSchedule(
                    instrument_id=instrument_id,
                    call_date=call_date,
                    call_price=price,
                    call_type=call_type,
                    method=ExtractionMethod.RULE,
                    confidence=0.85,
                    review_status=ReviewStatus.PENDING,
                )
            )
            counts.calls += 1

    # -- review routing ----------------------------------------------------

    def _review_status(
        self, confidence: float, threshold_amount: Decimal | None = None
    ) -> ReviewStatus:
        return (
            ReviewStatus.PENDING
            if self._review_trigger(confidence, threshold_amount) is not None
            else ReviewStatus.NOT_REQUIRED
        )

    def _review_trigger(
        self, confidence: float, threshold_amount: Decimal | None
    ) -> ReviewTrigger | None:
        if threshold_amount is not None and threshold_amount > HIGH_VALUE_THRESHOLD:
            return ReviewTrigger.HIGH_VALUE_THRESHOLD
        if confidence < self._settings.DEFAULT_CONFIDENCE_THRESHOLD:
            return ReviewTrigger.LOW_CONFIDENCE
        return None

    async def _queue_review(
        self,
        clause: Clause,
        covenant: Covenant,
        extraction: RuleExtraction,
        trigger: ReviewTrigger,
    ) -> None:
        await self._reviews.add(
            HumanReview(
                entity_type="covenant",
                entity_id=covenant.id,
                field_name=_review_field(extraction),
                new_value=_review_value(extraction),
                source_quote=extraction.quote[:2000],
                page_number=clause.page_number,
                confidence=extraction.confidence,
                trigger_reason=trigger,
                status=ReviewStatus.PENDING,
            )
        )

    # -- bookkeeping -------------------------------------------------------

    async def _resolve_instrument(self, document_id: uuid.UUID) -> uuid.UUID | None:
        """Link a document to an instrument by issuer name.

        Deliberately literal: an issuer's registered name must appear verbatim
        in the document text. Fuzzy matching here would attach covenants to the
        wrong issuer, and a covenant on the wrong instrument is worse than a
        covenant on none -- it produces a confident, wrong portfolio answer.
        """
        chunks = await self._chunks.list_for_document(document_id, limit=20)
        # Whitespace is collapsed on both sides before matching. A PDF wraps
        # "Synthetic Retail REIT\nBerhad" across a line, and a plain substring
        # test then silently fails to link the document to its instrument --
        # which looks like "no covenants found" rather than like a bug.
        haystack = _collapse(" ".join(chunk.chunk_text for chunk in chunks))
        if not haystack:
            return None

        result = await self._session.execute(select(Instrument))
        matches = [
            instrument
            for instrument in result.scalars().all()
            if _collapse(instrument.issuer_name) in haystack
        ]
        if len(matches) == 1:
            return matches[0].id
        if len(matches) > 1:
            logger.warning(
                "document names several issuers; leaving it unlinked",
                extra={
                    "document_id": str(document_id),
                    "issuers": [item.issuer_name for item in matches],
                },
            )
        return None

    async def _clear_previous(self, document_id: uuid.UUID) -> None:
        result = await self._session.execute(
            select(Clause).where(
                Clause.document_id == document_id,
                Clause.method == ExtractionMethod.RULE,
                Clause.review_status.not_in(_HUMAN_TOUCHED),
            )
        )
        for clause in result.scalars().all():
            # Covenants cascade with their clause; a covenant without its
            # evidence is exactly the row this schema refuses to have.
            await self._session.delete(clause)
        await self._session.flush()

    async def _existing_outcome(
        self, document_id: uuid.UUID, instrument_id: uuid.UUID | None, *, skipped: bool
    ) -> ExtractionOutcome:
        clauses = await self._clauses.list_for_document(document_id)
        return ExtractionOutcome(
            document_id=document_id,
            instrument_id=instrument_id,
            clauses=len(clauses),
            covenants=await self._covenants.count_for_document(document_id),
            call_schedules=0,
            rating_triggers=0,
            queued_for_review=0,
            skipped=skipped,
        )

    async def _ensure_job(self, document: Document) -> ExtractionJob:
        job = await self._jobs.find_by_identity(
            document_sha256=document.sha256,
            job_type=JobType.EXTRACT_COVENANT,
            prompt_version=EXTRACT_PROMPT_VERSION,
            model_id=EXTRACT_MODEL_ID,
            extractor_version=EXTRACTOR_VERSION,
        )
        if job is not None:
            return job
        return await self._jobs.add(
            ExtractionJob(
                document_id=document.id,
                document_sha256=document.sha256,
                job_type=JobType.EXTRACT_COVENANT,
                status=JobStatus.QUEUED,
                model_id=EXTRACT_MODEL_ID,
                prompt_version=EXTRACT_PROMPT_VERSION,
                extractor_version=EXTRACTOR_VERSION,
            )
        )

    async def _fail(self, document_id: uuid.UUID, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        logger.error(
            "rule extraction failed",
            extra={"document_id": str(document_id), "error": message},
        )
        document = await self._documents.get(document_id)
        if document is None:
            return
        job = await self._ensure_job(document)
        if job.status is not JobStatus.SUCCEEDED:
            job.status = JobStatus.FAILED
            job.error_message = message[:2000]
            job.finished_at = _now()
        await self._session.commit()


@dataclass
class _Counts:
    clauses: int = 0
    covenants: int = 0
    calls: int = 0
    triggers: int = 0
    reviews: int = 0


def _review_field(extraction: RuleExtraction) -> str:
    if extraction.terms is None:
        return "covenant"
    if extraction.terms.threshold_amount is not None:
        return "threshold_amount"
    if extraction.terms.threshold_ratio is not None:
        return "threshold_ratio"
    if extraction.terms.trigger_rating:
        return "trigger_rating"
    return "covenant"


def _review_value(extraction: RuleExtraction) -> str | None:
    terms = extraction.terms
    if terms is None:
        return None
    for value in (terms.threshold_amount, terms.threshold_ratio, terms.trigger_rating):
        if value is not None:
            return str(value)
    return None


def _collapse(text: str) -> str:
    """Lower-case with runs of whitespace collapsed, for line-wrap-safe matching."""
    return " ".join(text.split()).lower()


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)
