"""Extraction pipeline — rule + LLM in parallel, disagreement → review.

PLAN.md §3: "We run the rule-based extractor and the Opus extractor on every
candidate span." This module orchestrates that:

1. Rule extraction runs on all chunks and is **persisted** (free, always) --
   before any billable call, so a budget-exhausted document still ends up with
   the deterministic extractor's clauses rather than with nothing.
2. Candidate detection narrows the document to LLM-worthy spans.
3. LLM extraction runs on each candidate (billable, budget-guarded).
4. Results are compared field by field — disagreement triggers human review.
5. Citation verification is the final gate before persistence (CLAUDE.md 1.3).

The pipeline is the transaction owner: nothing is committed until every
extraction that can succeed has succeeded, and every failure is routed to
the review queue.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass, field
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models.clauses import Clause, Covenant
from app.db.models.documents import Document, DocumentChunk
from app.db.models.ops import ExtractionJob, HumanReview
from app.db.repositories.clauses import ClauseRepository, CovenantRepository
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
from app.extract.candidates import Candidate, CandidateDetectionService
from app.extract.citations import verify_quote
from app.extract.linking import resolve_instrument
from app.extract.llm_extractor import LLMExtraction, LLMExtractor
from app.extract.prompts import PROMPT_VERSION
from app.extract.rule_extractor import extract as rule_extract
from app.extract.schemas import LLMCovenantExtraction
from app.extract.service import RuleExtractionService
from app.llm.budget import BudgetExceededError
from app.llm.router import LLMRouter

# One rule extraction plus the chunk it came from.
_RuleResult = tuple[RuleExtraction, DocumentChunk]
# Rule extractions grouped by the chunk whose coordinate system they use.
_RuleIndex = dict[uuid.UUID, list[_RuleResult]]

logger = get_logger(__name__)

# Part of the extraction identity (CLAUDE.md 1.7). Bump it when a change to
# this pipeline should invalidate prior runs even though the prompt and model
# are unchanged.
LLM_EXTRACTOR_VERSION = "llm-pipeline-v1"

# CLAUDE.md §5: monetary thresholds above this go to human review.
HIGH_VALUE_THRESHOLD = Decimal("100000000")

# `human_reviews.entity_type` is a free-text discriminator; these are the values
# this pipeline writes, named so a typo is a NameError rather than an orphan row.
_CLAUSE_ENTITY = "clause"
_COVENANT_ENTITY = "covenant"
_CHUNK_ENTITY = "document_chunk"

# Review outcomes that represent human work and must survive a re-extraction.
_HUMAN_TOUCHED = (ReviewStatus.APPROVED, ReviewStatus.CORRECTED, ReviewStatus.REJECTED)


@dataclass
class PipelineOutcome:
    """Summary of one pipeline run against one document.

    Rule and LLM counts are kept apart on purpose. They used to share
    `rule_clauses`, so every persisted LLM clause incremented the rule counter
    and the reported numbers described neither extractor -- which also meant
    the "did the LLM actually help?" measurement PLAN.md 3 exists to enable was
    reading a number that mixed both.

    `rule_clauses` and `rule_covenants` are rows `RuleExtractionService` wrote,
    which this pipeline now runs before spending anything. `rule_extractions`
    is a different number: what the regex pass *found* while building the
    comparison baseline, which is not the same as what survived citation
    verification and got persisted. Both are reported because a gap between
    them is worth noticing.
    """

    document_id: uuid.UUID
    rule_clauses: int = 0
    rule_covenants: int = 0
    rule_extractions: int = 0
    rule_skipped: bool = False
    llm_candidates: int = 0
    llm_extracted: int = 0
    llm_failed: int = 0
    llm_clauses: int = 0
    llm_covenants: int = 0
    disagreements: int = 0
    queued_for_review: int = 0
    total_cost_usd: Decimal = Decimal("0")
    budget_exceeded: bool = False
    skipped: bool = False
    errors: list[str] = field(default_factory=list)


class ExtractionPipeline:
    """Orchestrate rule + LLM extraction and disagreement detection.

    Usage:
        pipeline = ExtractionPipeline(session, router=llm_router)
        outcome = await pipeline.extract(document_id)
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        router: LLMRouter | None = None,
    ) -> None:
        self._session = session
        self._router = router or LLMRouter(session)
        self._settings = get_settings()
        self._documents = DocumentRepository(session)
        self._chunks = DocumentChunkRepository(session)
        self._clauses = ClauseRepository(session)
        self._covenants = CovenantRepository(session)
        self._reviews = HumanReviewRepository(session)
        self._jobs = ExtractionJobRepository(session)
        self._rules = RuleExtractionService(session, self._settings)
        self._candidates_svc = CandidateDetectionService(session)
        self._llm_extractor = LLMExtractor(session, router=self._router)

    async def extract(
        self,
        document_id: uuid.UUID,
        *,
        instrument_id: uuid.UUID | None = None,
        llm_enabled: bool = True,
        force: bool = False,
    ) -> PipelineOutcome:
        """Run the full extraction pipeline on one document.

        Idempotent by extraction identity (CLAUDE.md 1.7): a completed run over
        the same `(document_sha256, prompt_version, model_id, extractor_version)`
        is a no-op costing $0. `force=True` re-runs anyway, discarding the
        previous machine output first -- which is what makes the re-run produce
        one set of clauses rather than a second one alongside the first.
        """
        document = await self._documents.get(document_id)
        if document is None:
            raise LookupError(f"document {document_id} not found")

        outcome = PipelineOutcome(document_id=document_id)

        job = await self._ensure_job(document)
        if job.status is JobStatus.SUCCEEDED and not force:
            logger.info(
                "llm extraction skipped; extraction identity already satisfied",
                extra={
                    "document_id": str(document_id),
                    "model_id": self._settings.EXTRACTION_MODEL,
                    "prompt_version": PROMPT_VERSION,
                },
            )
            outcome.skipped = True
            return outcome

        # Machine output from a previous attempt is disposable; a human's
        # verdict on it is not. Clearing here is what keeps a re-run
        # idempotent instead of duplicating every clause and covenant.
        await self._clear_previous_llm_output(document_id)

        job.status = JobStatus.RUNNING
        job.started_at = _now()
        await self._session.flush()

        # A covenant with no instrument is invisible to every portfolio query,
        # which is the whole point of extracting it. The rule extractor has
        # always resolved this; the LLM path did not, so its covenants were
        # persisted unlinked and silently absent from portfolio answers.
        resolved = instrument_id or await resolve_instrument(self._session, document_id)
        if resolved is None:
            logger.warning(
                "document is not linked to an instrument; "
                "its covenants will not appear in portfolio queries",
                extra={"document_id": str(document_id)},
            )

        try:
            return await self._run(document, job, outcome, resolved, llm_enabled)
        except Exception as exc:
            # Without this the job stays RUNNING for ever and the session is
            # left dirty, so the next run finds a job that is neither complete
            # nor retryable. `RuleExtractionService` has always done this; the
            # LLM pipeline did not.
            await self._session.rollback()
            await self._mark_failed(document_id, exc)
            raise

    async def _run(
        self,
        document: Document,
        job: ExtractionJob,
        outcome: PipelineOutcome,
        instrument_id: uuid.UUID | None,
        llm_enabled: bool,
    ) -> PipelineOutcome:
        """The pipeline body, with the extraction identity already resolved."""
        document_id = document.id

        # --- Rule extraction (always, free, and persisted) ------------------
        # Persisted *before* any billable call, which is what makes PLAN.md 3's
        # fallback real: when the budget guard trips mid-document, the rule
        # extractor's clauses and covenants are already rows rather than
        # something that was computed and discarded. This pipeline used to run
        # the rules purely as a comparison baseline, so a budget-exhausted
        # document ended up with nothing at all.
        try:
            persisted = await self._rules.extract_document(document_id, instrument_id=instrument_id)
            outcome.rule_clauses = persisted.clauses
            outcome.rule_covenants = persisted.covenants
            outcome.rule_skipped = persisted.skipped
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "rule extraction failed",
                extra={"document_id": str(document_id), "error": str(exc)},
            )
            outcome.errors.append(f"rule: {exc}")

        # The in-memory pass is the comparison baseline, and is deliberately
        # separate from the rows above: `_fields_disagree` compares span-level
        # `RuleExtraction` objects, and re-deriving them from persisted rows
        # would lose the offsets the overlap check needs. It is regex over text
        # already in memory, so running it twice costs nothing.
        try:
            rule_results = await self._run_rule_extraction(document)
            outcome.rule_extractions = sum(len(items) for items in rule_results.values())
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "rule comparison baseline failed",
                extra={"document_id": str(document_id), "error": str(exc)},
            )
            outcome.errors.append(f"rule-baseline: {exc}")
            rule_results = {}

        # --- LLM extraction (billable, budget-guarded) ----------------------
        if not llm_enabled:
            await self._finalize(document, job, outcome)
            return outcome

        try:
            candidates = await self._candidates_svc.detect(document_id)
            outcome.llm_candidates = len(candidates)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "candidate detection failed",
                extra={"document_id": str(document_id), "error": str(exc)},
            )
            outcome.errors.append(f"candidates: {exc}")
            await self._finalize(document, job, outcome)
            return outcome

        llm_results: list[LLMExtraction] = []
        for candidate in candidates:
            try:
                llm_result = await self._llm_extractor.extract(candidate, document_id=document_id)
                llm_results.append(llm_result)
                outcome.total_cost_usd += llm_result.cost_usd
            except BudgetExceededError as exc:
                logger.warning(
                    "budget exceeded during LLM extraction; stopping",
                    extra={
                        "document_id": str(document_id),
                        "reason": exc.decision.reason if hasattr(exc, "decision") else str(exc),
                    },
                )
                outcome.budget_exceeded = True
                break
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "LLM extraction failed for candidate",
                    extra={
                        "document_id": str(document_id),
                        "candidate_page": candidate.page_number,
                        "error": str(exc),
                    },
                )
                outcome.errors.append(f"llm: {exc}")

        # --- Compare, persist, and route to review --------------------------
        for llm_result in llm_results:
            if llm_result.output is not None:
                await self._handle_llm_success(
                    document, llm_result, rule_results, instrument_id, outcome
                )
                outcome.llm_extracted += 1
            else:
                await self._handle_llm_failure(document, llm_result, outcome)
                outcome.llm_failed += 1

        await self._finalize(document, job, outcome)
        return outcome

    # -- rule extraction -----------------------------------------------------

    async def _run_rule_extraction(self, document: Document) -> _RuleIndex:
        """Run the rule extractor over all chunks; index results by chunk.

        Keyed by `chunk_id`, not by page. Rule offsets and candidate offsets
        are both *chunk-local*, so comparing spans across two chunks that
        happen to share a page compares unrelated coordinate systems -- which
        produced both phantom disagreements and missed real ones. A page
        routinely holds several chunks, so this was not a corner case.
        """
        chunks = await self._chunks.list_for_document(document.id, limit=100_000)
        results: _RuleIndex = {}
        for chunk in chunks:
            for extraction in rule_extract(chunk.chunk_text):
                results.setdefault(chunk.id, []).append((extraction, chunk))
        return results

    # -- LLM success path ----------------------------------------------------

    async def _handle_llm_success(
        self,
        document: Document,
        llm_result: LLMExtraction,
        rule_results: _RuleIndex,
        instrument_id: uuid.UUID | None,
        outcome: PipelineOutcome,
    ) -> None:
        output = llm_result.output
        assert output is not None
        candidate = llm_result.candidate

        # Full citation verification against the chunk text.
        chunk = await self._chunks.get(candidate.chunk_id)
        if chunk is None:
            await self._queue_review(
                entity_type=_CHUNK_ENTITY,
                entity_id=candidate.chunk_id,
                field_name="source_quote",
                new_value=output.source_quote,
                source_quote=output.source_quote,
                page_number=candidate.page_number,
                confidence=output.confidence,
                trigger=ReviewTrigger.CITATION_UNVERIFIED,
            )
            outcome.queued_for_review += 1
            return

        citation = verify_quote(output.source_quote, chunk.chunk_text)
        if not citation.verified or citation.char_start is None or citation.char_end is None:
            # CLAUDE.md 1.3: unverifiable citation → review, never persist.
            # A verified check with no span is treated the same way: CLAUDE.md
            # 1.2 requires the span, and a clause that cannot name one is not
            # traceable evidence.
            await self._queue_review(
                entity_type=_CHUNK_ENTITY,
                entity_id=candidate.chunk_id,
                field_name="source_quote",
                new_value=output.source_quote,
                source_quote=output.source_quote,
                page_number=candidate.page_number,
                confidence=output.confidence,
                trigger=ReviewTrigger.CITATION_UNVERIFIED,
            )
            outcome.queued_for_review += 1
            return

        # Find the rule extraction covering the same span, if any. Computed
        # once and passed down -- recomputing it in the review-trigger check
        # was both wasted work and a place for the two answers to drift.
        rule_match = self._find_rule_match(candidate, rule_results)
        disagrees = rule_match is not None and _fields_disagree(output, rule_match[0])
        if disagrees:
            outcome.disagreements += 1

        # CLAUDE.md 5 routes on the clause's own confidence, not only the
        # covenant's. A clause below the threshold used to be persisted as
        # NOT_REQUIRED because the confidence check lived solely in the
        # covenant branch -- so a low-confidence clause carrying no covenant
        # was never reviewed by anyone.
        clause_needs_review = (
            output.confidence < self._settings.DEFAULT_CONFIDENCE_THRESHOLD or disagrees
        )

        clause = Clause(
            document_id=document.id,
            instrument_id=instrument_id,
            source_chunk_id=candidate.chunk_id,
            clause_type=output.clause_type,
            clause_text=output.source_quote,
            page_number=candidate.page_number,
            section_title=candidate.section_title,
            source_quote=output.source_quote,
            char_start=citation.char_start,
            char_end=citation.char_end,
            citation_verified=True,
            citation_match_score=citation.score,
            normalized_json=self._build_thresholds(output),
            method=ExtractionMethod.LLM,
            confidence=output.confidence,
            extraction_status=ExtractionStatus.EXTRACTED,
            review_status=(
                ReviewStatus.PENDING if clause_needs_review else ReviewStatus.NOT_REQUIRED
            ),
        )
        clause = await self._clauses.add_verified(clause)
        outcome.llm_clauses += 1

        if output.covenant_type is not None:
            await self._persist_llm_covenant(clause, output, instrument_id, disagrees, outcome)
        elif clause_needs_review:
            # No covenant to hang the review on, but the clause still needs a
            # human. Queue against the clause itself rather than dropping it.
            await self._queue_review(
                entity_type=_CLAUSE_ENTITY,
                entity_id=clause.id,
                field_name="clause_type",
                new_value=output.clause_type.value,
                source_quote=output.source_quote,
                page_number=clause.page_number,
                confidence=output.confidence,
                trigger=(
                    ReviewTrigger.RULE_LLM_DISAGREEMENT
                    if disagrees
                    else ReviewTrigger.LOW_CONFIDENCE
                ),
            )
            outcome.queued_for_review += 1

    async def _persist_llm_covenant(
        self,
        clause: Clause,
        output: LLMCovenantExtraction,
        instrument_id: uuid.UUID | None,
        disagrees: bool,
        outcome: PipelineOutcome,
    ) -> None:
        """Persist a covenant from LLM output, with review routing."""
        review_trigger = self._covenant_review_trigger(output, disagrees=disagrees)

        covenant = Covenant(
            clause_id=clause.id,
            instrument_id=instrument_id,
            covenant_type=output.covenant_type,
            summary=output.summary,
            conditions_json=self._build_conditions(output),
            thresholds_json=self._build_thresholds(output),
            threshold_amount=output.threshold_amount,
            threshold_currency=output.threshold_currency,
            severity=output.severity,
            method=ExtractionMethod.LLM,
            confidence=output.confidence,
            review_status=(
                ReviewStatus.PENDING if review_trigger is not None else ReviewStatus.NOT_REQUIRED
            ),
        )
        await self._covenants.add(covenant)
        outcome.llm_covenants += 1

        if review_trigger is not None:
            await self._queue_review(
                entity_type=_COVENANT_ENTITY,
                entity_id=covenant.id,
                field_name=_review_field_name(output),
                new_value=_review_value(output),
                source_quote=output.source_quote,
                page_number=clause.page_number,
                confidence=output.confidence,
                trigger=review_trigger,
            )
            outcome.queued_for_review += 1

    # -- LLM failure path ----------------------------------------------------

    async def _handle_llm_failure(
        self,
        document: Document,
        llm_result: LLMExtraction,
        outcome: PipelineOutcome,
    ) -> None:
        """Route a failed LLM extraction to the review queue.

        The queue entry points at the chunk the candidate came from. It used to
        carry a freshly minted `uuid4()`, which resolved to no row anywhere --
        a reviewer opening the item had nothing to open.
        """
        trigger = (
            ReviewTrigger.VALIDATION_RETRY
            if llm_result.retry_attempted
            else ReviewTrigger.LOW_CONFIDENCE
        )
        await self._queue_review(
            entity_type=_CHUNK_ENTITY,
            entity_id=llm_result.candidate.chunk_id,
            field_name="llm_extraction",
            new_value="; ".join(llm_result.validation_errors),
            source_quote=llm_result.candidate.text[:2000],
            page_number=llm_result.candidate.page_number,
            confidence=0.0,
            trigger=trigger,
        )
        outcome.queued_for_review += 1

    # -- helpers -------------------------------------------------------------

    def _find_rule_match(
        self,
        candidate: Candidate,
        rule_results: _RuleIndex,
    ) -> _RuleResult | None:
        """The rule extraction overlapping the candidate, within its own chunk.

        Where several rule extractions overlap the span -- a sentence carrying
        two financial covenants does exactly that -- the one overlapping most
        wins, so the comparison is against the closest rule reading rather than
        whichever happened to be found first.
        """
        best: _RuleResult | None = None
        best_overlap = 0
        for rule_extraction, chunk in rule_results.get(candidate.chunk_id, ()):
            overlap = min(rule_extraction.char_end, candidate.char_end) - max(
                rule_extraction.char_start, candidate.char_start
            )
            if overlap > best_overlap:
                best_overlap = overlap
                best = (rule_extraction, chunk)
        return best

    def _covenant_review_trigger(
        self,
        output: LLMCovenantExtraction,
        *,
        disagrees: bool,
    ) -> ReviewTrigger | None:
        """Determine if this covenant needs human review (CLAUDE.md 5)."""
        # High-value threshold always triggers review.
        if output.threshold_amount is not None and output.threshold_amount > HIGH_VALUE_THRESHOLD:
            return ReviewTrigger.HIGH_VALUE_THRESHOLD

        # Low confidence triggers review.
        if output.confidence < self._settings.DEFAULT_CONFIDENCE_THRESHOLD:
            return ReviewTrigger.LOW_CONFIDENCE

        # Rule/LLM disagreement triggers review.
        if disagrees:
            return ReviewTrigger.RULE_LLM_DISAGREEMENT

        return None

    def _build_conditions(self, output: LLMCovenantExtraction) -> dict[str, object]:
        conditions: dict[str, object] = {}
        if output.operator is not None:
            conditions["operator"] = output.operator.value
        return conditions

    def _build_thresholds(self, output: LLMCovenantExtraction) -> dict[str, object]:
        thresholds: dict[str, object] = {}
        if output.threshold_amount is not None:
            thresholds["threshold_amount"] = str(output.threshold_amount)
            if output.threshold_currency:
                thresholds["threshold_currency"] = output.threshold_currency
        if output.threshold_ratio is not None:
            thresholds["threshold_ratio"] = str(output.threshold_ratio)
        if output.trigger_rating:
            thresholds["trigger_rating"] = output.trigger_rating
            if output.rating_agency:
                thresholds["rating_agency"] = output.rating_agency.value
        return thresholds

    async def _queue_review(
        self,
        *,
        entity_type: str,
        entity_id: uuid.UUID,
        field_name: str,
        new_value: str | None,
        source_quote: str,
        page_number: int,
        confidence: float,
        trigger: ReviewTrigger,
    ) -> None:
        """Create a human review queue entry."""
        await self._reviews.add(
            HumanReview(
                entity_type=entity_type,
                entity_id=entity_id,
                field_name=field_name,
                new_value=new_value,
                source_quote=source_quote[:2000],
                page_number=page_number,
                confidence=confidence,
                trigger_reason=trigger,
                status=ReviewStatus.PENDING,
            )
        )

    async def _finalize(
        self, document: Document, job: ExtractionJob, outcome: PipelineOutcome
    ) -> None:
        """Record the terminal state of this run and commit.

        A budget-exhausted run is neither a success nor a failure: PLAN.md 2
        calls for the document to be marked `budget_exceeded`. It used to be
        marked EXTRACTED, which told every downstream reader that the document
        had been fully processed when the candidates after the ceiling were
        never looked at.
        """
        job.estimated_cost_usd = outcome.total_cost_usd

        if outcome.budget_exceeded:
            document.status = DocumentStatus.BUDGET_EXCEEDED
            job.status = JobStatus.BUDGET_EXCEEDED
            job.error_message = "per-document or global budget ceiling reached mid-extraction"
        elif outcome.errors and outcome.llm_extracted == 0:
            job.status = JobStatus.FAILED
            job.error_message = "; ".join(outcome.errors)[:2000]
        else:
            if outcome.llm_clauses > 0:
                # Only rows this pipeline actually wrote move the document on.
                # Rule extractions found here are a comparison baseline, not
                # persisted output, so counting them would mark a document
                # EXTRACTED on the strength of clauses nobody stored.
                document.status = DocumentStatus.EXTRACTED
            job.status = JobStatus.SUCCEEDED

        job.finished_at = _now()
        await self._session.commit()

        logger.info(
            "extraction pipeline complete",
            extra={
                "document_id": str(outcome.document_id),
                "rule_clauses": outcome.rule_clauses,
                "rule_covenants": outcome.rule_covenants,
                "rule_extractions": outcome.rule_extractions,
                "llm_candidates": outcome.llm_candidates,
                "llm_extracted": outcome.llm_extracted,
                "llm_failed": outcome.llm_failed,
                "llm_clauses": outcome.llm_clauses,
                "llm_covenants": outcome.llm_covenants,
                "disagreements": outcome.disagreements,
                "queued_for_review": outcome.queued_for_review,
                "total_cost_usd": str(outcome.total_cost_usd),
                "budget_exceeded": outcome.budget_exceeded,
                "job_status": job.status.value,
            },
        )

    # -- extraction identity -------------------------------------------------

    async def _ensure_job(self, document: Document) -> ExtractionJob:
        """The job row keyed on this run's extraction identity (CLAUDE.md 1.7).

        The model id and prompt version are part of the key, so switching model
        or editing a prompt correctly re-runs, while re-running an unchanged
        pipeline is free.
        """
        job = await self._jobs.find_by_identity(
            document_sha256=document.sha256,
            job_type=JobType.EXTRACT_COVENANT,
            prompt_version=PROMPT_VERSION,
            model_id=self._settings.EXTRACTION_MODEL,
            extractor_version=LLM_EXTRACTOR_VERSION,
        )
        if job is not None:
            return job
        return await self._jobs.add(
            ExtractionJob(
                document_id=document.id,
                document_sha256=document.sha256,
                job_type=JobType.EXTRACT_COVENANT,
                status=JobStatus.QUEUED,
                model_id=self._settings.EXTRACTION_MODEL,
                prompt_version=PROMPT_VERSION,
                extractor_version=LLM_EXTRACTOR_VERSION,
            )
        )

    async def _mark_failed(self, document_id: uuid.UUID, exc: BaseException) -> None:
        """Record the failure on the job so the run is retryable, then commit."""
        message = f"{type(exc).__name__}: {exc}"
        logger.error(
            "extraction pipeline failed",
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

    async def _clear_previous_llm_output(self, document_id: uuid.UUID) -> None:
        """Drop LLM-authored clauses from an earlier run of this document.

        Machine output is disposable; human judgement is not, so anything a
        reviewer has already approved, corrected or rejected survives. Covenants
        cascade with their clause. This mirrors `RuleExtractionService`, which
        has always done it -- the LLM path did not, so a second run doubled
        every clause and covenant rather than replacing them.
        """
        result = await self._session.execute(
            select(Clause).where(
                Clause.document_id == document_id,
                Clause.method == ExtractionMethod.LLM,
                Clause.review_status.not_in(_HUMAN_TOUCHED),
            )
        )
        for clause in result.scalars().all():
            await self._session.delete(clause)
        await self._session.flush()


def _fields_disagree(
    llm_output: LLMCovenantExtraction,
    rule_extraction: RuleExtraction,
) -> bool:
    """Compare key fields between LLM and rule extractions.

    Returns True if there is a material disagreement that needs human review.
    """
    # Compare covenant type.
    if llm_output.covenant_type != rule_extraction.covenant_type:
        return True

    # If the LLM quantified a threshold the rule missed (or vice versa),
    # that is a material disagreement worth reviewing.
    rule_terms = rule_extraction.terms
    llm_has_threshold = (
        llm_output.threshold_amount is not None or llm_output.threshold_ratio is not None
    )
    if llm_has_threshold and rule_terms is None:
        return True
    if (
        not llm_has_threshold
        and rule_terms is not None
        and (rule_terms.threshold_amount is not None or rule_terms.threshold_ratio is not None)
    ):
        return True

    if rule_terms is not None:
        # Compare threshold amounts (within 1% tolerance for rounding).
        rule_amount = rule_terms.threshold_amount
        if llm_output.threshold_amount is not None and rule_amount is not None:
            try:
                diff = abs(llm_output.threshold_amount - rule_amount)
                if diff > Decimal("0") and diff / max(
                    llm_output.threshold_amount, rule_amount
                ) > Decimal("0.01"):
                    return True
            except (TypeError, ValueError, ArithmeticError):
                return True
        elif (llm_output.threshold_amount is None) != (rule_amount is None):
            return True

        # Compare threshold ratios.
        rule_ratio = rule_terms.threshold_ratio
        if llm_output.threshold_ratio is not None and rule_ratio is not None:
            try:
                diff = abs(llm_output.threshold_ratio - rule_ratio)
                if diff > Decimal("0") and diff / max(
                    llm_output.threshold_ratio, rule_ratio
                ) > Decimal("0.01"):
                    return True
            except (TypeError, ValueError, ArithmeticError):
                return True
        elif (llm_output.threshold_ratio is None) != (rule_ratio is None):
            return True

        # Compare operators.
        if llm_output.operator != rule_terms.operator:
            return True

    return False


def _review_field_name(output: LLMCovenantExtraction) -> str:
    if output.threshold_amount is not None:
        return "threshold_amount"
    if output.threshold_ratio is not None:
        return "threshold_ratio"
    if output.trigger_rating:
        return "trigger_rating"
    return "covenant_type"


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def _review_value(output: LLMCovenantExtraction) -> str | None:
    for value in (
        output.threshold_amount,
        output.threshold_ratio,
        output.trigger_rating,
    ):
        if value is not None:
            return str(value)
    return output.covenant_type.value if output.covenant_type else None
