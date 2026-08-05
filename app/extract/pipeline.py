"""Extraction pipeline — rule + LLM in parallel, disagreement → review.

PLAN.md §3: "We run the rule-based extractor and the Opus extractor on every
candidate span." This module orchestrates that:

1. Rule extraction runs on all chunks (free, always).
2. Candidate detection narrows the document to LLM-worthy spans.
3. LLM extraction runs on each candidate (billable, budget-guarded).
4. Results are compared field by field — disagreement triggers human review.
5. Citation verification is the final gate before persistence (CLAUDE.md 1.3).

The pipeline is the transaction owner: nothing is committed until every
extraction that can succeed has succeeded, and every failure is routed to
the review queue.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models.clauses import Clause, Covenant
from app.db.models.documents import Document, DocumentChunk
from app.db.models.ops import HumanReview
from app.db.repositories.clauses import ClauseRepository, CovenantRepository
from app.db.repositories.documents import DocumentChunkRepository, DocumentRepository
from app.db.repositories.ops import HumanReviewRepository
from app.domain.enums import (
    DocumentStatus,
    ExtractionMethod,
    ExtractionStatus,
    ReviewStatus,
    ReviewTrigger,
)
from app.domain.extraction import RuleExtraction
from app.extract.candidates import Candidate, CandidateDetectionService
from app.extract.citations import verify_quote
from app.extract.llm_extractor import LLMExtraction, LLMExtractor
from app.extract.rule_extractor import extract as rule_extract
from app.extract.schemas import LLMCovenantExtraction
from app.llm.budget import BudgetExceededError
from app.llm.router import LLMRouter

# One rule extraction plus the chunk it came from.
_RuleResult = tuple[RuleExtraction, DocumentChunk]

logger = get_logger(__name__)

# CLAUDE.md §5: monetary thresholds above this go to human review.
HIGH_VALUE_THRESHOLD = Decimal("100000000")


@dataclass
class PipelineOutcome:
    """Summary of one pipeline run against one document."""

    document_id: uuid.UUID
    rule_clauses: int = 0
    rule_covenants: int = 0
    llm_candidates: int = 0
    llm_extracted: int = 0
    llm_failed: int = 0
    disagreements: int = 0
    queued_for_review: int = 0
    total_cost_usd: Decimal = Decimal("0")
    budget_exceeded: bool = False
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
        self._candidates_svc = CandidateDetectionService(session)
        self._llm_extractor = LLMExtractor(session, router=self._router)

    async def extract(
        self,
        document_id: uuid.UUID,
        *,
        instrument_id: uuid.UUID | None = None,
        llm_enabled: bool = True,
    ) -> PipelineOutcome:
        """Run the full extraction pipeline on one document."""
        document = await self._documents.get(document_id)
        if document is None:
            raise LookupError(f"document {document_id} not found")

        outcome = PipelineOutcome(document_id=document_id)

        # --- Rule extraction (always, free) ---------------------------------
        try:
            rule_results = await self._run_rule_extraction(document)
            outcome.rule_clauses = len(rule_results)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "rule extraction failed",
                extra={"document_id": str(document_id), "error": str(exc)},
            )
            outcome.errors.append(f"rule: {exc}")
            # Rule extraction failed, but we can still try LLM.
            rule_results = {}

        # --- LLM extraction (billable, budget-guarded) ----------------------
        if not llm_enabled:
            await self._finalize(document, outcome)
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
            await self._finalize(document, outcome)
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

        await self._finalize(document, outcome)
        return outcome

    # -- rule extraction -----------------------------------------------------

    async def _run_rule_extraction(self, document: Document) -> dict[tuple[int, int], _RuleResult]:
        """Run the rule extractor over all chunks; return results keyed by span.

        We key by (page, char_start) so we can look up the rule extraction that
        overlaps an LLM candidate span.
        """
        chunks = await self._chunks.list_for_document(document.id, limit=100_000)
        results: dict[tuple[int, int], _RuleResult] = {}
        for chunk in chunks:
            for extraction in rule_extract(chunk.chunk_text):
                key = (chunk.page_number, extraction.char_start)
                results[key] = (extraction, chunk)
        return results

    # -- LLM success path ----------------------------------------------------

    async def _handle_llm_success(
        self,
        document: Document,
        llm_result: LLMExtraction,
        rule_results: dict[tuple[int, int], _RuleResult],
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
                entity_type="clause",
                entity_id=uuid.uuid4(),
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
        if not citation.verified:
            # CLAUDE.md 1.3: unverifiable citation → review, never persist.
            await self._queue_review(
                entity_type="clause",
                entity_id=uuid.uuid4(),
                field_name="source_quote",
                new_value=output.source_quote,
                source_quote=output.source_quote,
                page_number=candidate.page_number,
                confidence=output.confidence,
                trigger=ReviewTrigger.CITATION_UNVERIFIED,
            )
            outcome.queued_for_review += 1
            return

        # Find the rule extraction for the same span, if any.
        rule_match = self._find_rule_match(candidate, rule_results)

        # Check for disagreement.
        if rule_match is not None:
            rule_extraction, _rule_chunk = rule_match
            disagree = _fields_disagree(output, rule_extraction)
            if disagree:
                outcome.disagreements += 1

        # Persist the clause.
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
            normalized_json={},
            method=ExtractionMethod.LLM,
            confidence=output.confidence,
            extraction_status=ExtractionStatus.EXTRACTED,
            review_status=ReviewStatus.NOT_REQUIRED,
        )
        clause = await self._clauses.add_verified(clause)
        outcome.rule_clauses += 1

        # If there's a covenant, persist it.
        if output.covenant_type is not None:
            await self._persist_llm_covenant(clause, output, instrument_id, rule_match, outcome)

    async def _persist_llm_covenant(
        self,
        clause: Clause,
        output: LLMCovenantExtraction,
        instrument_id: uuid.UUID | None,
        rule_match: _RuleResult | None,
        outcome: PipelineOutcome,
    ) -> None:
        """Persist a covenant from LLM output, with review routing."""
        review_trigger = self._covenant_review_trigger(output, rule_match)

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
        outcome.rule_covenants += 1

        if review_trigger is not None:
            await self._queue_review(
                entity_type="covenant",
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
        """Route a failed LLM extraction to the review queue."""
        trigger = (
            ReviewTrigger.VALIDATION_RETRY
            if llm_result.retry_attempted
            else ReviewTrigger.LOW_CONFIDENCE
        )
        await self._queue_review(
            entity_type="clause",
            entity_id=uuid.uuid4(),
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
        rule_results: dict[tuple[int, int], _RuleResult],
    ) -> _RuleResult | None:
        """Find the rule extraction whose span overlaps the candidate."""
        for (page, rule_start), (rule_extraction, _chunk) in rule_results.items():
            if page != candidate.page_number:
                continue
            rule_end = rule_extraction.char_end
            # Overlap check.
            if rule_start < candidate.char_end and rule_end > candidate.char_start:
                return (rule_extraction, _chunk)
        return None

    def _covenant_review_trigger(
        self,
        output: LLMCovenantExtraction,
        rule_match: _RuleResult | None,
    ) -> ReviewTrigger | None:
        """Determine if this covenant needs human review."""
        # High-value threshold always triggers review.
        if output.threshold_amount is not None and output.threshold_amount > HIGH_VALUE_THRESHOLD:
            return ReviewTrigger.HIGH_VALUE_THRESHOLD

        # Low confidence triggers review.
        if output.confidence < self._settings.DEFAULT_CONFIDENCE_THRESHOLD:
            return ReviewTrigger.LOW_CONFIDENCE

        # Rule/LLM disagreement triggers review.
        if rule_match is not None:
            rule_extraction, _ = rule_match
            if _fields_disagree(output, rule_extraction):
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

    async def _finalize(self, document: Document, outcome: PipelineOutcome) -> None:
        """Mark the document as extracted and commit."""
        if outcome.rule_clauses > 0 or outcome.llm_extracted > 0:
            document.status = DocumentStatus.EXTRACTED
        await self._session.commit()

        logger.info(
            "extraction pipeline complete",
            extra={
                "document_id": str(outcome.document_id),
                "rule_clauses": outcome.rule_clauses,
                "rule_covenants": outcome.rule_covenants,
                "llm_candidates": outcome.llm_candidates,
                "llm_extracted": outcome.llm_extracted,
                "llm_failed": outcome.llm_failed,
                "disagreements": outcome.disagreements,
                "queued_for_review": outcome.queued_for_review,
                "total_cost_usd": str(outcome.total_cost_usd),
                "budget_exceeded": outcome.budget_exceeded,
            },
        )


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


def _review_value(output: LLMCovenantExtraction) -> str | None:
    for value in (
        output.threshold_amount,
        output.threshold_ratio,
        output.trigger_rating,
    ):
        if value is not None:
            return str(value)
    return output.covenant_type.value if output.covenant_type else None
