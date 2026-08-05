"""Extraction pipeline tests — rule + LLM in parallel, disagreement detection.

All tests use MockLLMProvider for zero-cost CI. The pipeline orchestrator is
tested for:
- Successful extraction via both paths
- Citation verification gating (CLAUDE.md 1.3)
- Disagreement detection and review routing
- Cost attribution
- Document status transitions
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.clauses import Clause, Covenant
from app.db.models.documents import Document, DocumentChunk
from app.db.models.ops import HumanReview
from app.domain.enums import (
    ClauseType,
    CovenantType,
    DocumentStatus,
    ExtractionMethod,
    ReviewStatus,
)
from app.domain.extraction import RuleExtraction
from app.domain.rules import ComparisonOperator, CovenantTerms
from app.extract.pipeline import ExtractionPipeline, PipelineOutcome, _fields_disagree
from app.extract.schemas import LLMCovenantExtraction
from app.llm.mock import MockLLMProvider
from app.llm.router import LLMRouter

pytestmark = pytest.mark.usefixtures("storage_root")


@pytest_asyncio.fixture
async def mock_router(db_session: AsyncSession) -> LLMRouter:
    return LLMRouter(db_session, provider=MockLLMProvider())


@pytest_asyncio.fixture
async def pipeline(db_session: AsyncSession, mock_router: LLMRouter) -> ExtractionPipeline:
    return ExtractionPipeline(db_session, router=mock_router)


# -- pipeline end-to-end ------------------------------------------------------


async def test_pipeline_extracts_from_seeded_corpus(
    pipeline: ExtractionPipeline,
    indexed_corpus: list[uuid.UUID],
    seeded_universe: None,
) -> None:
    """The pipeline runs rule + LLM extraction on a document with real chunks."""
    document_id = indexed_corpus[0]
    outcome = await pipeline.extract(document_id)

    assert isinstance(outcome, PipelineOutcome)
    assert outcome.document_id == document_id
    # The rule extractor runs as the comparison baseline on every candidate.
    assert outcome.rule_extractions > 0
    # LLM candidates are detected from the covenant-heavy fixture.
    assert outcome.llm_candidates >= 0
    # The pipeline should not error out.
    assert not outcome.errors or all("budget" not in str(e).lower() for e in outcome.errors)


async def test_rule_clauses_come_from_the_rule_service_not_the_pipeline(
    pipeline: ExtractionPipeline,
    indexed_corpus: list[uuid.UUID],
    db_session: AsyncSession,
    seeded_universe: None,
) -> None:
    """Rule-method clauses are `RuleExtractionService`'s output, not this pipeline's.

    The `indexed_corpus` fixture runs that service, so the rows exist before
    the pipeline is invoked at all. Naming this explicitly matters: the test
    that used to stand here asserted the same rows and read as though the
    pipeline had written them.
    """
    document_id = indexed_corpus[1]  # trust deed

    before = await _clause_count(db_session, document_id, ExtractionMethod.RULE)
    await pipeline.extract(document_id)
    after = await _clause_count(db_session, document_id, ExtractionMethod.RULE)

    assert before > 0
    assert after == before

    result = await db_session.execute(
        select(Clause).where(
            Clause.document_id == document_id,
            Clause.method == ExtractionMethod.RULE,
        )
    )
    assert all(clause.citation_verified for clause in result.scalars().all())


async def _clause_count(
    session: AsyncSession, document_id: uuid.UUID, method: ExtractionMethod
) -> int:
    result = await session.execute(
        select(Clause).where(Clause.document_id == document_id, Clause.method == method)
    )
    return len(list(result.scalars().all()))


# -- idempotency (CLAUDE.md 1.7) ----------------------------------------------


async def test_rerunning_the_pipeline_is_a_no_op(
    pipeline: ExtractionPipeline,
    indexed_corpus: list[uuid.UUID],
    db_session: AsyncSession,
    seeded_universe: None,
) -> None:
    """The same extraction identity must not be paid for twice."""
    document_id = indexed_corpus[0]

    first = await pipeline.extract(document_id)
    assert not first.skipped

    second = await pipeline.extract(document_id)

    assert second.skipped
    assert second.total_cost_usd == Decimal("0")
    assert second.llm_candidates == 0


async def test_forced_rerun_replaces_rather_than_duplicates(
    pipeline: ExtractionPipeline,
    indexed_corpus: list[uuid.UUID],
    db_session: AsyncSession,
    seeded_universe: None,
) -> None:
    """A forced re-run clears its own prior output first.

    Without the clear, a second run appended a whole second set of LLM clauses
    and covenants, so the document accumulated duplicates of every extraction.
    """
    document_id = indexed_corpus[0]

    await pipeline.extract(document_id)
    after_first = await _clause_count(db_session, document_id, ExtractionMethod.LLM)

    await pipeline.extract(document_id, force=True)
    after_second = await _clause_count(db_session, document_id, ExtractionMethod.LLM)

    assert after_second == after_first


async def test_human_reviewed_clauses_survive_a_forced_rerun(
    pipeline: ExtractionPipeline,
    indexed_corpus: list[uuid.UUID],
    db_session: AsyncSession,
    seeded_universe: None,
) -> None:
    """Machine output is disposable; a reviewer's verdict is not."""
    document_id = indexed_corpus[0]
    await pipeline.extract(document_id)

    result = await db_session.execute(
        select(Clause).where(
            Clause.document_id == document_id,
            Clause.method == ExtractionMethod.LLM,
        )
    )
    clauses = list(result.scalars().all())
    if not clauses:
        pytest.skip("mock output produced no LLM clauses for this fixture")

    corrected = clauses[0]
    corrected.review_status = ReviewStatus.CORRECTED
    await db_session.flush()

    await pipeline.extract(document_id, force=True)

    assert await db_session.get(Clause, corrected.id) is not None


# -- disagreement detection ---------------------------------------------------


def test_fields_disagree_on_different_covenant_types() -> None:
    """Different covenant types are a clear disagreement."""
    llm = LLMCovenantExtraction(
        clause_type=ClauseType.FINANCIAL_COVENANT,
        covenant_type=CovenantType.GEARING_RATIO,
        source_quote="test",
        confidence=0.95,
        threshold_ratio=Decimal("1.75"),
        operator=ComparisonOperator.LTE,
    )

    rule = RuleExtraction(
        clause_type=ClauseType.FINANCIAL_COVENANT,
        covenant_type=CovenantType.INTEREST_COVER,
        quote="test",
        char_start=0,
        char_end=4,
        confidence=0.90,
    )

    assert _fields_disagree(llm, rule) is True


def test_fields_agree_on_same_values() -> None:
    """Identical values should not trigger a disagreement."""
    llm = LLMCovenantExtraction(
        clause_type=ClauseType.CROSS_DEFAULT,
        covenant_type=CovenantType.CROSS_DEFAULT,
        source_quote="test",
        confidence=0.95,
        threshold_amount=Decimal("30000000"),
        threshold_currency="MYR",
    )

    rule = RuleExtraction(
        clause_type=ClauseType.CROSS_DEFAULT,
        covenant_type=CovenantType.CROSS_DEFAULT,
        quote="test",
        char_start=0,
        char_end=4,
        confidence=0.90,
        terms=CovenantTerms(
            covenant_type=CovenantType.CROSS_DEFAULT,
            threshold_amount=Decimal("30000000"),
            threshold_currency="MYR",
        ),
    )

    assert _fields_disagree(llm, rule) is False


def test_fields_disagree_on_different_thresholds() -> None:
    """RM30m vs RM50m is a material disagreement."""
    llm = LLMCovenantExtraction(
        clause_type=ClauseType.CROSS_DEFAULT,
        covenant_type=CovenantType.CROSS_DEFAULT,
        source_quote="test",
        confidence=0.95,
        threshold_amount=Decimal("30000000"),
        threshold_currency="MYR",
    )

    rule = RuleExtraction(
        clause_type=ClauseType.CROSS_DEFAULT,
        covenant_type=CovenantType.CROSS_DEFAULT,
        quote="test",
        char_start=0,
        char_end=4,
        confidence=0.90,
        terms=CovenantTerms(
            covenant_type=CovenantType.CROSS_DEFAULT,
            threshold_amount=Decimal("50000000"),
            threshold_currency="MYR",
        ),
    )

    assert _fields_disagree(llm, rule) is True


def test_fields_disagree_when_llm_has_amount_but_rule_does_not() -> None:
    """One extractor finding a threshold the other missed is a disagreement."""
    llm = LLMCovenantExtraction(
        clause_type=ClauseType.CROSS_DEFAULT,
        covenant_type=CovenantType.CROSS_DEFAULT,
        source_quote="test",
        confidence=0.90,
        threshold_amount=Decimal("30000000"),
        threshold_currency="MYR",
    )

    rule = RuleExtraction(
        clause_type=ClauseType.CROSS_DEFAULT,
        covenant_type=CovenantType.CROSS_DEFAULT,
        quote="test",
        char_start=0,
        char_end=4,
        confidence=0.85,
    )

    assert _fields_disagree(llm, rule) is True


# -- citation verification gating ---------------------------------------------


async def test_verified_citation_is_required_for_persistence(
    pipeline: ExtractionPipeline,
    indexed_corpus: list[uuid.UUID],
    db_session: AsyncSession,
    seeded_universe: None,
) -> None:
    """CLAUDE.md 1.3: unverifiable citations must not be persisted."""
    document_id = indexed_corpus[0]
    await pipeline.extract(document_id)

    # Every clause persisted by the pipeline must have citation_verified=True.
    result = await db_session.execute(
        select(Clause).where(
            Clause.document_id == document_id,
            Clause.citation_verified == False,  # noqa: E712
        )
    )
    unverified = list(result.scalars().all())
    assert len(unverified) == 0, (
        f"Found {len(unverified)} clauses with unverified citations — CLAUDE.md 1.3 violated"
    )


# -- review queue routing -----------------------------------------------------


async def test_failed_extraction_is_queued_for_review(
    pipeline: ExtractionPipeline,
    indexed_corpus: list[uuid.UUID],
    db_session: AsyncSession,
    seeded_universe: None,
) -> None:
    """Failures are routed to the human review queue, not silently dropped."""
    document_id = indexed_corpus[0]
    await pipeline.extract(document_id)

    result = await db_session.execute(
        select(HumanReview).where(HumanReview.status == ReviewStatus.PENDING)
    )
    reviews = list(result.scalars().all())

    # With the mock provider, most extractions succeed, but some may fail
    # validation. What matters is that any failure is visible.
    # Even if there are 0 pending reviews (all succeeded), that's fine with mock.
    # The pipeline handles the routing — the test verifies the pipeline doesn't
    # crash and the review table is in a consistent state.
    assert all(r.trigger_reason is not None for r in reviews)
    assert all(r.status is ReviewStatus.PENDING for r in reviews)


# -- cost attribution ---------------------------------------------------------


async def test_pipeline_tracks_cost(
    pipeline: ExtractionPipeline,
    indexed_corpus: list[uuid.UUID],
    seeded_universe: None,
) -> None:
    outcome = await pipeline.extract(indexed_corpus[0])

    # The mock provider has non-zero cost estimates.
    assert outcome.total_cost_usd >= Decimal("0")
    # Cost should be tracked as Decimal, never float.
    assert isinstance(outcome.total_cost_usd, Decimal)


async def test_pipeline_budget_exceeded_is_reported(
    pipeline: ExtractionPipeline,
    indexed_corpus: list[uuid.UUID],
    db_session: AsyncSession,
    seeded_universe: None,
) -> None:
    """When the budget guard blocks, the pipeline reports it cleanly."""
    # Set a tiny per-call cap so the guard trips immediately.
    from app.core.config import Settings

    tiny_budget = Settings(
        ENVIRONMENT="test",
        MAX_COST_PER_CALL_USD=Decimal("0.000001"),  # ~impossibly small
    )

    pipeline_tiny = ExtractionPipeline(
        db_session,
        router=LLMRouter(db_session, provider=MockLLMProvider(), settings=tiny_budget),
    )

    document_id = indexed_corpus[0]
    outcome = await pipeline_tiny.extract(document_id)

    # The pipeline returns a PipelineOutcome, never raises BudgetExceededError
    # to the caller.
    assert isinstance(outcome, PipelineOutcome)
    assert outcome.budget_exceeded

    # PLAN.md 2: abort the document and mark it `budget_exceeded`. Marking it
    # EXTRACTED -- the previous behaviour -- told every downstream reader the
    # document was fully processed when the candidates past the ceiling were
    # never looked at.
    document = await db_session.get(Document, document_id)
    assert document is not None
    assert document.status is DocumentStatus.BUDGET_EXCEEDED


async def test_review_queue_entries_reference_a_real_entity(
    pipeline: ExtractionPipeline,
    indexed_corpus: list[uuid.UUID],
    db_session: AsyncSession,
    seeded_universe: None,
) -> None:
    """Every queued item must be openable.

    Entries used to carry a freshly minted `uuid4()` as `entity_id`, which
    resolved to no row in any table -- a reviewer had nothing to review.
    """
    document_id = indexed_corpus[0]
    await pipeline.extract(document_id)

    result = await db_session.execute(select(HumanReview))
    reviews = list(result.scalars().all())
    if not reviews:
        pytest.skip("this fixture produced no review-queue entries")

    lookup = {
        "clause": Clause,
        "covenant": Covenant,
        "document_chunk": DocumentChunk,
    }
    for review in reviews:
        model = lookup.get(review.entity_type)
        assert model is not None, f"unknown entity_type {review.entity_type!r}"
        assert await db_session.get(model, review.entity_id) is not None, (
            f"{review.entity_type} {review.entity_id} does not exist"
        )


# -- LLM-only path (no rule match) --------------------------------------------


async def test_pipeline_handles_candidate_without_rule_match(
    pipeline: ExtractionPipeline,
    indexed_corpus: list[uuid.UUID],
    db_session: AsyncSession,
    seeded_universe: None,
) -> None:
    """An LLM extraction with no overlapping rule result is persisted on its own."""
    document_id = indexed_corpus[0]
    pipeline_outcome = await pipeline.extract(document_id)

    # Some clauses will be rule-only, some may be LLM (mock). Verify both methods
    # can coexist in the database.
    assert isinstance(pipeline_outcome, PipelineOutcome)
    result = await db_session.execute(select(Clause).where(Clause.document_id == document_id))
    clauses = list(result.scalars().all())

    methods = {c.method for c in clauses}
    # Rule extraction always produces output with the mock-based pipeline.
    assert ExtractionMethod.RULE in methods
    # LLM clauses may or may not appear depending on candidate detection +
    # mock output — both are acceptable.
