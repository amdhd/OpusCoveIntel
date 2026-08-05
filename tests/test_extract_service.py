"""Persisting rule extractions.

What matters here is not that regexes match -- `test_rule_extractor.py` covers
that -- but that what reaches the database is *defensible*: every clause cited
and verified, low-confidence and high-value covenants queued for a human, and
human judgement surviving a re-extraction.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.clauses import Clause, Covenant
from app.db.models.documents import DocumentChunk
from app.db.models.ops import HumanReview
from app.db.repositories.clauses import ClauseRepository
from app.domain.enums import (
    CovenantType,
    DocumentStatus,
    ExtractionMethod,
    ReviewStatus,
    ReviewTrigger,
)
from app.extract.service import RuleExtractionService

pytestmark = pytest.mark.usefixtures("storage_root")


async def clauses_for(session: AsyncSession, document_id: uuid.UUID) -> list[Clause]:
    result = await session.execute(select(Clause).where(Clause.document_id == document_id))
    return list(result.scalars().all())


async def covenants_of_type(session: AsyncSession, covenant_type: CovenantType) -> list[Covenant]:
    result = await session.execute(select(Covenant).where(Covenant.covenant_type == covenant_type))
    return list(result.scalars().all())


# -- what gets persisted ---------------------------------------------------


async def test_extraction_produces_cited_clauses(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    clauses = await clauses_for(db_session, indexed_corpus[0])

    assert clauses
    assert all(clause.citation_verified for clause in clauses)
    assert all(clause.citation_match_score is not None for clause in clauses)
    assert all(clause.method is ExtractionMethod.RULE for clause in clauses)


async def test_every_clause_quote_is_found_in_its_chunk(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    for clause in await clauses_for(db_session, indexed_corpus[0]):
        chunk = await db_session.get(DocumentChunk, clause.source_chunk_id)
        assert chunk is not None
        # CLAUDE.md 1.2: the whole chain is only as good as this.
        assert clause.source_quote in chunk.chunk_text
        assert chunk.chunk_text[clause.char_start : clause.char_end] == clause.source_quote


async def test_thresholds_are_stored_as_decimals_not_text(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    cross_defaults = await covenants_of_type(db_session, CovenantType.CROSS_DEFAULT)

    amounts = {covenant.threshold_amount for covenant in cross_defaults}
    assert Decimal("30000000") in amounts
    # Two issuers, two different thresholds -- an extractor that attaches one
    # issuer's number to another is wrong in a way one document would hide.
    assert Decimal("50000000") in amounts
    assert all(isinstance(amount, Decimal) for amount in amounts)


async def test_a_rating_trigger_is_stored_with_its_ordinal_rank(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    from app.db.models.clauses import RatingTrigger

    result = await db_session.execute(select(RatingTrigger))
    triggers = list(result.scalars().all())

    assert triggers
    # The rank is what makes "downgraded below A" an integer comparison.
    assert all(trigger.trigger_rank >= 0 for trigger in triggers)
    assert any(trigger.trigger_rating == "BBB+" for trigger in triggers)


async def test_a_call_schedule_is_extracted_from_the_table(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    from app.db.models.clauses import CallSchedule

    result = await db_session.execute(
        select(CallSchedule).where(CallSchedule.method == ExtractionMethod.RULE)
    )
    schedule = sorted(result.scalars().all(), key=lambda item: item.call_date)

    # Rows the *extractor* produced, distinguished by method from the ones the
    # seed created for the same instrument.
    assert [item.call_date.isoformat() for item in schedule] == [
        "2028-06-15",
        "2029-06-15",
        "2030-06-15",
    ]
    assert schedule[0].call_price == Decimal("102.000000")
    assert all(item.review_status is ReviewStatus.PENDING for item in schedule)


async def test_the_document_is_marked_extracted(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    from app.db.models.documents import Document

    document = await db_session.get(Document, indexed_corpus[0])

    assert document is not None
    assert document.status is DocumentStatus.EXTRACTED


async def test_documents_are_linked_to_the_issuer_they_name(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
) -> None:
    service = RuleExtractionService(db_session)
    # Re-run against the seeded universe so instrument linking has candidates.
    for document_id in indexed_corpus:
        await _force_reextract(db_session, service, document_id)

    clauses = await clauses_for(db_session, indexed_corpus[1])
    assert clauses
    # The trust deed names Synthetic Infrastructure Holdings Berhad, whose name
    # the PDF wraps across a line -- whitespace-insensitive matching is what
    # makes this link at all.
    assert all(clause.instrument_id is not None for clause in clauses)


# -- review routing --------------------------------------------------------


async def test_a_low_confidence_covenant_is_queued_for_review(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    result = await db_session.execute(
        select(HumanReview).where(HumanReview.trigger_reason == ReviewTrigger.LOW_CONFIDENCE)
    )
    reviews = list(result.scalars().all())

    assert reviews
    assert all(review.status is ReviewStatus.PENDING for review in reviews)
    assert all(review.source_quote for review in reviews)


async def test_a_high_value_threshold_is_queued_regardless_of_confidence(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """CLAUDE.md 5: any monetary threshold over RM100m goes to a human.

    The trust deed carries a minimum net worth covenant of RM500,000,000,
    extracted at confidence 0.88 -- above the review threshold. It must still
    be queued, because at this size a misread number is a portfolio-level
    error regardless of how sure the extractor was.
    """
    result = await db_session.execute(
        select(HumanReview).where(HumanReview.trigger_reason == ReviewTrigger.HIGH_VALUE_THRESHOLD)
    )
    reviews = list(result.scalars().all())

    assert reviews
    assert all(Decimal(review.new_value or "0") > Decimal("100000000") for review in reviews)
    assert all(review.confidence is not None for review in reviews)
    # Confidence alone would not have flagged this one.
    assert any((review.confidence or 0) >= 0.85 for review in reviews)


# -- idempotency and human work -------------------------------------------


async def test_re_extraction_is_skipped_when_the_identity_is_satisfied(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    outcome = await RuleExtractionService(db_session).extract_document(indexed_corpus[0])

    assert outcome.skipped is True
    assert outcome.clauses > 0


async def test_re_extraction_does_not_duplicate_clauses(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
) -> None:
    service = RuleExtractionService(db_session)
    before = len(await clauses_for(db_session, indexed_corpus[0]))

    await _force_reextract(db_session, service, indexed_corpus[0])

    assert len(await clauses_for(db_session, indexed_corpus[0])) == before


async def test_an_approved_clause_survives_re_extraction(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
) -> None:
    clauses = await clauses_for(db_session, indexed_corpus[0])
    approved = clauses[0]
    approved.review_status = ReviewStatus.APPROVED
    await db_session.commit()

    await _force_reextract(db_session, RuleExtractionService(db_session), indexed_corpus[0])

    # Machine output is disposable; human judgement is not.
    assert await db_session.get(Clause, approved.id) is not None


# -- the gate --------------------------------------------------------------


async def test_the_repository_refuses_an_unverified_citation(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    clause = (await clauses_for(db_session, indexed_corpus[0]))[0]
    forged = Clause(
        document_id=clause.document_id,
        source_chunk_id=clause.source_chunk_id,
        clause_type=clause.clause_type,
        clause_text="invented",
        page_number=clause.page_number,
        source_quote="a quote that appears in no chunk",
        citation_verified=False,
    )

    # CLAUDE.md 1.3 / 9, enforced at the last place that can stop it.
    with pytest.raises(ValueError, match="unverified citation"):
        await ClauseRepository(db_session).add_verified(forged)


async def test_extracting_an_unknown_document_raises(db_session: AsyncSession) -> None:
    with pytest.raises(LookupError):
        await RuleExtractionService(db_session).extract_document(uuid.uuid4())


async def _force_reextract(
    session: AsyncSession, service: RuleExtractionService, document_id: uuid.UUID
) -> None:
    """Clear the job guard so extraction runs again, as a version bump would."""
    from sqlalchemy import delete

    from app.db.models.ops import ExtractionJob
    from app.domain.enums import JobType

    await session.execute(
        delete(ExtractionJob).where(ExtractionJob.job_type == JobType.EXTRACT_COVENANT)
    )
    await session.commit()
    await service.extract_document(document_id)
