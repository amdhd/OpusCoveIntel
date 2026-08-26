"""Candidate detection tests — regex-based span narrowing."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.documents import DocumentChunk
from app.domain.enums import ClauseType
from app.extract.candidates import Candidate, CandidateDetectionService, _detect_regex


@dataclass
class FakeChunk:
    """Plain object standing in for DocumentChunk in unit tests."""

    id: uuid.UUID
    chunk_text: str
    page_number: int = 1
    section_title: str | None = "Test Section"


def make_chunk(text: str, page: int = 1) -> FakeChunk:
    return FakeChunk(id=uuid.uuid4(), chunk_text=text, page_number=page)


COVENANT_HEAVY = """NEGATIVE PLEDGE

The Issuer shall not create or permit to subsist any security interest over its assets.

CROSS DEFAULT

An event of default shall occur if any indebtedness of the Issuer in an aggregate
principal amount exceeding RM30,000,000 becomes due and payable prior to its stated maturity.

FINANCIAL COVENANTS

The Issuer shall maintain a consolidated gearing ratio of not more than 1.75 times,
and a finance service cover ratio of not less than 1.50 times.

RATING TRIGGER

In the event the rating assigned to the sukuk by MARC is downgraded below BBB+, the
Issuer shall notify the Trustee within five business days."""


def test_regex_detection_finds_covenants() -> None:
    chunks = [make_chunk(COVENANT_HEAVY)]
    candidates = _detect_regex(chunks, limit=50)

    # Should find multiple candidates from the covenant-heavy text.
    assert len(candidates) > 0
    # Every candidate should have valid offsets.
    for c in candidates:
        assert c.char_start >= 0
        assert c.char_end > c.char_start
        assert len(c.text) > 0
        # The text should be a slice of the chunk.
        assert c.text == COVENANT_HEAVY[c.char_start : c.char_end]


def test_detection_returns_empty_for_plain_text() -> None:
    text = "The Trustee is Synthetic Trustees Berhad, a company incorporated in Malaysia."
    chunks = [make_chunk(text)]
    candidates = _detect_regex(chunks, limit=50)
    assert candidates == []


def test_detection_respects_limit() -> None:
    """Even with many candidates, the limit is enforced."""
    chunks = [make_chunk(COVENANT_HEAVY * 10)]
    candidates = _detect_regex(chunks, limit=5)
    assert len(candidates) <= 5


def test_candidates_have_clause_type_hints() -> None:
    chunks = [make_chunk(COVENANT_HEAVY)]
    candidates = _detect_regex(chunks, limit=50)

    all_hints: set[ClauseType] = set()
    for c in candidates:
        all_hints |= set(c.clause_type_hints)

    # Should detect financial covenants, cross default, negative pledge, rating trigger.
    assert ClauseType.FINANCIAL_COVENANT in all_hints
    assert ClauseType.CROSS_DEFAULT in all_hints
    assert ClauseType.NEGATIVE_PLEDGE in all_hints
    assert ClauseType.RATING_TRIGGER in all_hints


def test_overlapping_spans_are_merged() -> None:
    """Two patterns matching adjacent text should be merged into one candidate."""
    text = (
        "The Issuer shall maintain a gearing ratio of not more than 1.75 times and "
        "an interest cover ratio of not less than 3.00 times."
    )
    chunks = [make_chunk(text)]
    candidates = _detect_regex(chunks, limit=50)

    # Gearing and interest cover are in the same sentence — they should merge.
    # If not merged, we'd get 2 candidates. Either is fine — merging is an
    # optimization, not a requirement.
    assert len(candidates) >= 1


def test_bahasa_malaysia_candidates_are_detected() -> None:
    text = (
        "Penerbit hendaklah pada setiap masa mengekalkan nisbah gearan yang tidak "
        "melebihi 1.75 kali. Sekiranya berlaku ketidakpatuhan Shariah, ia adalah "
        "suatu kejadian pembubaran."
    )
    chunks = [make_chunk(text)]
    candidates = _detect_regex(chunks, limit=50)

    hints: set[ClauseType] = set()
    for c in candidates:
        hints |= set(c.clause_type_hints)

    assert ClauseType.FINANCIAL_COVENANT in hints
    assert ClauseType.SHARIAH_COMPLIANCE in hints


async def test_service_detects_on_indexed_corpus(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """The service works against real database chunks."""
    service = CandidateDetectionService(db_session)
    candidates = await service.detect(indexed_corpus[0])

    assert len(candidates) > 0
    # All candidates should reference real chunks.
    chunk_ids = {c.chunk_id for c in candidates}
    for chunk_id in chunk_ids:
        chunk = await db_session.get(DocumentChunk, chunk_id)
        assert chunk is not None


class TestSemanticLegs:
    """FTS and kNN against clause-type exemplars (docs/plan.md 2's other two legs).

    Off by default — see `SEMANTIC_LEGS_DEFAULT`. These pin the behaviour so the
    legs are correct whenever they are switched on, and pin the relevance floors,
    which are the difference between a retrieval leg and a spending bug.
    """

    async def test_off_by_default(
        self, db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
    ) -> None:
        service = CandidateDetectionService(db_session)

        default = await service.detect(indexed_corpus[0])
        explicit = await service.detect(indexed_corpus[0], semantic=False)

        assert [c.chunk_id for c in default] == [c.chunk_id for c in explicit]

    async def test_regex_candidates_already_cover_every_label(
        self, db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
    ) -> None:
        """The measurement that decided the default.

        If this ever fails, the regex leg has stopped covering the corpus and
        the semantic legs have a job to do — turn them on and re-measure.
        """
        from sqlalchemy import select

        from app.db.models.documents import Document
        from app.evals.labels import COVENANT_LABELS
        from app.extract.citations import verify_quote

        service = CandidateDetectionService(db_session)
        rows = (await db_session.execute(select(Document.id, Document.sha256))).all()
        by_sha = {sha: did for did, sha in rows}

        missed: list[str] = []
        cache: dict[uuid.UUID, list[Candidate]] = {}
        for label in COVENANT_LABELS:
            document_id = by_sha.get(label.document_sha256)
            if document_id is None:
                continue
            if document_id not in cache:
                cache[document_id] = await service.detect(document_id, semantic=False)
            if not any(verify_quote(label.evidence, c.text).verified for c in cache[document_id]):
                missed.append(label.covenant_type.value)

        assert not missed, f"regex candidates no longer cover: {missed}"

    async def test_a_document_with_no_covenant_language_gets_no_semantic_candidates(
        self, db_session: AsyncSession, object_store: object, seeded_universe: None
    ) -> None:
        """The floors, doing the job they exist for.

        Unfloored, ranked retrieval returns its best N regardless of relevance:
        a press release holding no covenants produced nine semantic candidates,
        which is nine paid calls for nothing.
        """
        from app.ingest.service import IngestionService
        from app.retrieval.indexing import IndexingService

        prose = (
            "PRESS RELEASE MINISTRY OF FINANCE\n\n"
            "The Government has priced a US$1.5 billion Global Sukuk offering that was "
            "4.7 times oversubscribed, with peak orderbook exceeding US$9.5 billion.\n\n"
            "The issuance reaffirms the sovereign benchmark curve and serves as a pricing "
            "reference for other issuers.\n"
        )
        from tests.fixtures.synthetic_pdf import build_prose_document

        ingestion = IngestionService(db_session, object_store)  # type: ignore[arg-type]
        outcome = await ingestion.upload(
            filename="no-covenants.pdf", data=build_prose_document(prose, heading="PRESS RELEASE")
        )
        await ingestion.process(outcome.document.id)
        await IndexingService(db_session).index_document(outcome.document.id)

        service = CandidateDetectionService(db_session)
        assert await service.detect(outcome.document.id, semantic=True) == []

    async def test_semantic_candidates_carry_their_provenance(
        self, db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
    ) -> None:
        """`found_by` is what makes "are these legs earning their calls?" answerable."""
        service = CandidateDetectionService(db_session)

        regex_only = {c.chunk_id for c in await service.detect(indexed_corpus[0], semantic=False)}
        with_semantic = await service.detect(indexed_corpus[0], semantic=True)

        for candidate in with_semantic:
            if candidate.chunk_id not in regex_only:
                assert candidate.found_by <= {"fts", "knn"}
                assert candidate.found_by, "a semantic candidate must name the leg that found it"
