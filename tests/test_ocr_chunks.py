"""Chunking what the VLM read — the second half of the VLM feature.

The gap these cover: Phase 5 wrote `document_pages.ocr_text` and nothing read
it, so a scanned page was transcribed at real cost and remained invisible to
retrieval and to every extractor. A test that only checked the transcription
was stored would have passed throughout.

Everything here runs on `MockLLMProvider`, so no page image reaches a paid
provider (CLAUDE.md 7).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.documents import DocumentChunk
from app.db.repositories.documents import DocumentChunkRepository, DocumentPageRepository
from app.ingest.ocr_chunks import OcrChunkingService
from app.llm.mock import MockLLMProvider
from app.llm.router import LLMRouter
from app.llm.vlm import VlmService

pytestmark = pytest.mark.usefixtures("storage_root")


@pytest_asyncio.fixture
async def ocrd_document(db_session: AsyncSession, object_store: object) -> AsyncIterator[uuid.UUID]:
    """A scanned document, ingested and run through the VLM.

    `build_scanned_document` is a picture of a page with no text layer, so the
    confidence heuristic flags it exactly as a real scan would.
    """
    from app.ingest.service import IngestionService
    from tests.fixtures.synthetic_pdf import build_mixed_document

    ingestion = IngestionService(db_session, object_store)  # type: ignore[arg-type]
    outcome = await ingestion.upload(filename="scan.pdf", data=build_mixed_document())
    await ingestion.process(outcome.document.id)

    vlm = VlmService(db_session, router=LLMRouter(db_session, provider=MockLLMProvider()))
    await vlm.process_document(outcome.document.id)

    yield outcome.document.id


async def _chunks(session: AsyncSession, document_id: uuid.UUID) -> list[DocumentChunk]:
    return list(await DocumentChunkRepository(session).list_for_document(document_id))


class TestTranscriptionsBecomeChunks:
    async def test_an_ocrd_page_produces_chunks(
        self, db_session: AsyncSession, ocrd_document: uuid.UUID
    ) -> None:
        """The whole point: OCR output must reach `document_chunks`."""
        before = await _chunks(db_session, ocrd_document)
        outcome = await OcrChunkingService(db_session).rechunk_document(ocrd_document)
        after = await _chunks(db_session, ocrd_document)

        assert outcome.pages_rechunked > 0
        assert outcome.chunks_created > 0
        assert len(after) > len(before)

    async def test_the_new_chunks_carry_the_transcribed_text(
        self, db_session: AsyncSession, ocrd_document: uuid.UUID
    ) -> None:
        await OcrChunkingService(db_session).rechunk_document(ocrd_document)

        pages = await DocumentPageRepository(db_session).list_for_document(ocrd_document)
        ocrd = [page for page in pages if page.ocr_text]
        assert ocrd

        chunks = await _chunks(db_session, ocrd_document)
        for page in ocrd:
            on_page = [c for c in chunks if c.page_number == page.page_number]
            assert on_page, f"page {page.page_number} was transcribed but never chunked"
            assert any(c.chunk_text.strip() in (page.ocr_text or "") for c in on_page)

    async def test_spans_resolve_against_the_stored_transcription(
        self, db_session: AsyncSession, ocrd_document: uuid.UUID
    ) -> None:
        """CLAUDE.md 1.2: a chunk's offsets must reproduce its text.

        For an OCR'd page the anchor is `ocr_text`, not the text layer that
        failed -- so a citation into a scanned page resolves against what the
        model actually read.
        """
        await OcrChunkingService(db_session).rechunk_document(ocrd_document)

        pages = {
            page.page_number: (page.ocr_text or "")
            for page in await DocumentPageRepository(db_session).list_for_document(ocrd_document)
            if page.ocr_text
        }
        for chunk in await _chunks(db_session, ocrd_document):
            source = pages.get(chunk.page_number)
            if source is None:
                continue  # a text-layer page, anchored elsewhere
            assert source[chunk.char_start : chunk.char_end] == chunk.chunk_text


class TestIdempotency:
    async def test_rechunking_twice_creates_nothing_new(
        self, db_session: AsyncSession, ocrd_document: uuid.UUID
    ) -> None:
        """CLAUDE.md 1.7: re-running an unchanged pipeline is a no-op."""
        service = OcrChunkingService(db_session)

        first = await service.rechunk_document(ocrd_document)
        count_after_first = len(await _chunks(db_session, ocrd_document))
        second = await service.rechunk_document(ocrd_document)

        assert first.chunks_created > 0
        assert second.chunks_created == 0
        assert second.chunks_already_present > 0
        assert len(await _chunks(db_session, ocrd_document)) == count_after_first

    async def test_ordinals_do_not_collide_with_text_layer_chunks(
        self, db_session: AsyncSession, ocrd_document: uuid.UUID
    ) -> None:
        """Ordinal orders a document's chunks; a restarted sequence interleaves them."""
        await OcrChunkingService(db_session).rechunk_document(ocrd_document)

        ordinals = [chunk.ordinal for chunk in await _chunks(db_session, ocrd_document)]
        assert len(ordinals) == len(set(ordinals)), "duplicate ordinals within one document"


class TestNothingToDo:
    async def test_a_document_with_no_transcription_is_a_no_op(
        self, db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
    ) -> None:
        """The text-layer corpus has no OCR'd pages, so this must not touch it."""
        document_id = indexed_corpus[0]
        before = len(await _chunks(db_session, document_id))

        outcome = await OcrChunkingService(db_session).rechunk_document(document_id)

        assert outcome.pages_rechunked == 0
        assert outcome.chunks_created == 0
        assert len(await _chunks(db_session, document_id)) == before


class TestOcrdChunksAreRetrievable:
    async def test_they_index_and_come_back_from_search(
        self, db_session: AsyncSession, ocrd_document: uuid.UUID
    ) -> None:
        """Chunked is not the goal; findable is.

        A chunk that never gets an embedding or an FTS vector is as invisible
        as the transcription was.
        """
        from app.retrieval.indexing import IndexingService

        await OcrChunkingService(db_session).rechunk_document(ocrd_document)
        await db_session.commit()
        await IndexingService(db_session).index_document(ocrd_document)

        result = await db_session.execute(
            select(DocumentChunk).where(
                DocumentChunk.document_id == ocrd_document,
                DocumentChunk.embedding.is_not(None),
                DocumentChunk.fts.is_not(None),
            )
        )
        assert list(result.scalars().all()), "OCR'd chunks were never indexed"
