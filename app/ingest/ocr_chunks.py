"""Turning a VLM transcription into retrievable chunks.

This is the second half of the VLM feature. Phase 3 detects pages that need the
vision model, Phase 5 OCRs them into `document_pages.ocr_text` -- and until
this module existed, that was where it stopped. The transcription was stored
and nothing read it, so a scanned page stayed invisible to retrieval and to
every extractor: money spent, page still unsearchable.

**The same chunker as the text layer, deliberately.** `chunk_page` does heading
detection, language detection, FTS config selection, span offsets and content
hashing; re-implementing any of that here would give OCR'd pages subtly
different provenance from parsed ones, and the citation chain (CLAUDE.md 1.2)
depends on both behaving identically.

Offsets are into `ocr_text`, which is what makes them verifiable: a citation
into an OCR'd page resolves against the transcription actually stored, not
against a text layer that failed to parse.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models.documents import DocumentChunk, DocumentPage
from app.db.repositories.documents import DocumentChunkRepository, DocumentPageRepository
from app.domain.enums import ParseMethod
from app.domain.ingest import ChunkDraft, PageAssessment, PageMetrics, ParsedPage
from app.ingest.chunking import chunk_page

logger = get_logger(__name__)


@dataclass(frozen=True)
class RechunkOutcome:
    document_id: uuid.UUID
    pages_rechunked: int
    chunks_created: int
    chunks_already_present: int

    @property
    def did_work(self) -> bool:
        return self.chunks_created > 0


class OcrChunkingService:
    """Chunk the pages a VLM transcribed, so they become retrievable."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._pages = DocumentPageRepository(session)
        self._chunks = DocumentChunkRepository(session)

    async def rechunk_document(self, document_id: uuid.UUID) -> RechunkOutcome:
        """Chunk every OCR'd page of a document. Idempotent.

        Re-running creates nothing new: chunk hashes cover `(page, span, text)`,
        so an unchanged transcription produces the hashes already stored
        (CLAUDE.md 1.7).
        """
        pages = [
            page for page in await self._pages.list_for_document(document_id) if _has_ocr(page)
        ]
        if not pages:
            logger.info("no OCR'd pages to chunk", extra={"document_id": str(document_id)})
            return RechunkOutcome(document_id, 0, 0, 0)

        existing = await self._chunks.list_for_document(document_id, limit=100_000)
        known_hashes = {chunk.hash for chunk in existing}
        # Continue the document's numbering rather than restarting it: ordinal
        # is what orders a document's chunks, and a second sequence starting at
        # zero would interleave OCR'd pages into the middle of parsed ones.
        next_ordinal = max((chunk.ordinal for chunk in existing), default=-1) + 1

        created = 0
        duplicates = 0
        for page in sorted(pages, key=lambda item: item.page_number):
            drafts = chunk_page(_as_parsed_page(page), start_ordinal=next_ordinal)
            _assert_spans_are_real(page, drafts)
            for draft in drafts:
                if draft.hash in known_hashes:
                    duplicates += 1
                    continue
                self._session.add(_to_row(document_id, draft))
                known_hashes.add(draft.hash)
                created += 1
                next_ordinal += 1

        await self._session.flush()
        logger.info(
            "OCR pages chunked",
            extra={
                "document_id": str(document_id),
                "pages": len(pages),
                "chunks_created": created,
                "chunks_already_present": duplicates,
            },
        )
        return RechunkOutcome(document_id, len(pages), created, duplicates)

    async def documents_awaiting_rechunk(self) -> list[uuid.UUID]:
        """Documents holding a transcription that has not been chunked.

        The query that makes the gap visible: a page with `ocr_text` and no
        chunk of its own is spend that bought nothing retrievable.
        """
        result = await self._session.execute(
            select(DocumentPage.document_id)
            .where(DocumentPage.ocr_text.is_not(None), DocumentPage.vlm_used.is_(True))
            .distinct()
        )
        return list(result.scalars().all())


def _has_ocr(page: DocumentPage) -> bool:
    return bool(page.ocr_text and page.ocr_text.strip())


def _as_parsed_page(page: DocumentPage) -> ParsedPage:
    """Present an OCR'd row to the chunker as the page it now is.

    The metrics describe the transcription rather than the failed text layer,
    because that is the text being chunked. `needs_vlm` is False with no
    reasons: the VLM has already run, and this is its output.
    """
    text = page.ocr_text or ""
    return ParsedPage(
        page_number=page.page_number,
        text=text,
        metrics=PageMetrics(
            page_number=page.page_number,
            char_count=len(text),
            image_area_ratio=0.0,
            has_text_layer=True,
            garbled_unicode_ratio=0.0,
        ),
        assessment=PageAssessment(needs_vlm=False, confidence=page.confidence, reasons=()),
        parse_method=ParseMethod.VLM,
        # A transcription carries no anchored table spans. The prompt asks for
        # "|" column markers, which the chunker reads as ordinary text rather
        # than as a table it can cite coordinates for -- inventing spans here
        # is precisely what CLAUDE.md 1.2 forbids.
        tables=(),
    )


def _to_row(document_id: uuid.UUID, draft: ChunkDraft) -> DocumentChunk:
    return DocumentChunk(
        document_id=document_id,
        page_number=draft.page_number,
        section_title=draft.section_title,
        chunk_text=draft.text,
        chunk_type=draft.chunk_type,
        language=draft.language,
        char_start=draft.char_start,
        char_end=draft.char_end,
        ordinal=draft.ordinal,
        fts_config=draft.fts_config,
        hash=draft.hash,
    )


def _assert_spans_are_real(page: DocumentPage, drafts: list[ChunkDraft]) -> None:
    """Refuse a chunk whose offsets do not reproduce its text.

    The same guard ingestion applies, for the same reason: a wrong offset does
    not fail here, it fails much later as a citation nobody can verify.
    """
    text = page.ocr_text or ""
    for draft in drafts:
        if text[draft.char_start : draft.char_end] != draft.text:
            raise ValueError(
                f"OCR chunk on page {draft.page_number} cites "
                f"({draft.char_start}, {draft.char_end}), which does not reproduce its text"
            )


__all__ = ["OcrChunkingService", "RechunkOutcome"]
