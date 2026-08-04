"""Document, page and chunk repositories."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import func, select

from app.db.models.documents import Document, DocumentChunk, DocumentPage
from app.db.repositories.base import BaseRepository
from app.domain.enums import DocumentStatus


class DocumentRepository(BaseRepository[Document]):
    model = Document

    async def get_by_sha256(self, sha256: str) -> Document | None:
        """Deduplication lookup -- the same bytes are the same document."""
        result = await self.session.execute(select(Document).where(Document.sha256 == sha256))
        return result.scalar_one_or_none()

    async def exists_by_sha256(self, sha256: str) -> bool:
        result = await self.session.execute(
            select(func.count()).select_from(Document).where(Document.sha256 == sha256)
        )
        return int(result.scalar_one()) > 0

    async def list_by_status(
        self, status: DocumentStatus, *, limit: int = 100
    ) -> Sequence[Document]:
        result = await self.session.execute(
            select(Document)
            .where(Document.status == status)
            .order_by(Document.created_at)
            .limit(limit)
        )
        return result.scalars().all()


class DocumentPageRepository(BaseRepository[DocumentPage]):
    model = DocumentPage

    async def list_for_document(self, document_id: uuid.UUID) -> Sequence[DocumentPage]:
        result = await self.session.execute(
            select(DocumentPage)
            .where(DocumentPage.document_id == document_id)
            .order_by(DocumentPage.page_number)
        )
        return result.scalars().all()

    async def list_needing_vlm(
        self, document_id: uuid.UUID, *, confidence_threshold: float
    ) -> Sequence[DocumentPage]:
        """Low-confidence pages not yet sent to the VLM.

        The VLM budget cap (CLAUDE.md 4) is applied by the caller against the
        length of this list -- a document that would blow the cap must fail
        loudly rather than silently process the first N pages.
        """
        result = await self.session.execute(
            select(DocumentPage)
            .where(
                DocumentPage.document_id == document_id,
                DocumentPage.confidence < confidence_threshold,
                DocumentPage.vlm_used.is_(False),
            )
            .order_by(DocumentPage.page_number)
        )
        return result.scalars().all()

    async def count_vlm_used(self, document_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(DocumentPage)
            .where(DocumentPage.document_id == document_id, DocumentPage.vlm_used.is_(True))
        )
        return int(result.scalar_one())


class DocumentChunkRepository(BaseRepository[DocumentChunk]):
    model = DocumentChunk

    async def list_for_document(
        self, document_id: uuid.UUID, *, limit: int = 1000
    ) -> Sequence[DocumentChunk]:
        result = await self.session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.page_number, DocumentChunk.ordinal)
            .limit(limit)
        )
        return result.scalars().all()

    async def get_by_hash(self, document_id: uuid.UUID, chunk_hash: str) -> DocumentChunk | None:
        result = await self.session.execute(
            select(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.hash == chunk_hash,
            )
        )
        return result.scalar_one_or_none()

    async def count_embedded(self, document_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(DocumentChunk)
            .where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.embedding.is_not(None),
            )
        )
        return int(result.scalar_one())
