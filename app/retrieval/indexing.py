"""Populating the two retrieval columns: `fts` and `embedding`.

Phase 3 deliberately left both NULL -- ingestion's job is provenance, and a
tsvector written by a chunker is a retrieval decision in the wrong place. This
is where a chunk becomes findable.

Indexing is a pipeline stage like parsing and chunking, with the same identity
key (CLAUDE.md 1.7), so re-indexing an unchanged document is a no-op. It is
also still **$0**: the embedder is the offline hashing one until Phase 5.

The embedding model is recorded on every row. Vectors from two models are not
comparable, so `search_by_vector` filters on it -- and that filter is only
meaningful because this writes it.
"""

from __future__ import annotations

import datetime as dt
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models.documents import Document
from app.db.models.ops import ExtractionJob
from app.db.repositories.documents import DocumentChunkRepository, DocumentRepository
from app.db.repositories.ops import ExtractionJobRepository
from app.domain.enums import DocumentStatus, JobStatus, JobType
from app.llm.embeddings import Embedder, get_embedder

logger = get_logger(__name__)

INDEX_VERSION = "index-v1"
INDEX_PROMPT_VERSION = "v0"

# Chunks per embedding call. Irrelevant to the offline embedder and the reason
# the interface is batched at all: Phase 5's provider bills per request.
EMBED_BATCH_SIZE = 64


@dataclass(frozen=True)
class IndexingOutcome:
    document_id: uuid.UUID
    chunks_embedded: int
    chunks_indexed_for_fts: int
    embedding_model: str
    skipped: bool


class IndexingService:
    def __init__(self, session: AsyncSession, embedder: Embedder | None = None) -> None:
        self._session = session
        self._embedder = embedder or get_embedder()
        self._documents = DocumentRepository(session)
        self._chunks = DocumentChunkRepository(session)
        self._jobs = ExtractionJobRepository(session)

    async def index_document(self, document_id: uuid.UUID) -> IndexingOutcome:
        """Embed and FTS-index one document's chunks. Idempotent."""
        document = await self._documents.get(document_id)
        if document is None:
            raise LookupError(f"document {document_id} not found")

        job = await self._ensure_job(document)
        if job.status is JobStatus.SUCCEEDED:
            logger.info(
                "indexing skipped; extraction identity already satisfied",
                extra={"document_id": str(document_id), "model_id": self._embedder.model_id},
            )
            return IndexingOutcome(
                document_id=document_id,
                chunks_embedded=await self._chunks.count_embedded(document_id),
                chunks_indexed_for_fts=0,
                embedding_model=self._embedder.model_id,
                skipped=True,
            )

        job.status = JobStatus.RUNNING
        job.started_at = _now()
        await self._session.flush()

        try:
            fts_rows = await self._chunks.refresh_fts(document_id)
            embedded = await self._embed_chunks(document_id)

            document.status = DocumentStatus.EMBEDDED
            job.status = JobStatus.SUCCEEDED
            job.finished_at = _now()
            await self._session.commit()
        except Exception as exc:
            await self._session.rollback()
            await self._fail(document_id, exc)
            raise

        logger.info(
            "document indexed",
            extra={
                "document_id": str(document_id),
                "chunks_embedded": embedded,
                "chunks_indexed_for_fts": fts_rows,
                "embedding_model": self._embedder.model_id,
            },
        )
        return IndexingOutcome(
            document_id=document_id,
            chunks_embedded=embedded,
            chunks_indexed_for_fts=fts_rows,
            embedding_model=self._embedder.model_id,
            skipped=False,
        )

    async def _embed_chunks(self, document_id: uuid.UUID) -> int:
        chunks = list(await self._chunks.list_for_document(document_id, limit=100_000))
        embedded = 0
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[start : start + EMBED_BATCH_SIZE]
            vectors = await self._embedder.embed([chunk.chunk_text for chunk in batch])
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"embedder returned {len(vectors)} vectors for {len(batch)} chunks"
                )
            for chunk, vector in zip(batch, vectors, strict=True):
                if len(vector) != self._embedder.dimension:
                    raise RuntimeError(
                        f"embedder returned {len(vector)} dimensions, "
                        f"expected {self._embedder.dimension}"
                    )
                chunk.embedding = vector
                chunk.embedding_model = self._embedder.model_id
                embedded += 1
            await self._session.flush()
        return embedded

    async def _ensure_job(self, document: Document) -> ExtractionJob:
        job = await self._jobs.find_by_identity(
            document_sha256=document.sha256,
            job_type=JobType.EMBED,
            prompt_version=INDEX_PROMPT_VERSION,
            # The embedding model is part of the identity: switching models
            # must re-index rather than silently mix vector spaces.
            model_id=self._embedder.model_id,
            extractor_version=INDEX_VERSION,
        )
        if job is not None:
            return job
        return await self._jobs.add(
            ExtractionJob(
                document_id=document.id,
                document_sha256=document.sha256,
                job_type=JobType.EMBED,
                status=JobStatus.QUEUED,
                model_id=self._embedder.model_id,
                prompt_version=INDEX_PROMPT_VERSION,
                extractor_version=INDEX_VERSION,
            )
        )

    async def _fail(self, document_id: uuid.UUID, exc: BaseException) -> None:
        message = f"{type(exc).__name__}: {exc}"
        logger.error("indexing failed", extra={"document_id": str(document_id), "error": message})
        document = await self._documents.get(document_id)
        if document is None:
            return
        job = await self._ensure_job(document)
        if job.status is not JobStatus.SUCCEEDED:
            job.status = JobStatus.FAILED
            job.error_message = message[:2000]
            job.finished_at = _now()
        await self._session.commit()


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)
