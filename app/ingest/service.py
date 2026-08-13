"""Ingestion service -- the transaction owner for the document pipeline.

Repositories never commit (transaction scope belongs to the caller) and route
handlers hold no business logic (CLAUDE.md 3, 9), so this is where both live.

Two properties are worth stating outright:

**Deduplication is by content, not by name.** The same prospectus uploaded
twice under two filenames is one document. `documents.sha256` is unique, so
this holds even when two uploads race.

**Re-processing is a no-op.** The extraction identity `(document_sha256,
job_type, prompt_version, model_id, extractor_version)` is unique in the
schema (CLAUDE.md 1.7). When both stage jobs already succeeded, `process()`
returns without re-reading the PDF. When they have not, the write path upserts
on `(document_id, page_number)` and `(document_id, hash)`, so a retry after a
crash converges instead of duplicating.

Phase 3 spends nothing: `model_id` is `"none"` and no adapter is imported.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.models.documents import Document, DocumentChunk, DocumentPage
from app.db.models.ops import ExtractionJob
from app.db.repositories.documents import (
    DocumentChunkRepository,
    DocumentPageRepository,
    DocumentRepository,
)
from app.db.repositories.ops import ExtractionJobRepository
from app.domain.enums import (
    DocumentStatus,
    DocumentType,
    JobStatus,
    JobType,
    SourceType,
)
from app.domain.ingest import ChunkDraft, ParsedDocument
from app.ingest.chunking import chunk_document
from app.ingest.pdf import PdfParseError, parse_pdf
from app.ingest.storage import ObjectStore

logger = get_logger(__name__)

# Bump when parsing, scoring or chunking changes in a way that should force
# re-ingestion of already-parsed documents. It is part of the identity key.
INGEST_VERSION = "ingest-v1"
INGEST_PROMPT_VERSION = "v0"
# No model is involved in Phase 3; the identity key still needs a value.
INGEST_MODEL_ID = "none"

_PDF_MAGIC = b"%PDF-"
_STAGES = (JobType.PARSE, JobType.CHUNK)


class IngestionError(RuntimeError):
    """Ingestion failed in a way the caller should surface."""


class UnsupportedMediaTypeError(IngestionError):
    pass


class PayloadTooLargeError(IngestionError):
    pass


class VlmBudgetExceededError(IngestionError):
    """More pages need the VLM than the per-document cap allows.

    CLAUDE.md 4: a 400-page scan that trips every check must fail loudly, not
    quietly spend $80. The page telemetry is still persisted so an operator can
    see exactly which pages caused it.
    """


def storage_key(sha256: str) -> str:
    """Content-addressed key. Fanned out so no directory holds every document."""
    return f"documents/{sha256[:2]}/{sha256[2:4]}/{sha256}.pdf"


@dataclass(frozen=True)
class UploadOutcome:
    document: Document
    duplicate: bool


@dataclass(frozen=True)
class IngestionOutcome:
    document_id: uuid.UUID
    status: DocumentStatus
    page_count: int
    chunk_count: int
    pages_flagged_for_vlm: int
    skipped: bool


class IngestionService:
    def __init__(
        self,
        session: AsyncSession,
        store: ObjectStore,
        settings: Settings | None = None,
    ) -> None:
        self._session = session
        self._store = store
        self._settings = settings or get_settings()
        self._documents = DocumentRepository(session)
        self._pages = DocumentPageRepository(session)
        self._chunks = DocumentChunkRepository(session)
        self._jobs = ExtractionJobRepository(session)

    # -- upload ------------------------------------------------------------

    async def upload(
        self,
        *,
        filename: str,
        data: bytes,
        source_type: SourceType = SourceType.UPLOAD,
        document_type: DocumentType = DocumentType.UNKNOWN,
        uploaded_by: str | None = None,
    ) -> UploadOutcome:
        """Store a PDF, record it, and queue it for parsing."""
        self._validate(data)
        digest = hashlib.sha256(data).hexdigest()

        existing = await self._documents.get_by_sha256(digest)
        if existing is not None:
            logger.info(
                "duplicate upload ignored",
                extra={"sha256": digest, "document_id": str(existing.id)},
            )
            return UploadOutcome(document=existing, duplicate=True)

        # Written before the row so a committed document always has its bytes.
        # The reverse leaves an orphan object, which is inert and reclaimable;
        # a row pointing at nothing is neither.
        uri = await self._store.put(storage_key(digest), data, content_type="application/pdf")

        document = Document(
            sha256=digest,
            filename=_safe_filename(filename),
            storage_uri=uri,
            byte_size=len(data),
            source_type=source_type,
            document_type=document_type,
            status=DocumentStatus.UPLOADED,
            uploaded_by=uploaded_by,
        )
        try:
            await self._documents.add(document)
            await self._ensure_job(document, JobType.PARSE)
            await self._session.commit()
        except IntegrityError:
            # Two uploads of the same bytes raced; the unique constraint on
            # sha256 decided which one won.
            await self._session.rollback()
            winner = await self._documents.get_by_sha256(digest)
            if winner is None:
                raise
            return UploadOutcome(document=winner, duplicate=True)

        logger.info(
            "document uploaded",
            extra={"document_id": str(document.id), "sha256": digest, "bytes": len(data)},
        )
        return UploadOutcome(document=document, duplicate=False)

    def _validate(self, data: bytes) -> None:
        limit = self._settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if len(data) > limit:
            raise PayloadTooLargeError(
                f"upload is {len(data)} bytes, above the "
                f"{self._settings.MAX_UPLOAD_SIZE_MB} MB limit"
            )
        if not data.startswith(_PDF_MAGIC):
            raise UnsupportedMediaTypeError("only PDF uploads are supported")

    # -- processing --------------------------------------------------------

    async def process(self, document_id: uuid.UUID) -> IngestionOutcome:
        """Parse, score and chunk a document. Idempotent."""
        document = await self._documents.get(document_id)
        if document is None:
            raise LookupError(f"document {document_id} not found")

        jobs = {stage: await self._ensure_job(document, stage) for stage in _STAGES}
        if all(job.status is JobStatus.SUCCEEDED for job in jobs.values()):
            logger.info(
                "ingestion skipped; extraction identity already satisfied",
                extra={"document_id": str(document_id), "extractor_version": INGEST_VERSION},
            )
            return IngestionOutcome(
                document_id=document_id,
                status=document.status,
                page_count=document.page_count or 0,
                chunk_count=await self._chunks.count(document_id=document_id),
                pages_flagged_for_vlm=await self._flagged_page_count(document_id),
                skipped=True,
            )

        try:
            parsed = await self._run_parse(document, jobs[JobType.PARSE])
            chunk_count = await self._run_chunk(document, parsed, jobs[JobType.CHUNK])
            await self._session.commit()
        except Exception as exc:
            await self._session.rollback()
            await self._record_failure(document_id, exc)
            raise

        flagged = len(parsed.pages_needing_vlm)
        logger.info(
            "document ingested",
            extra={
                "document_id": str(document_id),
                "page_count": parsed.page_count,
                "chunk_count": chunk_count,
                "pages_flagged_for_vlm": flagged,
                "parse_confidence": round(parsed.parse_confidence, 4),
            },
        )
        return IngestionOutcome(
            document_id=document_id,
            status=DocumentStatus.CHUNKED,
            page_count=parsed.page_count,
            chunk_count=chunk_count,
            pages_flagged_for_vlm=flagged,
            skipped=False,
        )

    async def _run_parse(self, document: Document, job: ExtractionJob) -> ParsedDocument:
        self._start(job)
        document.status = DocumentStatus.PARSING
        await self._session.flush()

        data = await self._store.get(storage_key(document.sha256))
        actual = hashlib.sha256(data).hexdigest()
        if actual != document.sha256:
            raise IngestionError(
                f"stored bytes for {document.id} hash to {actual}, expected {document.sha256}"
            )

        try:
            # Parsing is CPU-bound; keep it off the event loop.
            parsed = await asyncio.to_thread(
                parse_pdf, data, max_pages=self._settings.MAX_PDF_PAGES
            )
        except PdfParseError as exc:
            raise IngestionError(str(exc)) from exc

        await self._upsert_pages(document, parsed)

        document.page_count = parsed.page_count
        document.parse_confidence = parsed.parse_confidence
        document.language = parsed.language

        flagged = parsed.pages_needing_vlm
        if len(flagged) > self._settings.MAX_VLM_PAGES_PER_DOC:
            # Commit the page telemetry before failing. The cap exists so a
            # 400-page scan fails loudly instead of spending $80 (CLAUDE.md 4),
            # and "which pages tripped it?" must be answerable by query rather
            # than by re-running the parser.
            await self._session.commit()
            raise VlmBudgetExceededError(
                f"{len(flagged)} of {parsed.page_count} pages need the VLM, above the "
                f"cap of {self._settings.MAX_VLM_PAGES_PER_DOC}"
            )

        document.status = DocumentStatus.PARSED
        self._finish(job)
        await self._session.flush()
        return parsed

    async def _upsert_pages(self, document: Document, parsed: ParsedDocument) -> None:
        rows = await self._pages.list_for_document(document.id)
        existing = {page.page_number: page for page in rows}

        for page in parsed.pages:
            row = existing.get(page.page_number)
            values = {
                "char_count": page.metrics.char_count,
                "image_area_ratio": page.metrics.image_area_ratio,
                "has_text_layer": page.metrics.has_text_layer,
                "garbled_unicode_ratio": page.metrics.garbled_unicode_ratio,
                "parse_method": page.parse_method,
                # Phase 3 detects; Phase 5 invokes. `vlm_used` stays false, and
                # the CHECK constraint only demands a reason once it is true.
                "vlm_reason": page.assessment.reason_text,
                "confidence": page.assessment.confidence,
            }
            if row is None:
                self._session.add(
                    DocumentPage(document_id=document.id, page_number=page.page_number, **values)
                )
            else:
                for field, value in values.items():
                    setattr(row, field, value)

        await self._session.flush()

    async def _run_chunk(
        self, document: Document, parsed: ParsedDocument, job: ExtractionJob
    ) -> int:
        self._start(job)
        drafts = chunk_document(parsed)
        _assert_spans_are_real(parsed, drafts)

        known = {
            chunk.hash for chunk in await self._chunks.list_for_document(document.id, limit=100_000)
        }
        for draft in drafts:
            if draft.hash in known:
                continue
            self._session.add(
                DocumentChunk(
                    document_id=document.id,
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
            )

        document.status = DocumentStatus.CHUNKED
        self._finish(job)
        await self._session.flush()
        return len(drafts)

    async def _record_failure(self, document_id: uuid.UUID, exc: BaseException) -> None:
        """Mark the document and its unfinished jobs failed, in a new transaction."""
        message = f"{type(exc).__name__}: {exc}"
        logger.error("ingestion failed", extra={"document_id": str(document_id), "error": message})
        document = await self._documents.get(document_id)
        if document is None:
            return
        document.status = DocumentStatus.FAILED
        for stage in _STAGES:
            job = await self._find_job(document, stage)
            if job is not None and job.status is not JobStatus.SUCCEEDED:
                job.status = JobStatus.FAILED
                job.error_message = message[:2000]
                job.finished_at = _now()
        await self._session.commit()

    # -- jobs --------------------------------------------------------------

    async def _find_job(self, document: Document, job_type: JobType) -> ExtractionJob | None:
        return await self._jobs.find_by_identity(
            document_sha256=document.sha256,
            job_type=job_type,
            prompt_version=INGEST_PROMPT_VERSION,
            model_id=INGEST_MODEL_ID,
            extractor_version=INGEST_VERSION,
        )

    async def _ensure_job(self, document: Document, job_type: JobType) -> ExtractionJob:
        job = await self._find_job(document, job_type)
        if job is not None:
            return job
        return await self._jobs.add(
            ExtractionJob(
                document_id=document.id,
                document_sha256=document.sha256,
                job_type=job_type,
                status=JobStatus.QUEUED,
                model_id=INGEST_MODEL_ID,
                prompt_version=INGEST_PROMPT_VERSION,
                extractor_version=INGEST_VERSION,
            )
        )

    @staticmethod
    def _start(job: ExtractionJob) -> None:
        job.status = JobStatus.RUNNING
        job.started_at = _now()
        job.error_message = None

    @staticmethod
    def _finish(job: ExtractionJob) -> None:
        job.status = JobStatus.SUCCEEDED
        job.finished_at = _now()

    # -- reads -------------------------------------------------------------

    async def get_document(self, document_id: uuid.UUID) -> Document | None:
        return await self._documents.get(document_id)

    async def list_documents(
        self,
        *,
        status: DocumentStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Document]:
        filters = {"status": status} if status is not None else {}
        return await self._documents.list(limit=limit, offset=offset, **filters)

    async def list_pages(self, document_id: uuid.UUID) -> Sequence[DocumentPage]:
        return await self._pages.list_for_document(document_id)

    async def list_chunks(
        self, document_id: uuid.UUID, *, limit: int = 200
    ) -> Sequence[DocumentChunk]:
        return await self._chunks.list_for_document(document_id, limit=limit)

    async def _flagged_page_count(self, document_id: uuid.UUID) -> int:
        pages = await self._pages.list_for_document(document_id)
        return sum(1 for page in pages if page.vlm_reason)


async def ingest_and_index(
    session: AsyncSession,
    store: ObjectStore,
    document_id: uuid.UUID,
    settings: Settings | None = None,
) -> IngestionOutcome:
    """Parse and chunk a document, then make it searchable.

    **Ingesting used to stop at chunked.** Indexing was a separate `opuscovintel
    index` nobody ran, so a document could sit in the corpus, appear on the
    Documents screen with a healthy status, and be invisible to every question
    asked about it -- both retrieval legs read columns that were still null. A
    user uploaded three real prospectuses, asked about one, and was answered
    from the synthetic fixtures instead (docs/review.md, findings 11 and 15).

    One function rather than a step in each caller: the worker and the
    `/documents/{id}/process` endpoint both ingest, and a document that reached
    the corpus through the endpoint would otherwise never be indexed at all --
    the worker has no queued parse job left to claim.

    Indexing failures propagate. The document is parsed either way, and the
    `embed` job records the failure, so the status endpoint shows a document
    that is ingested and not searchable rather than silently claiming both.
    """
    # Imported here rather than at module scope: `app.retrieval` pulls in the
    # embedder and its provider configuration, and ingestion is used by paths
    # (the CLI's `ingest`, the upload endpoint) that must keep working when
    # none of that is configured.
    from app.retrieval.indexing import IndexingService

    outcome = await IngestionService(session, store, settings).process(document_id)
    indexed = await IndexingService(session).index_document(document_id)
    await session.commit()

    logger.info(
        "document ingested and indexed",
        extra={
            "document_id": str(document_id),
            "chunks_embedded": indexed.chunks_embedded,
            "embedding_model": indexed.embedding_model,
        },
    )
    return outcome


def _assert_spans_are_real(parsed: ParsedDocument, drafts: list[ChunkDraft]) -> None:
    """Refuse to persist a chunk whose offsets do not reproduce its text.

    CLAUDE.md 1.2 makes the span the load-bearing part of a chunk. A silently
    wrong offset would not fail here -- it would fail much later, as a citation
    that cannot be verified against a quote nobody can find.
    """
    pages = {page.page_number: page.text for page in parsed.pages}
    for draft in drafts:
        text = pages.get(draft.page_number)
        if text is None:
            raise IngestionError(f"chunk cites page {draft.page_number}, which was not parsed")
        if text[draft.char_start : draft.char_end] != draft.text:
            raise IngestionError(
                f"chunk span ({draft.page_number}, {draft.char_start}, {draft.char_end}) "
                "does not reproduce its text"
            )


def _safe_filename(filename: str) -> str:
    """Keep the leaf only: the client's path is not ours."""
    name = PurePosixPath(filename.replace("\\", "/")).name.strip()
    return (name or "upload.pdf")[:512]


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)
