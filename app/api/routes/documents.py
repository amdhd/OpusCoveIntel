"""Document upload and inspection.

Handlers translate HTTP to the ingestion service and back, and do nothing else
(CLAUDE.md 3, 9): no SQL, no parsing, no policy. The service decides what a
duplicate is, what a valid PDF is, and when work may be skipped.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_session
from app.domain.documents import (
    DocumentChunkRead,
    DocumentPageRead,
    DocumentRead,
    IngestionResponse,
    UploadResponse,
)
from app.domain.enums import DocumentStatus, DocumentType, SourceType
from app.ingest.service import (
    IngestionError,
    IngestionService,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    VlmBudgetExceededError,
)
from app.ingest.storage import ObjectStore, get_object_store

router = APIRouter(prefix="/documents", tags=["documents"])


def get_ingestion_service(
    session: AsyncSession = Depends(get_session),
    store: ObjectStore = Depends(get_object_store),
) -> IngestionService:
    return IngestionService(session, store)


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF for ingestion",
)
async def upload_document(
    response: Response,
    file: UploadFile = File(..., description="PDF to ingest"),
    source_type: SourceType = Form(SourceType.UPLOAD),
    document_type: DocumentType = Form(DocumentType.UNKNOWN),
    uploaded_by: str | None = Form(None),
    service: IngestionService = Depends(get_ingestion_service),
) -> UploadResponse:
    """Store the PDF and queue it for parsing.

    Returns 201 for a new document and 200 when the bytes were already known --
    a duplicate is a correct outcome, not an error (CLAUDE.md 1.7).
    """
    # Reject on the declared size before reading the body into memory; the
    # service re-checks the bytes it actually received.
    limit = get_settings().MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if file.size is not None and file.size > limit:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"upload exceeds the {get_settings().MAX_UPLOAD_SIZE_MB} MB limit",
        )

    data = await file.read()
    try:
        outcome = await service.upload(
            filename=file.filename or "upload.pdf",
            data=data,
            source_type=source_type,
            document_type=document_type,
            uploaded_by=uploaded_by,
        )
    except PayloadTooLargeError as exc:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
    except UnsupportedMediaTypeError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc

    if outcome.duplicate:
        response.status_code = status.HTTP_200_OK
    return UploadResponse(
        duplicate=outcome.duplicate,
        document=DocumentRead.model_validate(outcome.document),
    )


@router.get("", response_model=list[DocumentRead], summary="List documents")
async def list_documents(
    status_filter: DocumentStatus | None = None,
    limit: int = 50,
    offset: int = 0,
    service: IngestionService = Depends(get_ingestion_service),
) -> list[DocumentRead]:
    documents = await service.list_documents(
        status=status_filter, limit=min(limit, 200), offset=offset
    )
    return [DocumentRead.model_validate(document) for document in documents]


@router.get("/{document_id}", response_model=DocumentRead, summary="Get one document")
async def get_document(
    document_id: uuid.UUID,
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentRead:
    document = await service.get_document(document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="document not found")
    return DocumentRead.model_validate(document)


@router.get(
    "/{document_id}/pages",
    response_model=list[DocumentPageRead],
    summary="Per-page parse telemetry",
)
async def list_pages(
    document_id: uuid.UUID,
    service: IngestionService = Depends(get_ingestion_service),
) -> list[DocumentPageRead]:
    await _require_document(service, document_id)
    pages = await service.list_pages(document_id)
    return [DocumentPageRead.model_validate(page) for page in pages]


@router.get(
    "/{document_id}/chunks",
    response_model=list[DocumentChunkRead],
    summary="Chunks with their character spans",
)
async def list_chunks(
    document_id: uuid.UUID,
    limit: int = 200,
    service: IngestionService = Depends(get_ingestion_service),
) -> list[DocumentChunkRead]:
    await _require_document(service, document_id)
    chunks = await service.list_chunks(document_id, limit=min(limit, 1000))
    return [DocumentChunkRead.model_validate(chunk) for chunk in chunks]


@router.post(
    "/{document_id}/process",
    response_model=IngestionResponse,
    summary="Parse and chunk now, without waiting for the worker",
)
async def process_document(
    document_id: uuid.UUID,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionResponse:
    """Run ingestion inline.

    The worker picks queued documents up on its own; this exists for operators
    and demos that do not want to wait for a poll interval.
    """
    try:
        outcome = await service.process(document_id)
    except LookupError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="document not found") from exc
    except VlmBudgetExceededError as exc:
        # 422: the document is well-formed but cannot be processed under the
        # configured VLM page cap. Silently truncating it is not an option.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except IngestionError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return IngestionResponse(
        document_id=outcome.document_id,
        status=outcome.status,
        page_count=outcome.page_count,
        chunk_count=outcome.chunk_count,
        pages_flagged_for_vlm=outcome.pages_flagged_for_vlm,
        skipped=outcome.skipped,
    )


async def _require_document(service: IngestionService, document_id: uuid.UUID) -> None:
    if await service.get_document(document_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="document not found")
