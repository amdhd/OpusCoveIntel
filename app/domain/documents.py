"""Read schemas for the document API.

Separate from the ORM models so the wire format is a deliberate choice rather
than whatever the table happens to hold. `from_attributes` lets a route return
`DocumentRead.model_validate(document)` without hand-copying fields.
"""

from __future__ import annotations

import datetime as dt
import uuid

from pydantic import BaseModel, ConfigDict

from app.domain.enums import (
    ChunkType,
    DocumentStatus,
    DocumentType,
    Language,
    ParseMethod,
    SourceType,
)


class DocumentRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    sha256: str
    filename: str
    storage_uri: str | None
    byte_size: int | None
    source_type: SourceType
    document_type: DocumentType
    status: DocumentStatus
    language: Language
    issuer_name_guess: str | None
    page_count: int | None
    parse_confidence: float | None
    uploaded_by: str | None
    created_at: dt.datetime
    updated_at: dt.datetime


class DocumentPageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    page_number: int
    char_count: int
    image_area_ratio: float
    has_text_layer: bool
    garbled_unicode_ratio: float
    parse_method: ParseMethod
    vlm_used: bool
    vlm_reason: str | None
    confidence: float


class DocumentChunkRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    page_number: int
    ordinal: int
    section_title: str | None
    chunk_type: ChunkType
    language: Language
    fts_config: str
    char_start: int
    char_end: int
    hash: str
    chunk_text: str


class UploadResponse(BaseModel):
    """Upload outcome.

    `duplicate` is a normal result, not an error: the same bytes uploaded twice
    are one document (CLAUDE.md 1.7), and the caller gets the existing row back.
    """

    duplicate: bool
    document: DocumentRead


class IngestionResponse(BaseModel):
    document_id: uuid.UUID
    status: DocumentStatus
    page_count: int
    chunk_count: int
    pages_flagged_for_vlm: int
    # True when the extraction identity had already succeeded, so nothing ran.
    skipped: bool
