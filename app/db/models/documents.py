"""Document, page and chunk models -- the provenance spine.

CLAUDE.md 1.2 requires every extracted fact to trace to a span. That chain
starts here: chunks carry `(page_number, char_start, char_end)` so a covenant
can point at the exact characters it came from.
"""

from __future__ import annotations

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin, enum_column
from app.domain.enums import (
    ChunkType,
    DocumentStatus,
    DocumentType,
    Language,
    ParseMethod,
    SourceType,
)

# Fixed at table-definition time; changing it forces a re-embed and index
# rebuild (docs/plan.md 9, Q2), which is a migration, not a config toggle.
_VECTOR_DIM = get_settings().VECTOR_DIMENSION


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "documents"

    # Content hash is the deduplication key: the same PDF uploaded twice is one
    # document, regardless of filename.
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_uri: Mapped[str | None] = mapped_column(String(1024))
    byte_size: Mapped[int | None] = mapped_column(Integer)

    source_type: Mapped[SourceType] = enum_column(SourceType, default=SourceType.UPLOAD)
    document_type: Mapped[DocumentType] = enum_column(DocumentType, default=DocumentType.UNKNOWN)
    status: Mapped[DocumentStatus] = enum_column(
        DocumentStatus, default=DocumentStatus.UPLOADED, index=True
    )
    language: Mapped[Language] = enum_column(Language, default=Language.UNKNOWN)

    # "_guess" is deliberate: these are pre-extraction heuristics, superseded by
    # a linked Instrument once extraction confirms identity.
    issuer_name_guess: Mapped[str | None] = mapped_column(String(512))
    instrument_name_guess: Mapped[str | None] = mapped_column(String(512))

    page_count: Mapped[int | None] = mapped_column(Integer)
    parse_confidence: Mapped[float | None] = mapped_column(Float)
    classification_confidence: Mapped[float | None] = mapped_column(Float)
    uploaded_by: Mapped[str | None] = mapped_column(String(255))

    pages: Mapped[list[DocumentPage]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "parse_confidence IS NULL OR (parse_confidence >= 0 AND parse_confidence <= 1)",
            name="parse_confidence_range",
        ),
        CheckConstraint("page_count IS NULL OR page_count >= 0", name="page_count_non_negative"),
    )


class DocumentPage(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Per-page parse telemetry.

    Neither source plan had this table. It exists because page confidence is
    what routes VLM spend (CLAUDE.md 4) -- "why did this document cost $8?"
    must be answerable with a query, not a log grep.
    """

    __tablename__ = "document_pages"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)

    char_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    image_area_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    has_text_layer: Mapped[bool] = mapped_column(default=True, nullable=False)
    garbled_unicode_ratio: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    parse_method: Mapped[ParseMethod] = enum_column(ParseMethod, default=ParseMethod.PYMUPDF)
    vlm_used: Mapped[bool] = mapped_column(default=False, nullable=False)
    # Which confidence check tripped. Null when the VLM was not invoked.
    vlm_reason: Mapped[str | None] = mapped_column(String(255))
    # What the VLM actually transcribed. This is the entire product of the
    # spend: without it the page is marked `vlm_used` and re-processing is
    # excluded, so a discarded transcription is money paid for nothing.
    ocr_text: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)

    document: Mapped[Document] = relationship(back_populates="pages")

    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_document_pages_document_id_page"),
        CheckConstraint("page_number >= 1", name="page_number_positive"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
        CheckConstraint(
            "NOT vlm_used OR vlm_reason IS NOT NULL",
            name="vlm_use_requires_reason",
        ),
        # A page cannot be marked VLM-processed with nothing to show for it.
        # The flag excludes the page from re-processing, so an empty
        # transcription would make the spend both wasted and unrepeatable.
        CheckConstraint(
            "NOT vlm_used OR (ocr_text IS NOT NULL AND length(ocr_text) > 0)",
            name="vlm_use_requires_ocr_text",
        ),
    )


class DocumentChunk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A retrievable span of text, anchored to exact character offsets."""

    __tablename__ = "document_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    section_title: Mapped[str | None] = mapped_column(String(512))
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_type: Mapped[ChunkType] = enum_column(ChunkType, default=ChunkType.PARAGRAPH)
    language: Mapped[Language] = enum_column(Language, default=Language.UNKNOWN)

    # Offsets into the page's extracted text. The anchor for citation
    # verification (CLAUDE.md 1.3).
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Postgres ships no Malay stemmer, so BM chunks index under 'simple' while
    # English chunks use 'english' (CLAUDE.md 6). Stored per row because a
    # single document mixes both.
    fts_config: Mapped[str] = mapped_column(String(32), default="english", nullable=False)
    fts: Mapped[str | None] = mapped_column(TSVECTOR)

    embedding: Mapped[list[float] | None] = mapped_column(Vector(_VECTOR_DIM))
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    hash: Mapped[str] = mapped_column(String(64), nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        CheckConstraint("char_end >= char_start", name="char_span_ordered"),
        CheckConstraint("char_start >= 0", name="char_start_non_negative"),
        CheckConstraint("page_number >= 1", name="page_number_positive"),
        CheckConstraint("fts_config IN ('english', 'simple')", name="fts_config_supported"),
        UniqueConstraint("document_id", "hash", name="uq_document_chunks_document_id_hash"),
        Index("ix_document_chunks_fts", "fts", postgresql_using="gin"),
        # HNSW over cosine distance. Built now so Phase 4 retrieval has it;
        # it is inert until embeddings are populated.
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_document_chunks_document_id_page_number", "document_id", "page_number"),
    )
