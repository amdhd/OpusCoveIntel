"""Ingestion value objects.

Pure data carried between the parser, the confidence scorer and the chunker.
`domain/` imports nothing from `db/` or `llm/` (CLAUDE.md 3), so these are the
types the ingestion service translates into rows -- never ORM objects.

Every span type here carries `(page_number, char_start, char_end)` because the
citation chain (CLAUDE.md 1.2) is only as good as the offsets recorded at parse
time. Offsets are into the *page's* extracted text, not the document's.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import ChunkType, Language, ParseMethod, VlmReason


class PageMetrics(BaseModel):
    """Raw parse telemetry for one page -- the input to the VLM heuristic."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    char_count: int = Field(ge=0)
    image_area_ratio: float = Field(ge=0.0, le=1.0)
    has_text_layer: bool
    garbled_unicode_ratio: float = Field(ge=0.0, le=1.0)
    has_table_hint: bool = False
    table_extraction_failed: bool = False


class PageAssessment(BaseModel):
    """The heuristic's verdict for one page."""

    model_config = ConfigDict(frozen=True)

    needs_vlm: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: tuple[VlmReason, ...] = ()

    @model_validator(mode="after")
    def _reasons_agree_with_verdict(self) -> PageAssessment:
        if self.needs_vlm != bool(self.reasons):
            raise ValueError("needs_vlm must hold exactly when a reason was recorded")
        return self

    @property
    def reason_text(self) -> str | None:
        """Reasons as stored on `document_pages.vlm_reason`, or None if clean."""
        return ",".join(reason.value for reason in self.reasons) or None


class TextSpan(BaseModel):
    """A half-open `[char_start, char_end)` range within one page's text."""

    model_config = ConfigDict(frozen=True)

    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> TextSpan:
        if self.char_end < self.char_start:
            raise ValueError("char_end must not precede char_start")
        return self

    def overlaps(self, other: TextSpan) -> bool:
        return self.char_start < other.char_end and other.char_start < self.char_end


class ParsedPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    text: str
    metrics: PageMetrics
    assessment: PageAssessment
    parse_method: ParseMethod = ParseMethod.PYMUPDF
    # Spans of `text` that a table extractor claimed. Only tables that could be
    # anchored to real offsets appear here -- an unanchorable table is dropped
    # rather than given invented coordinates (CLAUDE.md 1.2).
    tables: tuple[TextSpan, ...] = ()


class ParsedDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_count: int = Field(ge=0)
    pages: tuple[ParsedPage, ...] = ()
    parse_confidence: float = Field(ge=0.0, le=1.0)
    language: Language = Language.UNKNOWN

    @property
    def pages_needing_vlm(self) -> tuple[ParsedPage, ...]:
        """Pages the heuristic flagged. Phase 5 turns this into spend."""
        return tuple(page for page in self.pages if page.assessment.needs_vlm)


class ChunkDraft(BaseModel):
    """A chunk before it becomes a row.

    `text` is verbatim `page_text[char_start:char_end]` -- never reflowed,
    normalised or re-rendered, because citation verification (CLAUDE.md 1.3)
    matches quotes against exactly these characters.
    """

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    ordinal: int = Field(ge=0)
    text: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    language: Language = Language.UNKNOWN
    fts_config: str = "english"
    section_title: str | None = None
    hash: str
