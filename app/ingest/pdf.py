"""PDF parsing: PyMuPDF for text, pdfplumber for tables.

Two libraries because they are good at different things. PyMuPDF gives fast,
reading-order text and image geometry; pdfplumber understands ruled tables,
which PyMuPDF flattens into ambiguous prose. The PyMuPDF text is *canonical*:
all character offsets -- and therefore the entire citation chain (CLAUDE.md
1.2) -- are into it. pdfplumber only tells us *where* a table is, so a table
chunk is still a verbatim slice of the canonical text rather than a re-rendered
grid that no quote could ever be matched against.

A table that cannot be anchored to real offsets is dropped, not persisted with
guessed coordinates. An unanchored table costs us one chunk of table typing;
a fabricated span would silently corrupt every citation drawn from it.

Nothing here calls a model. Phase 3 is $0.
"""

from __future__ import annotations

import io
import re
import unicodedata
from typing import Any

import pdfplumber
import pymupdf

from app.core.logging import get_logger
from app.domain.enums import Language, ParseMethod
from app.domain.ingest import PageMetrics, ParsedDocument, ParsedPage, TextSpan
from app.ingest.confidence import assess_page, document_confidence
from app.ingest.language import aggregate_language, detect_language

logger = get_logger(__name__)

# Mojibake marker from PDFs whose fonts carry no usable ToUnicode map -- common
# in older Malaysian scans, and invisible to a plain "did we get text?" check.
_CID_PATTERN = re.compile(r"\(cid:\d+\)")
# Control, private-use and unassigned characters. Tabs and newlines are real
# layout, not damage.
_GARBLED_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cn"})
_ALLOWED_CONTROLS = frozenset("\n\r\t")


class PdfParseError(RuntimeError):
    """The PDF could not be parsed. Fail loudly (CLAUDE.md 4)."""


def parse_pdf(data: bytes, *, max_pages: int) -> ParsedDocument:
    """Parse a PDF into pages with text, metrics and a confidence verdict.

    CPU-bound and synchronous by design; callers run it on a worker thread.
    """
    tables_by_page = _extract_tables(data)

    try:
        document = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PdfParseError(f"could not open PDF: {exc}") from exc

    with document:
        if document.needs_pass:
            raise PdfParseError("PDF is password-protected")
        if document.page_count > max_pages:
            raise PdfParseError(
                f"PDF has {document.page_count} pages, above the {max_pages}-page limit"
            )

        pages: list[ParsedPage] = []
        for index in range(document.page_count):
            page_number = index + 1
            cells, extraction_failed = tables_by_page.get(page_number, ([], False))
            pages.append(
                _parse_page(document.load_page(index), page_number, cells, extraction_failed)
            )

    languages = [detect_language(page.text) for page in pages]
    return ParsedDocument(
        page_count=len(pages),
        pages=tuple(pages),
        parse_confidence=document_confidence([page.assessment for page in pages]),
        language=aggregate_language(languages),
    )


def _canonical_text(page: Any) -> str:
    """The page's text, with paragraph boundaries preserved.

    PyMuPDF's plain `"text"` mode separates every *line* with a newline and
    gives no way to tell a line break inside a sentence from the gap between
    two clauses -- which would collapse a whole page into one undifferentiated
    block and lose the structure the chunker needs.

    So the canonical text is built from the layout blocks PyMuPDF already
    identified, joined by a blank line. This is deterministic (the same bytes
    always produce the same string), and it is the *only* definition of page
    text in the system: every `char_start`/`char_end` in the database indexes
    into exactly this.
    """
    blocks = page.get_text("blocks", sort=True)
    parts = [
        str(block[4]).strip()
        for block in blocks
        # index 6 is the block type; 1 means image, which carries no text.
        if len(block) > 6 and block[6] == 0 and str(block[4]).strip()
    ]
    return "\n\n".join(parts)


def _parse_page(
    page: Any,
    page_number: int,
    tables: list[list[str]],
    extraction_failed: bool,
) -> ParsedPage:
    text = _canonical_text(page)
    has_text_layer = bool(text.strip())

    metrics = PageMetrics(
        page_number=page_number,
        char_count=len(text.strip()),
        image_area_ratio=_image_area_ratio(page),
        has_text_layer=has_text_layer,
        garbled_unicode_ratio=_garbled_ratio(text),
        has_table_hint=bool(tables) or extraction_failed,
        table_extraction_failed=extraction_failed,
    )

    return ParsedPage(
        page_number=page_number,
        text=text,
        metrics=metrics,
        assessment=assess_page(metrics),
        parse_method=ParseMethod.PYMUPDF if has_text_layer else ParseMethod.NONE,
        tables=_locate_tables(text, tables),
    )


def _image_area_ratio(page: Any) -> float:
    """Fraction of the page covered by raster images.

    Images only -- vector drawings are ruling lines and logos, which do not
    indicate a scan. Overlapping images can sum past the page area, so the
    result is clamped.
    """
    page_rect = page.rect
    page_area = abs(page_rect.get_area())
    if page_area <= 0:
        return 0.0

    covered = 0.0
    for info in page.get_image_info():
        bbox = pymupdf.Rect(info["bbox"]) & page_rect
        if not bbox.is_empty:
            covered += abs(bbox.get_area())
    return float(min(covered / page_area, 1.0))


def _garbled_ratio(text: str) -> float:
    """Share of characters that carry no recoverable meaning."""
    if not text:
        return 0.0

    cid_chars = sum(len(match.group()) for match in _CID_PATTERN.finditer(text))
    damaged = sum(
        1
        for char in text
        if char == "�"
        or (unicodedata.category(char) in _GARBLED_CATEGORIES and char not in _ALLOWED_CONTROLS)
    )
    return min((cid_chars + damaged) / len(text), 1.0)


def _extract_tables(data: bytes) -> dict[int, tuple[list[list[str]], bool]]:
    """Per page: the cell text of each table, and whether extraction failed.

    "Failed" means pdfplumber found table *structure* it could not read cells
    out of -- the ruled-scan signature that routes a page to the VLM.
    """
    results: dict[int, tuple[list[list[str]], bool]] = {}
    try:
        pdf = pdfplumber.open(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 -- table detection is best-effort
        logger.warning("pdfplumber could not open the document", extra={"error": str(exc)})
        return results

    with pdf:
        for index, page in enumerate(pdf.pages):
            page_number = index + 1
            try:
                found = page.find_tables()
                cells = [_table_cells(table) for table in found]
                readable = [cell for cell in cells if cell]
                results[page_number] = (readable, bool(found) and not readable)
            except Exception as exc:  # noqa: BLE001 -- one bad page must not stop the parse
                logger.warning(
                    "table extraction failed",
                    extra={"page_number": page_number, "error": str(exc)},
                )
                results[page_number] = ([], True)
    return results


def _table_cells(table: Any) -> list[str]:
    """Flatten a table to its non-empty cell strings, in reading order."""
    try:
        rows = table.extract()
    except Exception as exc:  # noqa: BLE001 -- an unreadable table is a signal, not a crash
        logger.debug("table.extract failed", extra={"error": str(exc)})
        return []
    return [str(cell).strip() for row in rows for cell in row if cell and str(cell).strip()]


def _locate_tables(text: str, tables: list[list[str]]) -> tuple[TextSpan, ...]:
    """Anchor each table onto the canonical page text.

    Matching is token-wise (`\\s+` between tokens) because the two libraries
    disagree about intra-cell whitespace -- but the resulting offsets are into
    the unmodified text, so `text[start:end]` is still verbatim.
    """
    spans: list[TextSpan] = []
    cursor = 0

    for cells in tables:
        span = _locate_one_table(text, cells, cursor)
        if span is None:
            logger.debug("table could not be anchored to page text; dropped")
            continue
        spans.append(span)
        cursor = span.char_end

    return tuple(spans)


def _locate_one_table(text: str, cells: list[str], cursor: int) -> TextSpan | None:
    start: int | None = None
    for cell in cells:
        found = _find_tokens(text, cell, cursor)
        if found is not None:
            start = found[0]
            break
    if start is None:
        return None

    end: int | None = None
    for cell in reversed(cells):
        found = _find_tokens(text, cell, start)
        if found is not None:
            end = found[1]
            break
    if end is None or end <= start:
        return None

    return TextSpan(char_start=start, char_end=end)


def _find_tokens(text: str, needle: str, start: int) -> tuple[int, int] | None:
    tokens = needle.split()
    if not tokens:
        return None
    pattern = re.compile(r"\s+".join(re.escape(token) for token in tokens))
    match = pattern.search(text, start)
    return (match.start(), match.end()) if match else None


def page_language(page: ParsedPage) -> Language:
    """Convenience for callers that want a page-level language."""
    return detect_language(page.text)
