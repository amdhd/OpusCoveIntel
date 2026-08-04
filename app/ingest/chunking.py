"""Chunking with real character spans.

Every chunk is a verbatim slice of its page's text: `page.text[char_start:
char_end] == chunk.text`, always. That equality is the foundation of the
citation chain (CLAUDE.md 1.2) and of citation verification (CLAUDE.md 1.3) --
a chunk whose text has been reflowed, de-hyphenated or re-rendered can never
have a model's quote matched against it honestly.

Consequences of holding that line:

* Merged blocks keep the whitespace *between* them, because a span is a slice,
  not a join.
* Tables are sliced from the canonical text, not re-rendered from cells.
* Splitting an oversized paragraph happens at sentence boundaries found in the
  original string, so the pieces still abut exactly.

Boundaries otherwise follow the document's own structure: blank lines separate
blocks, an upper-case single line is a heading and names the section for
everything under it.
"""

from __future__ import annotations

import hashlib
import re

from app.domain.enums import ChunkType
from app.domain.ingest import ChunkDraft, ParsedDocument, ParsedPage, TextSpan
from app.ingest.language import detect_language, fts_config_for

# Roughly 300 tokens: large enough to hold a whole covenant clause, small
# enough that a retrieved chunk is mostly signal. Candidate narrowing in
# Phase 6 sends these to Opus, so chunk size is also a cost lever.
MAX_CHUNK_CHARS = 1200

HEADING_MAX_CHARS = 120
_HEADING_UPPER_RATIO = 0.70

_BLOCK_SEPARATOR = re.compile(r"\n[ \t]*\n+")
# Split after sentence-ending punctuation, and after the ";" that separates
# limbs of a long covenant clause.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?;])\s+")


def chunk_document(
    document: ParsedDocument, *, max_chars: int = MAX_CHUNK_CHARS
) -> list[ChunkDraft]:
    """Chunk every page, numbering ordinals across the whole document."""
    drafts: list[ChunkDraft] = []
    for page in document.pages:
        drafts.extend(chunk_page(page, start_ordinal=len(drafts), max_chars=max_chars))
    return drafts


def chunk_page(
    page: ParsedPage, *, start_ordinal: int = 0, max_chars: int = MAX_CHUNK_CHARS
) -> list[ChunkDraft]:
    text = page.text
    items = _page_items(text, page.tables, max_chars)

    drafts: list[ChunkDraft] = []
    section_title: str | None = None
    ordinal = start_ordinal

    for spans, chunk_type in _group(items, max_chars):
        char_start = spans[0].char_start
        char_end = spans[-1].char_end
        chunk_text = text[char_start:char_end]
        if not chunk_text.strip():
            continue

        lead = text[spans[0].char_start : spans[0].char_end]
        if chunk_type is ChunkType.HEADING or _is_heading(lead):
            section_title = lead.strip()[:512]

        language = detect_language(chunk_text)
        drafts.append(
            ChunkDraft(
                page_number=page.page_number,
                ordinal=ordinal,
                text=chunk_text,
                char_start=char_start,
                char_end=char_end,
                chunk_type=chunk_type,
                language=language,
                fts_config=fts_config_for(language),
                section_title=section_title,
                hash=chunk_hash(page.page_number, char_start, char_end, chunk_text),
            )
        )
        ordinal += 1

    return drafts


def chunk_hash(page_number: int, char_start: int, char_end: int, text: str) -> str:
    """Content address for a chunk.

    Includes the span, not just the text: a document may repeat a boilerplate
    paragraph verbatim on several pages, and those are distinct chunks with
    distinct provenance. Stable across re-parses of the same bytes, which is
    what makes re-ingestion a no-op (CLAUDE.md 1.7).
    """
    payload = f"{page_number}|{char_start}|{char_end}|{text}".encode()
    return hashlib.sha256(payload).hexdigest()


# -- block segmentation ----------------------------------------------------


def _page_items(
    text: str, tables: tuple[TextSpan, ...], max_chars: int
) -> list[tuple[TextSpan, ChunkType]]:
    """Ordered spans covering the page: tables, headings and paragraphs."""
    items: list[tuple[TextSpan, ChunkType]] = [(span, ChunkType.TABLE) for span in tables]

    for block in _blocks(text):
        for residual in _subtract(block, tables):
            body = text[residual.char_start : residual.char_end]
            if not body.strip():
                continue
            if _is_heading(body):
                items.append((residual, ChunkType.HEADING))
                continue
            for piece in _split_long(text, residual, max_chars):
                items.append((piece, ChunkType.PARAGRAPH))

    items.sort(key=lambda item: (item[0].char_start, item[0].char_end))
    return items


def _blocks(text: str) -> list[TextSpan]:
    """Blank-line-separated blocks, as trimmed spans of `text`."""
    spans: list[TextSpan] = []
    cursor = 0
    for match in _BLOCK_SEPARATOR.finditer(text):
        spans.append(TextSpan(char_start=cursor, char_end=match.start()))
        cursor = match.end()
    spans.append(TextSpan(char_start=cursor, char_end=len(text)))

    trimmed = [_trim(text, span) for span in spans]
    return [span for span in trimmed if span.char_end > span.char_start]


def _trim(text: str, span: TextSpan) -> TextSpan:
    start, end = span.char_start, span.char_end
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return TextSpan(char_start=start, char_end=end)


def _subtract(block: TextSpan, tables: tuple[TextSpan, ...]) -> list[TextSpan]:
    """Remove table regions from a block, keeping whatever sits either side."""
    remaining = [block]
    for table in tables:
        nxt: list[TextSpan] = []
        for span in remaining:
            if not span.overlaps(table):
                nxt.append(span)
                continue
            if span.char_start < table.char_start:
                nxt.append(TextSpan(char_start=span.char_start, char_end=table.char_start))
            if table.char_end < span.char_end:
                nxt.append(TextSpan(char_start=table.char_end, char_end=span.char_end))
        remaining = nxt
    return remaining


def _split_long(text: str, span: TextSpan, max_chars: int) -> list[TextSpan]:
    """Split an oversized block at sentence boundaries, abutting exactly."""
    if span.char_end - span.char_start <= max_chars:
        return [span]

    body = text[span.char_start : span.char_end]
    cuts = [span.char_start + match.end() for match in _SENTENCE_BOUNDARY.finditer(body)]
    boundaries = [span.char_start, *cuts, span.char_end]

    pieces: list[TextSpan] = []
    start = span.char_start
    packed = span.char_start

    for boundary in boundaries[1:]:
        if boundary - start <= max_chars:
            packed = boundary
            continue
        # The next sentence overflows the budget: close the piece at the last
        # boundary that fitted.
        if packed > start:
            pieces.append(TextSpan(char_start=start, char_end=packed))
            start = packed
        # A single sentence longer than the whole budget still has to be cut. A
        # hard cut is worse prose but a truthful span, which is what matters.
        while boundary - start > max_chars:
            pieces.append(TextSpan(char_start=start, char_end=start + max_chars))
            start += max_chars
        packed = boundary

    if packed > start:
        pieces.append(TextSpan(char_start=start, char_end=packed))

    return [_trim(text, piece) for piece in pieces if piece.char_end > piece.char_start]


def _is_heading(body: str) -> bool:
    """A single short, mostly upper-case line -- how offering documents shout.

    Deliberately conservative: a false negative merely loses a section title,
    while a false positive fragments a clause into unretrievable pieces.
    """
    stripped = body.strip()
    if not stripped or "\n" in stripped or len(stripped) > HEADING_MAX_CHARS:
        return False
    if stripped.endswith((".", ",", ";")):
        return False
    letters = [char for char in stripped if char.isalpha()]
    if not letters:
        return False
    upper = sum(1 for char in letters if char.isupper())
    return upper / len(letters) >= _HEADING_UPPER_RATIO


# -- grouping --------------------------------------------------------------


def _group(
    items: list[tuple[TextSpan, ChunkType]], max_chars: int
) -> list[tuple[list[TextSpan], ChunkType]]:
    """Merge consecutive prose spans up to the budget.

    Tables are never merged into prose: their type is the signal that Phase 6
    should read them as tabular, and diluting that with surrounding narrative
    would lose it.
    """
    groups: list[tuple[list[TextSpan], ChunkType]] = []
    current: list[TextSpan] = []
    current_type = ChunkType.PARAGRAPH

    def flush() -> None:
        nonlocal current, current_type
        if current:
            groups.append((current, current_type))
        current = []
        current_type = ChunkType.PARAGRAPH

    for span, chunk_type in items:
        if chunk_type is ChunkType.TABLE:
            flush()
            groups.append(([span], ChunkType.TABLE))
            continue

        # A heading opens a new section, so it never joins the block above it.
        if chunk_type is ChunkType.HEADING and current:
            flush()

        span_end = span.char_end
        if current and span_end - current[0].char_start > max_chars:
            flush()

        current.append(span)
        if chunk_type is ChunkType.PARAGRAPH:
            current_type = ChunkType.PARAGRAPH
        elif not current[:-1]:
            current_type = ChunkType.HEADING

    flush()
    return groups
