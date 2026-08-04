"""Chunking, and the span invariant everything downstream depends on.

The single most important assertion in this file is
`page.text[chunk.char_start:chunk.char_end] == chunk.text`. If it ever stops
holding, citation verification (CLAUDE.md 1.3) starts comparing quotes against
text that was never on the page, and every covenant sourced from that chunk
becomes unauditable while still looking fine.
"""

from __future__ import annotations

import itertools

from app.domain.enums import ChunkType, Language, ParseMethod
from app.domain.ingest import ChunkDraft, PageAssessment, PageMetrics, ParsedPage, TextSpan
from app.ingest.chunking import MAX_CHUNK_CHARS, chunk_hash, chunk_page

CLAUSE = (
    "The Issuer shall not, and shall procure that none of its subsidiaries shall, "
    "create or permit to subsist any security interest over the whole or any part "
    "of its present or future assets or revenues."
)


def make_page(text: str, *, tables: tuple[TextSpan, ...] = (), page_number: int = 1) -> ParsedPage:
    metrics = PageMetrics(
        page_number=page_number,
        char_count=len(text.strip()),
        image_area_ratio=0.0,
        has_text_layer=bool(text.strip()),
        garbled_unicode_ratio=0.0,
    )
    return ParsedPage(
        page_number=page_number,
        text=text,
        metrics=metrics,
        assessment=PageAssessment(needs_vlm=False, confidence=1.0),
        parse_method=ParseMethod.PYMUPDF,
        tables=tables,
    )


def assert_spans_are_exact(page: ParsedPage, chunks: list[ChunkDraft]) -> None:
    for chunk in chunks:
        assert page.text[chunk.char_start : chunk.char_end] == chunk.text
        assert chunk.char_start >= 0
        assert chunk.char_end <= len(page.text)
        assert chunk.page_number == page.page_number


def test_every_chunk_reproduces_its_text_from_its_span() -> None:
    page = make_page(f"NEGATIVE PLEDGE\n\n{CLAUSE}\n\nCROSS DEFAULT\n\n{CLAUSE}")

    chunks = chunk_page(page)

    assert chunks
    assert_spans_are_exact(page, chunks)


def test_a_heading_names_the_section_of_the_clause_beneath_it() -> None:
    page = make_page(f"NEGATIVE PLEDGE\n\n{CLAUSE}\n\nCROSS DEFAULT\n\n{CLAUSE}")

    titles = [chunk.section_title for chunk in chunk_page(page)]

    assert titles == ["NEGATIVE PLEDGE", "CROSS DEFAULT"]


def test_a_heading_never_merges_into_the_clause_above_it() -> None:
    page = make_page(f"NEGATIVE PLEDGE\n\n{CLAUSE}\n\nCROSS DEFAULT\n\n{CLAUSE}")

    chunks = chunk_page(page)

    assert len(chunks) == 2
    assert chunks[0].text.startswith("NEGATIVE PLEDGE")
    assert "CROSS DEFAULT" not in chunks[0].text


def test_chunks_are_ordered_and_do_not_overlap() -> None:
    page = make_page("\n\n".join([f"HEADING {i}\n\n{CLAUSE}" for i in range(5)]))

    chunks = chunk_page(page)

    for previous, current in itertools.pairwise(chunks):
        assert previous.char_end <= current.char_start
        assert previous.ordinal < current.ordinal


def test_ordinals_continue_from_the_offset_the_caller_supplies() -> None:
    page = make_page(f"HEADING\n\n{CLAUSE}")

    chunks = chunk_page(page, start_ordinal=7)

    assert chunks[0].ordinal == 7


def test_an_oversized_paragraph_is_split_at_sentence_boundaries() -> None:
    page = make_page(" ".join([CLAUSE] * 12))

    chunks = chunk_page(page)

    assert len(chunks) > 1
    assert all(len(chunk.text) <= MAX_CHUNK_CHARS for chunk in chunks)
    assert_spans_are_exact(page, chunks)
    # Sentence-bounded, so a split lands after a full stop rather than mid-word.
    assert all(chunk.text.rstrip().endswith(".") for chunk in chunks[:-1])


def test_a_sentence_longer_than_the_budget_is_still_bounded() -> None:
    page = make_page("x" * (MAX_CHUNK_CHARS * 2 + 50))

    chunks = chunk_page(page)

    assert all(len(chunk.text) <= MAX_CHUNK_CHARS for chunk in chunks)
    assert_spans_are_exact(page, chunks)
    # No text is dropped on the floor by the hard cut.
    assert sum(len(chunk.text) for chunk in chunks) == MAX_CHUNK_CHARS * 2 + 50


def test_a_table_becomes_its_own_typed_chunk() -> None:
    text = "CALL SCHEDULE\n\nSee below.\n\nCall Date\n2028-06-15\nCall Price\n102.00"
    table = TextSpan(char_start=text.index("Call Date"), char_end=len(text))
    page = make_page(text, tables=(table,))

    chunks = chunk_page(page)

    tables = [chunk for chunk in chunks if chunk.chunk_type is ChunkType.TABLE]
    assert len(tables) == 1
    assert tables[0].char_start == table.char_start
    assert tables[0].char_end == table.char_end
    # Sliced from the page, not re-rendered from cells -- so a quote from the
    # table can still be verified against it.
    assert tables[0].text == text[table.char_start : table.char_end]
    assert_spans_are_exact(page, chunks)


def test_prose_wrapping_a_table_survives_on_both_sides() -> None:
    text = f"{CLAUSE}\n\nCall Date 2028-06-15\n\n{CLAUSE}"
    start = text.index("Call Date")
    table = TextSpan(char_start=start, char_end=start + len("Call Date 2028-06-15"))
    page = make_page(text, tables=(table,))

    chunks = chunk_page(page)

    types = [chunk.chunk_type for chunk in chunks]
    assert types == [ChunkType.PARAGRAPH, ChunkType.TABLE, ChunkType.PARAGRAPH]
    assert_spans_are_exact(page, chunks)


def test_malay_chunks_carry_the_simple_configuration() -> None:
    page = make_page(
        "Penerbit hendaklah pada setiap masa mengekalkan nisbah gearan yang tidak "
        "melebihi 1.75 kali seperti yang dinyatakan dalam perjanjian ini."
    )

    chunk = chunk_page(page)[0]

    assert chunk.language is Language.MS
    assert chunk.fts_config == "simple"


def test_an_empty_page_yields_no_chunks() -> None:
    assert chunk_page(make_page("   \n\n  \n")) == []


def test_chunk_hash_is_stable_across_runs_but_distinguishes_repeated_text() -> None:
    first = chunk_hash(1, 0, 10, "boilerplate")
    same = chunk_hash(1, 0, 10, "boilerplate")
    # Identical boilerplate on another page is a different chunk with its own
    # provenance, so it must not collide.
    other_page = chunk_hash(2, 0, 10, "boilerplate")
    other_span = chunk_hash(1, 40, 50, "boilerplate")

    assert first == same
    assert len({first, other_page, other_span}) == 3
