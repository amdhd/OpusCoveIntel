"""PDF parsing against generated synthetic documents.

The fixtures are built in `tests/fixtures/synthetic_pdf.py` rather than checked
in (CLAUDE.md 7). They are the only PDFs in this repository, and they are the
reason the pathological paths -- a ruled table, an image-only page -- are
tested at all: nobody can legally commit a real scanned prospectus.
"""

from __future__ import annotations

import pytest

from app.domain.enums import Language, ParseMethod, VlmReason
from app.domain.ingest import ParsedDocument
from app.ingest.pdf import PdfParseError, parse_pdf
from tests.fixtures.synthetic_pdf import (
    build_mixed_document,
    build_prospectus,
    build_scanned_document,
)


@pytest.fixture(scope="module")
def prospectus() -> ParsedDocument:
    return parse_pdf(build_prospectus(), max_pages=600)


def test_all_pages_are_parsed(prospectus: ParsedDocument) -> None:
    assert prospectus.page_count == 4
    assert [page.page_number for page in prospectus.pages] == [1, 2, 3, 4]


def test_text_is_extracted_with_paragraph_structure(prospectus: ParsedDocument) -> None:
    covenants = prospectus.pages[1].text

    assert "NEGATIVE PLEDGE" in covenants
    assert "RM30,000,000" in covenants
    # Blank lines between blocks are what the chunker segments on; without them
    # a page collapses into one undifferentiated span.
    assert "\n\n" in covenants


def test_a_ruled_table_is_found_and_anchored_to_real_offsets(prospectus: ParsedDocument) -> None:
    page = prospectus.pages[2]

    assert page.metrics.has_table_hint is True
    assert len(page.tables) == 1

    table = page.tables[0]
    body = page.text[table.char_start : table.char_end]
    assert "2028-06-15" in body
    assert "Clean-up" in body


def test_a_text_layer_document_needs_no_vlm(prospectus: ParsedDocument) -> None:
    assert prospectus.pages_needing_vlm == ()
    assert prospectus.parse_confidence == 1.0
    assert all(page.parse_method is ParseMethod.PYMUPDF for page in prospectus.pages)


def test_a_document_mixing_english_and_malay_is_mixed(prospectus: ParsedDocument) -> None:
    assert prospectus.language is Language.MIXED


def test_a_scanned_page_is_flagged_with_every_reason_it_failed() -> None:
    parsed = parse_pdf(build_scanned_document(), max_pages=600)
    page = parsed.pages[0]

    assert page.assessment.needs_vlm is True
    assert page.parse_method is ParseMethod.NONE
    assert set(page.assessment.reasons) == {
        VlmReason.NO_TEXT_LAYER,
        VlmReason.LOW_CHAR_COUNT,
        VlmReason.HIGH_IMAGE_AREA,
    }
    assert page.metrics.image_area_ratio > 0.70


def test_only_the_scanned_page_of_a_mixed_document_is_flagged() -> None:
    parsed = parse_pdf(build_mixed_document(), max_pages=600)

    # The whole point of page-level scoring: one bad page must not send the
    # readable ones to a paid model (CLAUDE.md 4).
    assert [page.page_number for page in parsed.pages_needing_vlm] == [2]
    assert 0.0 < parsed.parse_confidence < 1.0


def test_a_document_over_the_page_limit_fails_loudly() -> None:
    with pytest.raises(PdfParseError, match="page limit"):
        parse_pdf(build_prospectus(), max_pages=2)


def test_bytes_that_are_not_a_pdf_fail_loudly() -> None:
    with pytest.raises(PdfParseError):
        parse_pdf(b"%PDF-1.7 but not really", max_pages=600)


def test_the_generated_fixture_is_byte_stable() -> None:
    # `documents.sha256` is the deduplication key, so a fixture that rebuilds
    # to different bytes would quietly defeat every idempotency check that
    # uses it -- including `make ingest-sample`.
    assert build_prospectus() == build_prospectus()


def test_parsing_is_deterministic() -> None:
    data = build_prospectus()

    first = parse_pdf(data, max_pages=600)
    second = parse_pdf(data, max_pages=600)

    # Re-ingesting unchanged bytes must produce identical spans, or the
    # idempotency guarantee in CLAUDE.md 1.7 is decorative.
    assert first == second
