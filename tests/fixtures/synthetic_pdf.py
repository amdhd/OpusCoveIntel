"""Generate synthetic offering documents as PDFs.

Built with PyMuPDF rather than checked in, for three reasons: real prospectuses
are copyrighted (CLAUDE.md 7), a generated fixture is reviewable as code, and
only a generator can produce the pages that actually exercise the pipeline --
a ruled table pdfplumber must find, a page of Bahasa Malaysia that must index
under `simple`, and an image-only page with no text layer at all.

Every issuer here is invented. Structure and figures mirror `app/db/seed.py`
so ingestion and the seeded portfolio describe the same fictional universe.

Run directly to write a file:

    uv run python -m tests.fixtures.synthetic_pdf var/sample-prospectus.pdf
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import pymupdf

PAGE_WIDTH = 595.0
PAGE_HEIGHT = 842.0
MARGIN = 56.0
BODY_FONT = "helv"
BOLD_FONT = "hebo"

ISSUER = "Synthetic Green Energy Sdn Bhd"

_COVER_BLOCKS: list[tuple[str, str]] = [
    (BOLD_FONT, "PRINCIPAL TERMS AND CONDITIONS"),
    (
        BODY_FONT,
        f"This information memorandum relates to the issuance by {ISSUER} (the "
        "Issuer) of Islamic medium term notes of up to RM300,000,000 in nominal "
        "value under an Ijarah sukuk programme (the Sukuk Ijarah). The Sukuk "
        "Ijarah is rated A- by MARC and carries a tenure of ten years from the "
        "date of first issuance.",
    ),
    (BOLD_FONT, "PARTIES TO THE TRANSACTION"),
    (
        BODY_FONT,
        "The Issuer is a special purpose vehicle incorporated in Malaysia. The "
        "Trustee is Synthetic Trustees Berhad, acting for and on behalf of the "
        "holders of the Sukuk Ijarah. The Shariah adviser has confirmed that the "
        "structure is in compliance with Shariah principles as adopted by the "
        "Securities Commission Malaysia.",
    ),
]

_COVENANT_BLOCKS: list[tuple[str, str]] = [
    (BOLD_FONT, "NEGATIVE PLEDGE"),
    (
        BODY_FONT,
        "The Issuer shall not, and shall procure that none of its subsidiaries "
        "shall, create or permit to subsist any security interest over the whole "
        "or any part of its present or future assets or revenues, other than "
        "permitted security interests, without the prior written consent of the "
        "Trustee acting on the instructions of holders representing not less "
        "than 66 per cent of the nominal value of the outstanding Sukuk Ijarah.",
    ),
    (BOLD_FONT, "CROSS DEFAULT"),
    (
        BODY_FONT,
        "An event of default shall occur if any indebtedness of the Issuer or any "
        "of its material subsidiaries in an aggregate principal amount exceeding "
        "RM30,000,000 (or its equivalent in any other currency) becomes due and "
        "payable prior to its stated maturity, or is not paid when due within any "
        "applicable grace period.",
    ),
    (BOLD_FONT, "FINANCIAL COVENANTS"),
    (
        BODY_FONT,
        "The Issuer shall at all times maintain a consolidated gearing ratio of "
        "not more than 1.75 times, tested semi-annually on each financial "
        "half-year end by reference to the latest audited or unaudited "
        "consolidated financial statements. The Issuer shall further maintain a "
        "finance service cover ratio of not less than 1.50 times.",
    ),
    (BOLD_FONT, "RATING TRIGGER"),
    (
        BODY_FONT,
        "In the event the rating assigned to the Sukuk Ijarah by MARC is "
        "downgraded below BBB+, the Issuer shall notify the Trustee within five "
        "business days and shall procure additional security acceptable to the "
        "Trustee within thirty business days of such downgrade.",
    ),
]

# Bahasa Malaysia. Postgres has no Malay stemmer, so these chunks must land on
# the `simple` text-search configuration (CLAUDE.md 6).
_MALAY_BLOCKS: list[tuple[str, str]] = [
    (BOLD_FONT, "RINGKASAN TERBITAN"),
    (
        BODY_FONT,
        "Terbitan ini adalah bagi pembiayaan semula pinjaman sedia ada dan bagi "
        "tujuan modal kerja am syarikat. Penerbit hendaklah pada setiap masa "
        "mengekalkan nisbah gearan yang tidak melebihi 1.75 kali seperti yang "
        "dinyatakan dalam perjanjian ini, dan hendaklah memaklumkan kepada "
        "Pemegang Amanah sekiranya berlaku apa-apa kejadian keingkaran.",
    ),
    (
        BODY_FONT,
        "Sekiranya berlaku ketidakpatuhan Shariah, ia adalah suatu kejadian "
        "pembubaran dan Penerbit hendaklah melaksanakan aku janji pembelian pada "
        "harga yang ditetapkan dalam perjanjian tersebut, dengan syarat bahawa "
        "notis bertulis telah diberikan kepada Pemegang Amanah.",
    ),
]

_CALL_SCHEDULE: list[list[str]] = [
    ["Call Date", "Call Price", "Call Type"],
    ["2028-06-15", "102.00", "Optional"],
    ["2029-06-15", "101.00", "Optional"],
    ["2030-06-15", "100.00", "Clean-up"],
]


TRUSTEE_ISSUER = "Synthetic Infrastructure Holdings Berhad"
REIT_ISSUER = "Synthetic Retail REIT Berhad"

# A second issuer with *different* thresholds, so retrieval and extraction have
# to discriminate rather than pattern-match one document. Every covenant here
# differs numerically from the prospectus above -- an extractor that attaches
# RM30m to this issuer is wrong in a way a single-document corpus hides.
_TRUST_DEED_BLOCKS: list[tuple[str, str]] = [
    (BOLD_FONT, "TRUST DEED: PRINCIPAL COVENANTS"),
    (
        BODY_FONT,
        f"This trust deed is made between {TRUSTEE_ISSUER} (the Issuer) and "
        "Synthetic Trustees Berhad in respect of the RM500,000,000 Wakalah "
        "sukuk programme rated AA3 by RAM Rating Services Berhad.",
    ),
    (BOLD_FONT, "NEGATIVE PLEDGE"),
    (
        BODY_FONT,
        "The Issuer shall not create or permit to subsist any security interest "
        "over its concession assets save for security interests existing at the "
        "date of this deed and disclosed in schedule 3.",
    ),
    (BOLD_FONT, "CROSS DEFAULT"),
    (
        BODY_FONT,
        "It shall be an event of default if any financial indebtedness of the "
        "Issuer exceeding RM50,000,000 in aggregate is declared due and payable "
        "prior to its stated maturity by reason of an event of default.",
    ),
    (BOLD_FONT, "FINANCIAL COVENANTS"),
    (
        BODY_FONT,
        "The Issuer shall maintain an interest cover ratio of not less than 3.00 "
        "times measured on each testing date, and a consolidated gearing ratio "
        "of not more than 2.25 times. The Issuer shall further maintain a "
        "consolidated net worth of not less than RM500,000,000 at all times.",
    ),
    (BOLD_FONT, "CHANGE OF CONTROL"),
    (
        BODY_FONT,
        "A change in control of the Issuer without the prior written consent of "
        "the Trustee shall entitle the holders to require early redemption at "
        "par together with accrued but unpaid profit.",
    ),
]

_RATING_REPORT_BLOCKS: list[tuple[str, str]] = [
    (BOLD_FONT, "RATING RATIONALE"),
    (
        BODY_FONT,
        f"MARC has affirmed its BBB+ rating on the RM250,000,000 Musharakah sukuk "
        f"issued by {REIT_ISSUER}. The outlook is stable, reflecting the "
        "resilience of the trust's retail portfolio and its refinancing profile.",
    ),
    (BOLD_FONT, "RATING TRIGGER"),
    (
        BODY_FONT,
        "The transaction documents provide that where the rating is downgraded "
        "below BBB-, the Issuer shall procure additional security or, failing "
        "which, the trustee may declare the sukuk immediately due and payable.",
    ),
    (BOLD_FONT, "COVENANT HEADROOM"),
    (
        BODY_FONT,
        "The issuer is required to maintain a consolidated gearing ratio of not "
        "more than 2.50 times. MARC notes that headroom against this covenant "
        "narrowed following the acquisition completed in the period under review.",
    ),
]


def build_prospectus() -> bytes:
    """A four-page text-layer document: cover, covenants, table, Bahasa Malaysia."""
    document = pymupdf.open()
    _text_page(document, _COVER_BLOCKS)
    _text_page(document, _COVENANT_BLOCKS)
    _table_page(document)
    _text_page(document, _MALAY_BLOCKS)
    return _to_bytes(document)


def build_trust_deed() -> bytes:
    """A second issuer's trust deed, with thresholds that differ from every other."""
    document = pymupdf.open()
    _text_page(document, _TRUST_DEED_BLOCKS)
    return _to_bytes(document)


def build_rating_report() -> bytes:
    """A third issuer's rating report -- shorter, and rating-trigger heavy."""
    document = pymupdf.open()
    _text_page(document, _RATING_REPORT_BLOCKS)
    return _to_bytes(document)


CORPUS_BUILDERS: dict[str, Callable[[], bytes]] = {
    "prospectus.pdf": build_prospectus,
    "trust-deed.pdf": build_trust_deed,
    "rating-report.pdf": build_rating_report,
}


def write_corpus(directory: Path) -> list[Path]:
    """Write every document of the synthetic corpus into `directory`.

    `make eval` scores all three, so all three have to exist on disk before
    ingestion. Writing only the prospectus -- which is what this module did --
    left two thirds of the labelled corpus reported as "never ingested" and the
    extraction numbers computed over one document.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, builder in CORPUS_BUILDERS.items():
        destination = directory / filename
        destination.write_bytes(builder())
        written.append(destination)
    return written


def build_scanned_document() -> bytes:
    """One page that is a picture of a page: no text layer, near-total image cover.

    Trips `no_text_layer`, `low_char_count` and `high_image_area` at once, which
    is exactly the profile that must route to the VLM in Phase 5 -- and must be
    detected, but not acted on, in Phase 3.
    """
    document = pymupdf.open()
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_image(page.rect, pixmap=_grey_pixmap())
    return _to_bytes(document)


def build_mixed_document() -> bytes:
    """A readable page followed by a scanned one -- the common real-world shape."""
    document = pymupdf.open()
    _text_page(document, _COVENANT_BLOCKS)
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_image(page.rect, pixmap=_grey_pixmap())
    return _to_bytes(document)


# -- page builders ---------------------------------------------------------


def _text_page(document: pymupdf.Document, blocks: list[tuple[str, str]]) -> pymupdf.Page:
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    cursor = MARGIN + 20.0
    width = PAGE_WIDTH - 2 * MARGIN

    for font, body in blocks:
        size = 12.0 if font == BOLD_FONT else 10.0
        height = _text_height(body, width, size)
        rect = pymupdf.Rect(MARGIN, cursor, MARGIN + width, cursor + height)
        page.insert_textbox(rect, body, fontsize=size, fontname=font, align=0)
        # A blank line between blocks: the chunker segments on blank lines, and
        # a real document's leading gives the extractor the same signal.
        cursor += height + 22.0

    return page


def _table_page(document: pymupdf.Document) -> pymupdf.Page:
    page = document.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page.insert_textbox(
        pymupdf.Rect(MARGIN, MARGIN, PAGE_WIDTH - MARGIN, MARGIN + 40),
        "REDEMPTION AND CALL SCHEDULE",
        fontsize=12,
        fontname=BOLD_FONT,
    )
    page.insert_textbox(
        pymupdf.Rect(MARGIN, MARGIN + 44, PAGE_WIDTH - MARGIN, MARGIN + 110),
        "The Issuer may redeem the Sukuk Ijarah in whole or in part on any of the "
        "call dates set out below, at the corresponding call price expressed as a "
        "percentage of the nominal value.",
        fontsize=10,
        fontname=BODY_FONT,
    )

    # Ruled grid: pdfplumber's default strategy finds tables from stroked lines,
    # so the rules have to be real vector lines, not typographic spacing.
    top = MARGIN + 110
    row_height = 26.0
    col_width = (PAGE_WIDTH - 2 * MARGIN) / 3
    rows = len(_CALL_SCHEDULE)
    bottom = top + rows * row_height

    for index in range(rows + 1):
        y = top + index * row_height
        page.draw_line(pymupdf.Point(MARGIN, y), pymupdf.Point(PAGE_WIDTH - MARGIN, y))
    for index in range(4):
        x = MARGIN + index * col_width
        page.draw_line(pymupdf.Point(x, top), pymupdf.Point(x, bottom))

    for row_index, row in enumerate(_CALL_SCHEDULE):
        for col_index, cell in enumerate(row):
            page.insert_text(
                pymupdf.Point(
                    MARGIN + col_index * col_width + 8,
                    top + row_index * row_height + 17,
                ),
                cell,
                fontsize=10,
                fontname=BOLD_FONT if row_index == 0 else BODY_FONT,
            )

    return page


def _grey_pixmap() -> pymupdf.Pixmap:
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 620, 877))
    pixmap.set_rect(pixmap.irect, (208, 208, 208))
    return pixmap


def _text_height(body: str, width: float, size: float) -> float:
    """Rough textbox height. Generous, so nothing is silently clipped."""
    chars_per_line = max(int(width / (size * 0.5)), 1)
    lines = max(len(body) // chars_per_line + 1, 1)
    return lines * size * 1.35 + size


def _to_bytes(document: pymupdf.Document) -> bytes:
    """Serialise deterministically, so the same fixture always hashes the same.

    PyMuPDF stamps a fresh timestamp and a random `/ID` into every save. Left
    alone, that makes the fixture a *different document* on each build --
    `documents.sha256` is the deduplication key (CLAUDE.md 1.7), so a
    regenerated fixture would silently defeat every idempotency check that uses
    it, including `make ingest-sample`.
    """
    try:
        document.set_metadata(
            {
                "title": "Synthetic Information Memorandum",
                "author": "OpusCovIntel test fixtures",
                "producer": "OpusCovIntel",
                "creator": "OpusCovIntel",
                "creationDate": "D:20260101000000Z",
                "modDate": "D:20260101000000Z",
            }
        )
        return bytes(document.tobytes(garbage=4, deflate=True, no_new_id=True))
    finally:
        document.close()


def main(argv: list[str]) -> int:
    """Write one prospectus, or the whole corpus when handed a directory.

    A destination without a `.pdf` suffix means "put the corpus here" -- that is
    what `make corpus` passes, and it keeps one entry point for both the
    single-document demo and the three-document eval corpus.
    """
    destination = Path(argv[1] if len(argv) > 1 else "var/sample-prospectus.pdf")
    if destination.suffix.lower() != ".pdf":
        for path in write_corpus(destination):
            print(f"wrote {path}")
        return 0

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(build_prospectus())
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
