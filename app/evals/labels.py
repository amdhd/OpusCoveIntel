"""Ground truth for the synthetic corpus.

This is what a credit analyst would record after reading the three fixture
documents, written down before anything was measured. It is the reference the
extraction metrics score against, so two properties matter more than coverage:

**It is labelled from the document, not from the extractor.** Where the rule
extractor and the label disagree, the label is right by construction -- that is
the whole point. The prospectus states a 66 per cent consent level in its
negative pledge; that is a voting threshold, not a covenant threshold, so no
`threshold_ratio` is labelled and an extractor that reports one is wrong.
Likewise page 3's redemption language creates a *call schedule*, not a covenant,
so it is labelled under `CALL_SCHEDULE_LABELS` and a covenant found there is a
false positive.

**It joins on `sha256`, not on filename.** The same bytes arrive as
`prospectus.pdf` from the test fixtures and `sample-prospectus.pdf` from
`make ingest-sample`, and `documents.sha256` is the identity that survives both
(CLAUDE.md 1.7). `tests/test_evals_labels.py` asserts these hashes still match
what the builders produce, so a changed fixture fails loudly instead of quietly
scoring against labels that describe a document nobody has any more.

**Every number here comes from a synthetic document.** Real prospectuses are
copyrighted and none are in the corpus (CLAUDE.md 7), so an F1 computed against
this set says how the pipeline does on text that was written to be extractable.
It is a regression baseline, not a production accuracy estimate -- the report
writer says so on every run, and it will need re-baselining the day licensed
documents arrive (PLAN.md 9, open question 1).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from app.domain.enums import (
    CallType,
    ClauseType,
    CovenantType,
    Language,
    RatingAgency,
)
from app.domain.rules import ComparisonOperator

# sha256 of the deterministic fixture builders in tests/fixtures/synthetic_pdf.py.
# `_to_bytes` pins the PDF metadata and suppresses the random /ID precisely so
# these stay stable across rebuilds.
#
# **A PyMuPDF upgrade still moves them**, because the bytes it emits are its own
# business -- 1.28.0 -> 1.28.2 changed all three. That is not a silent failure:
# the labels join on sha256 (see the module docstring), so stale values score
# zero documents rather than the wrong ones, and `tests/test_eval_harness.py`
# asserts the builders still hash to these. Regenerate with:
#
#     python -c "import hashlib; from tests.fixtures.synthetic_pdf import \
#       build_prospectus as b; print(hashlib.sha256(b()).hexdigest())"
#
# They are hardcoded rather than computed because `app/` must not import from
# `tests/` -- the dependency would run the wrong way.
PROSPECTUS_SHA = "c08324a3442363bf1c6ccc377ccd1397e03eb4a04652581fe2f07142cb38aeed"
TRUST_DEED_SHA = "48fca71e6115fefb62d4f130a2191c523883f717b6b40069164c7bc257033383"
RATING_REPORT_SHA = "74d0b5f714ad4a7c7434884f7390de7b1af0dbb08a97c4d7096fafb8d0a44719"

# Human-readable names for the report. Not join keys -- see the module docstring.
DOCUMENT_NAMES: Final[dict[str, str]] = {
    PROSPECTUS_SHA: "prospectus",
    TRUST_DEED_SHA: "trust-deed",
    RATING_REPORT_SHA: "rating-report",
}


@dataclass(frozen=True)
class CovenantLabel:
    """One covenant that is genuinely in the document.

    `evidence` is a distinctive phrase from the clause it comes from. The
    citation metric checks that the extractor's own quote contains it, through
    `app.extract.citations.verify_quote` -- the same code the pipeline gates on
    (CLAUDE.md 1.3), so the harness cannot pass a citation the pipeline would
    reject or vice versa.

    A field left None means **the document does not state it**. That is a real
    assertion, not an absence of labelling: an extractor that fills it in is
    counted wrong, which is how a hallucinated threshold shows up as a number
    rather than as a shrug.
    """

    document_sha256: str
    covenant_type: CovenantType
    clause_type: ClauseType
    page_number: int
    evidence: str

    operator: ComparisonOperator | None = None
    threshold_amount: Decimal | None = None
    threshold_currency: str | None = None
    threshold_ratio: Decimal | None = None
    trigger_rating: str | None = None
    rating_agency: RatingAgency | None = None

    language: Language = Language.EN
    notes: str = ""


@dataclass(frozen=True)
class CallScheduleLabel:
    """One row of a redemption table, with the date the metric scores."""

    document_sha256: str
    call_date: dt.date
    call_price: Decimal
    call_type: CallType
    page_number: int


COVENANT_LABELS: Final[tuple[CovenantLabel, ...]] = (
    # -- prospectus, page 2: the covenant page ------------------------------
    CovenantLabel(
        document_sha256=PROSPECTUS_SHA,
        covenant_type=CovenantType.NEGATIVE_PLEDGE,
        clause_type=ClauseType.NEGATIVE_PLEDGE,
        page_number=2,
        evidence="create or permit to subsist any security interest",
        notes=(
            "The 66 per cent is the consent level for waiving the pledge, not a "
            "covenant threshold. Deliberately unlabelled."
        ),
    ),
    CovenantLabel(
        document_sha256=PROSPECTUS_SHA,
        covenant_type=CovenantType.CROSS_DEFAULT,
        clause_type=ClauseType.CROSS_DEFAULT,
        page_number=2,
        evidence="aggregate principal amount exceeding RM30,000,000",
        threshold_amount=Decimal("30000000"),
        threshold_currency="MYR",
        notes="The clause states an amount but no comparison direction.",
    ),
    CovenantLabel(
        document_sha256=PROSPECTUS_SHA,
        covenant_type=CovenantType.GEARING_RATIO,
        clause_type=ClauseType.FINANCIAL_COVENANT,
        page_number=2,
        evidence="consolidated gearing ratio of not more than 1.75 times",
        operator=ComparisonOperator.LTE,
        threshold_ratio=Decimal("1.75"),
    ),
    CovenantLabel(
        document_sha256=PROSPECTUS_SHA,
        covenant_type=CovenantType.FINANCE_SERVICE_COVER,
        clause_type=ClauseType.FINANCIAL_COVENANT,
        page_number=2,
        evidence="finance service cover ratio of not less than 1.50 times",
        operator=ComparisonOperator.GTE,
        threshold_ratio=Decimal("1.50"),
    ),
    CovenantLabel(
        document_sha256=PROSPECTUS_SHA,
        covenant_type=CovenantType.RATING_TRIGGER,
        clause_type=ClauseType.RATING_TRIGGER,
        page_number=2,
        evidence="downgraded below BBB+",
        trigger_rating="BBB+",
        rating_agency=RatingAgency.MARC,
        notes="The agency is named in the same sentence: 'by MARC is downgraded'.",
    ),
    # -- prospectus, page 4: the same universe, in Bahasa Malaysia ----------
    # Labelled, not skipped. The BM restatement is a real covenant statement,
    # and scoring it is what makes the English/Malay gap a number in the report
    # rather than an assumption (PLAN.md 9, open question 7).
    CovenantLabel(
        document_sha256=PROSPECTUS_SHA,
        covenant_type=CovenantType.GEARING_RATIO,
        clause_type=ClauseType.FINANCIAL_COVENANT,
        page_number=4,
        evidence="nisbah gearan yang tidak melebihi 1.75 kali",
        operator=ComparisonOperator.LTE,
        threshold_ratio=Decimal("1.75"),
        language=Language.MS,
    ),
    CovenantLabel(
        document_sha256=PROSPECTUS_SHA,
        covenant_type=CovenantType.SHARIAH_NON_COMPLIANCE,
        clause_type=ClauseType.SHARIAH_COMPLIANCE,
        page_number=4,
        evidence="ketidakpatuhan Shariah, ia adalah suatu kejadian pembubaran",
        language=Language.MS,
        notes=(
            "Dissolution event triggering a purchase undertaking -- linked "
            "concepts, not one free-text field (CLAUDE.md 6)."
        ),
    ),
    # -- trust deed: a second issuer, every threshold different -------------
    CovenantLabel(
        document_sha256=TRUST_DEED_SHA,
        covenant_type=CovenantType.NEGATIVE_PLEDGE,
        clause_type=ClauseType.NEGATIVE_PLEDGE,
        page_number=1,
        evidence="create or permit to subsist any security interest over its concession assets",
    ),
    CovenantLabel(
        document_sha256=TRUST_DEED_SHA,
        covenant_type=CovenantType.CROSS_DEFAULT,
        clause_type=ClauseType.CROSS_DEFAULT,
        page_number=1,
        evidence="financial indebtedness of the Issuer exceeding RM50,000,000",
        threshold_amount=Decimal("50000000"),
        threshold_currency="MYR",
    ),
    CovenantLabel(
        document_sha256=TRUST_DEED_SHA,
        covenant_type=CovenantType.INTEREST_COVER,
        clause_type=ClauseType.FINANCIAL_COVENANT,
        page_number=1,
        evidence="interest cover ratio of not less than 3.00 times",
        operator=ComparisonOperator.GTE,
        threshold_ratio=Decimal("3.00"),
    ),
    CovenantLabel(
        document_sha256=TRUST_DEED_SHA,
        covenant_type=CovenantType.GEARING_RATIO,
        clause_type=ClauseType.FINANCIAL_COVENANT,
        page_number=1,
        evidence="consolidated gearing ratio of not more than 2.25 times",
        operator=ComparisonOperator.LTE,
        threshold_ratio=Decimal("2.25"),
    ),
    CovenantLabel(
        document_sha256=TRUST_DEED_SHA,
        covenant_type=CovenantType.MINIMUM_NET_WORTH,
        clause_type=ClauseType.FINANCIAL_COVENANT,
        page_number=1,
        evidence="consolidated net worth of not less than RM500,000,000",
        operator=ComparisonOperator.GTE,
        threshold_amount=Decimal("500000000"),
        threshold_currency="MYR",
    ),
    CovenantLabel(
        document_sha256=TRUST_DEED_SHA,
        covenant_type=CovenantType.CHANGE_OF_CONTROL,
        clause_type=ClauseType.CHANGE_OF_CONTROL,
        page_number=1,
        evidence="A change in control of the Issuer without the prior written consent",
    ),
    # -- rating report: a third issuer -------------------------------------
    CovenantLabel(
        document_sha256=RATING_REPORT_SHA,
        covenant_type=CovenantType.RATING_TRIGGER,
        clause_type=ClauseType.RATING_TRIGGER,
        page_number=1,
        evidence="rating is downgraded below BBB-",
        trigger_rating="BBB-",
        rating_agency=RatingAgency.MARC,
        notes=(
            "The BBB+ elsewhere on this page is the current rating, not a "
            "trigger. An extractor that labels it one is wrong."
        ),
    ),
    CovenantLabel(
        document_sha256=RATING_REPORT_SHA,
        covenant_type=CovenantType.GEARING_RATIO,
        clause_type=ClauseType.FINANCIAL_COVENANT,
        page_number=1,
        evidence="consolidated gearing ratio of not more than 2.50 times",
        operator=ComparisonOperator.LTE,
        threshold_ratio=Decimal("2.50"),
    ),
)


CALL_SCHEDULE_LABELS: Final[tuple[CallScheduleLabel, ...]] = (
    CallScheduleLabel(
        document_sha256=PROSPECTUS_SHA,
        call_date=dt.date(2028, 6, 15),
        call_price=Decimal("102.00"),
        call_type=CallType.OPTIONAL,
        page_number=3,
    ),
    CallScheduleLabel(
        document_sha256=PROSPECTUS_SHA,
        call_date=dt.date(2029, 6, 15),
        call_price=Decimal("101.00"),
        call_type=CallType.OPTIONAL,
        page_number=3,
    ),
    CallScheduleLabel(
        document_sha256=PROSPECTUS_SHA,
        call_date=dt.date(2030, 6, 15),
        call_price=Decimal("100.00"),
        call_type=CallType.CLEAN_UP,
        page_number=3,
    ),
)


LABELLED_SHAS: Final[tuple[str, ...]] = (
    PROSPECTUS_SHA,
    TRUST_DEED_SHA,
    RATING_REPORT_SHA,
)


def covenant_labels_for(sha256: str) -> tuple[CovenantLabel, ...]:
    return tuple(label for label in COVENANT_LABELS if label.document_sha256 == sha256)


def call_schedule_labels_for(sha256: str) -> tuple[CallScheduleLabel, ...]:
    return tuple(label for label in CALL_SCHEDULE_LABELS if label.document_sha256 == sha256)


def document_name(sha256: str) -> str:
    """A short name for the report; the hash prefix for anything unlabelled."""
    return DOCUMENT_NAMES.get(sha256, sha256[:12])
