"""The deterministic extractor and its citation checking.

Two things are being tested. That the patterns find what they claim to find,
in both languages -- and, more importantly, that what they produce is *citable*:
a span that reproduces its quote, and a threshold that is a `Decimal` rather
than a string that looks like one.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

from app.domain.enums import CallType, ClauseType, CovenantType, RatingAgency
from app.domain.extraction import RuleExtraction
from app.domain.rules import ComparisonOperator
from app.extract.citations import normalise, verify_quote
from app.extract.rule_extractor import extract, extract_call_schedule

COVENANT_PAGE = """NEGATIVE PLEDGE

The Issuer shall not, and shall procure that none of its subsidiaries shall, create or permit to \
subsist any security interest over the whole or any part of its present or future assets.

CROSS DEFAULT

An event of default shall occur if any indebtedness of the Issuer in an aggregate principal \
amount exceeding RM30,000,000 becomes due and payable prior to its stated maturity.

FINANCIAL COVENANTS

The Issuer shall maintain a consolidated gearing ratio of not more than 1.75 times, and a \
finance service cover ratio of not less than 1.50 times.

RATING TRIGGER

In the event the rating assigned to the sukuk by MARC is downgraded below BBB+, the Issuer \
shall notify the Trustee within five business days."""

MALAY_PAGE = (
    "Penerbit hendaklah pada setiap masa mengekalkan nisbah gearan yang tidak melebihi "
    "1.75 kali. Sekiranya berlaku ketidakpatuhan Shariah, ia adalah suatu kejadian "
    "pembubaran dan Penerbit hendaklah melaksanakan aku janji pembelian."
)


def by_type(text: str, clause_type: ClauseType) -> list[RuleExtraction]:
    return [item for item in extract(text) if item.clause_type is clause_type]


def test_every_extraction_span_reproduces_its_quote() -> None:
    for extraction in extract(COVENANT_PAGE):
        # Without this, the clause it becomes cites text that was never there.
        assert COVENANT_PAGE[extraction.char_start : extraction.char_end] == extraction.quote


def test_a_gearing_covenant_captures_value_and_direction() -> None:
    covenant = next(
        item
        for item in extract(COVENANT_PAGE)
        if item.terms and item.terms.covenant_type is CovenantType.GEARING_RATIO
    )

    assert covenant.terms is not None
    assert covenant.terms.threshold_ratio == Decimal("1.75")
    # "not more than" is a ceiling; reading it as a floor inverts every verdict.
    assert covenant.terms.operator is ComparisonOperator.LTE


def test_a_cover_ratio_is_captured_as_a_floor() -> None:
    covenant = next(
        item
        for item in extract(COVENANT_PAGE)
        if item.terms and item.terms.covenant_type is CovenantType.FINANCE_SERVICE_COVER
    )

    assert covenant.terms is not None
    assert covenant.terms.threshold_ratio == Decimal("1.50")
    assert covenant.terms.operator is ComparisonOperator.GTE


def test_a_cross_default_threshold_is_a_decimal() -> None:
    covenant = by_type(COVENANT_PAGE, ClauseType.CROSS_DEFAULT)[0]

    assert covenant.terms is not None
    assert covenant.terms.threshold_amount == Decimal("30000000")
    assert covenant.terms.threshold_currency == "MYR"


def test_a_rating_trigger_captures_the_notch_and_the_agency() -> None:
    trigger = by_type(COVENANT_PAGE, ClauseType.RATING_TRIGGER)[0]

    assert trigger.terms is not None
    assert trigger.terms.trigger_rating == "BBB+"
    assert trigger.terms.rating_agency is RatingAgency.MARC


def test_a_negative_pledge_is_detected_without_being_quantified() -> None:
    pledge = by_type(COVENANT_PAGE, ClauseType.NEGATIVE_PLEDGE)

    assert len(pledge) == 1
    assert pledge[0].terms is not None
    # Detected but not quantified: the rules engine will report
    # INSUFFICIENT_DATA rather than treat it as satisfied.
    assert pledge[0].terms.threshold_ratio is None
    assert pledge[0].terms.threshold_amount is None


def test_a_heading_match_is_widened_into_usable_evidence() -> None:
    cross_default = by_type(COVENANT_PAGE, ClauseType.CROSS_DEFAULT)[0]

    # "CROSS DEFAULT" alone is evidence of nothing.
    assert "CROSS DEFAULT" in cross_default.quote
    assert "RM30,000,000" in cross_default.quote


def test_the_same_covenant_is_not_extracted_twice_from_one_span() -> None:
    keys = [(item.clause_type, item.covenant_type) for item in extract(COVENANT_PAGE)]

    assert len(keys) == len(set(keys))


def test_two_covenants_in_one_sentence_both_survive() -> None:
    sentence = (
        "The Issuer shall maintain a gearing ratio of not more than 1.75 times and an "
        "interest cover ratio of not less than 3.00 times."
    )

    found = {item.covenant_type for item in extract(sentence)}

    # Both are `financial_covenant` clauses. Deduplicating on clause type alone
    # would silently drop one and the instrument would lose a covenant.
    assert found == {CovenantType.GEARING_RATIO, CovenantType.INTEREST_COVER}


def test_bahasa_malaysia_covenants_are_extracted() -> None:
    found = {item.clause_type for item in extract(MALAY_PAGE)}

    assert ClauseType.FINANCIAL_COVENANT in found
    # CLAUDE.md 6: Shariah non-compliance, dissolution and the purchase
    # undertaking are distinct linked entities, not one free-text field.
    assert ClauseType.SHARIAH_COMPLIANCE in found
    assert ClauseType.DISSOLUTION_EVENT in found
    assert ClauseType.PURCHASE_UNDERTAKING in found


def test_the_malay_gearing_covenant_matches_the_english_one() -> None:
    covenant = next(
        item
        for item in extract(MALAY_PAGE)
        if item.terms and item.terms.covenant_type is CovenantType.GEARING_RATIO
    )

    assert covenant.terms is not None
    assert covenant.terms.threshold_ratio == Decimal("1.75")
    # "tidak melebihi" is "not exceeding".
    assert covenant.terms.operator is ComparisonOperator.LTE


def test_prose_without_covenants_extracts_nothing() -> None:
    assert extract("The Trustee is Synthetic Trustees Berhad, incorporated in Malaysia.") == []


# -- call schedules --------------------------------------------------------


def test_a_call_schedule_table_yields_dates_prices_and_types() -> None:
    table = (
        "Call Date\nCall Price\nCall Type\n\n2028-06-15\n102.00\nOptional\n\n"
        "2030-06-15\n100.00\nClean-up"
    )

    rows = extract_call_schedule(table)

    assert [row[0] for row in rows] == [dt.date(2028, 6, 15), dt.date(2030, 6, 15)]
    assert rows[0][1] == Decimal("102.00")
    assert rows[1][2] is CallType.CLEAN_UP


def test_a_year_beside_a_number_is_not_mistaken_for_a_call_row() -> None:
    assert extract_call_schedule("Issued in 2028 at a profit rate of 4.85 per cent") == []


# -- citation verification -------------------------------------------------


def test_a_literal_quote_verifies_exactly() -> None:
    check = verify_quote("gearing ratio of not more than 1.75", COVENANT_PAGE)

    assert check.verified
    assert check.score == 1.0
    assert COVENANT_PAGE[check.char_start : check.char_end] == "gearing ratio of not more than 1.75"


def test_a_quote_differing_only_in_whitespace_still_verifies() -> None:
    check = verify_quote("gearing   ratio of\nnot more than 1.75", COVENANT_PAGE)

    assert check.verified
    assert check.method == "normalised"
    # Offsets index the original text, not the normalised copy.
    assert COVENANT_PAGE[check.char_start : check.char_end].startswith("gearing")


def test_a_quote_that_is_not_in_the_chunk_fails() -> None:
    # A fabricated quote that shares no substantial text with the chunk.
    check = verify_quote("the trustee may declare an event of default at any time", COVENANT_PAGE)

    # CLAUDE.md 1.3: this is what stops a fabricated citation being persisted.
    assert not check.verified
    assert check.score == 0.0
    assert check.char_start is None


def test_an_empty_quote_fails_rather_than_matching_everything() -> None:
    assert verify_quote("   ", COVENANT_PAGE).verified is False


def test_a_quote_with_a_dropped_footnote_verifies_fuzzy() -> None:
    """Phase 6: an LLM might omit a footnote marker that the PDF text includes."""
    chunk = (
        "The Issuer shall not create or permit to subsist any security "
        "interest[1] over its assets."
    )
    quote = (
        "The Issuer shall not create or permit to subsist any security "
        "interest over its assets"
    )

    check = verify_quote(quote, chunk)

    assert check.verified
    assert check.method == "fuzzy"
    assert check.score >= 0.92
    assert check.char_start is None  # fuzzy cannot determine exact offsets


def test_normalisation_folds_smart_quotes_and_dashes() -> None:
    smart = "the \u201cIssuer\u201d \u2013 as defined"

    assert normalise(smart) == 'the "Issuer" - as defined'
