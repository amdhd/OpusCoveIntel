"""MYR parsing.

CLAUDE.md 6: money is `Decimal`, never `float`. These tests exist because a
covenant threshold that arrives as 30000000.000000001 is not a rounding
curiosity -- it is the number a breach test compares against.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.rules.money import find_money, format_myr, largest_money, parse_money

THIRTY_MILLION = Decimal("30000000")


@pytest.mark.parametrize(
    "text",
    [
        "RM30,000,000",
        "RM30 million",
        "RM30m",
        "RM 30 million",
        "MYR30,000,000",
        "MYR 30 million",
        "RM30 juta",
        "ringgit 30 million",
    ],
)
def test_the_same_threshold_written_eight_ways_parses_identically(text: str) -> None:
    money = parse_money(text)

    assert money is not None
    assert money.amount == THIRTY_MILLION
    assert money.currency == "MYR"


def test_amounts_are_decimal_not_float() -> None:
    money = parse_money("RM30.5 million")

    assert money is not None
    assert isinstance(money.amount, Decimal)
    # The float route gives 30500000.000000004; exactness is the whole point.
    assert money.amount == Decimal("30500000")


def test_bahasa_malaysia_scale_words() -> None:
    assert parse_money("RM1.2 bilion") == (Decimal("1200000000"), "MYR")
    assert parse_money("RM250 juta") == (Decimal("250000000"), "MYR")


def test_non_myr_currencies_are_preserved_not_converted() -> None:
    money = parse_money("USD10 million")

    assert money is not None
    assert money.currency == "USD"
    # Converting would need an FX rate the system does not have. The rules
    # engine refuses to compare across currencies rather than guess one.
    assert money.amount == Decimal("10000000")


def test_a_bare_number_is_not_money() -> None:
    # "within 14 days" must never be read as a threshold.
    assert parse_money("within 14 days of such downgrade") is None


def test_matches_carry_the_span_they_came_from() -> None:
    text = "exceeding RM30,000,000 (or its equivalent)"

    match = find_money(text)[0]

    assert text[match.char_start : match.char_end] == match.raw
    assert match.raw == "RM30,000,000"


def test_largest_wins_when_a_sentence_mixes_figures() -> None:
    clause = (
        "any indebtedness exceeding RM30,000,000 which is not paid within RM500 of the due amount"
    )

    assert largest_money(clause) == (THIRTY_MILLION, "MYR")


def test_largest_can_be_restricted_to_one_currency() -> None:
    clause = "RM30 million or USD40 million, whichever is higher"

    assert largest_money(clause, currency="MYR") == (THIRTY_MILLION, "MYR")


def test_every_amount_in_a_clause_is_found_in_order() -> None:
    amounts = [item.money.amount for item in find_money("RM5m, then RM20m, then RM1.5 billion")]

    assert amounts == [Decimal("5000000"), Decimal("20000000"), Decimal("1500000000")]


@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (Decimal("30000000"), "RM30 million"),
        (Decimal("1200000000"), "RM1.2 billion"),
        (Decimal("30500000"), "RM30.5 million"),
        (Decimal("1500"), "RM1,500"),
    ],
)
def test_formatting_reads_the_way_a_credit_note_would(amount: Decimal, expected: str) -> None:
    assert format_myr(amount) == expected
