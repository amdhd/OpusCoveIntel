"""Display formatting.

`CASES` is duplicated verbatim in `frontend/src/app/format/format.spec.ts`.
That duplication is the point: the two renderers show the same portfolio, and
the only way to notice they have drifted is to assert the same table on both
sides. If you add a case here and the client's suite still passes unchanged,
you have found the drift rather than avoided it.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.web.format import money

# (value, currency, expected)
CASES: list[tuple[str, str | None, str]] = [
    # The defect this landed for: a NUMERIC(20,4) straight off Postgres.
    ("300000000.0000", "MYR", "MYR 300,000,000"),
    ("500000000", None, "500,000,000"),
    # Real sen survive; zero sen do not.
    ("1234.5600", None, "1,234.56"),
    ("1234.0000", None, "1,234"),
    ("0.05", "MYR", "MYR 0.05"),
    ("0", "MYR", "MYR 0"),
    # Under four digits there is nothing to group.
    ("999", None, "999"),
    ("1000", None, "1,000"),
    ("-2500.50", None, "-2,500.5"),
    # Not a bare decimal: a review value someone typed. Left alone.
    ("RM30,000,000 or its equivalent", None, "RM30,000,000 or its equivalent"),
    ("n/a", None, "n/a"),
]


class TestMoney:
    @pytest.mark.parametrize(("value", "currency", "expected"), CASES)
    def test_a_case(self, value: str, currency: str | None, expected: str) -> None:
        assert money(value, currency) == expected

    @pytest.mark.parametrize(("value", "currency", "expected"), CASES)
    def test_a_decimal_formats_like_its_string(
        self, value: str, currency: str | None, expected: str
    ) -> None:
        """The server holds `Decimal`; the client holds the same value as text."""
        try:
            as_decimal = Decimal(value)
        except ArithmeticError:
            pytest.skip("not a decimal on this side")
        assert money(as_decimal, currency) == expected

    def test_an_exponent_is_not_shown_to_a_human(self) -> None:
        """`str(Decimal('3E+8'))` is '3E+8', which is not a ringgit figure."""
        assert money(Decimal("3E+8"), "MYR") == "MYR 300,000,000"

    def test_a_large_exact_amount_keeps_every_digit(self) -> None:
        """The whole reason Decimals cross the wire as strings."""
        assert money(Decimal("300000000.05")) == "300,000,000.05"

    @pytest.mark.parametrize("empty", [None, ""])
    def test_nothing_renders_as_a_dash(self, empty: str | None) -> None:
        assert money(empty) == "—"
