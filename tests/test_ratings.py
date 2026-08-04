"""Rating normalisation and ordinal comparison tests.

This module is where the flagship query lives or dies: "which holdings trip on
a downgrade below A" is wrong unless `AA-` ranks better than `A+`, and unless
RAM's `AA3` lands on the same notch as MARC's `AA-`.
"""

from __future__ import annotations

import pytest

from app.domain.enums import RatingAgency
from app.rules.ratings import (
    UnknownRatingError,
    is_at_or_below,
    is_below,
    is_investment_grade,
    normalise,
    notches_between,
    rank,
    try_rank,
)


class TestNormalise:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("AAA", "AAA"),
            ("aa+", "AA+"),
            ("  A-  ", "A-"),
            ("BBB", "BBB"),
        ],
    )
    def test_marc_notation(self, raw: str, expected: str) -> None:
        assert normalise(raw) == expected

    @pytest.mark.parametrize(
        ("ram", "marc_equivalent"),
        [
            ("AA1", "AA+"),
            ("AA2", "AA"),
            ("AA3", "AA-"),
            ("A1", "A+"),
            ("A2", "A"),
            ("A3", "A-"),
            ("BBB1", "BBB+"),
            ("BBB3", "BBB-"),
        ],
    )
    def test_ram_numeric_modifiers_map_onto_the_same_notches(
        self, ram: str, marc_equivalent: str
    ) -> None:
        """RAM says AA3 where MARC says AA-. A portfolio spans both agencies."""
        assert normalise(ram) == marc_equivalent
        assert rank(ram) == rank(marc_equivalent)

    @pytest.mark.parametrize("raw", ["AA1(s)", "AAA (bg)", "A+(cg)", "AA-  stable", "BBB positive"])
    def test_national_scale_suffixes_and_outlooks_are_stripped(self, raw: str) -> None:
        assert normalise(raw) in {"AA+", "AAA", "A+", "AA-", "BBB"}

    @pytest.mark.parametrize("raw", ["", "   ", "NR", "banana", "AA4", "A5", "not-a-rating"])
    def test_unparseable_ratings_raise(self, raw: str) -> None:
        """Coercing an unknown rating to a rank would corrupt breach evaluation."""
        with pytest.raises(UnknownRatingError):
            normalise(raw)


class TestOrdering:
    def test_lexical_comparison_would_be_wrong(self) -> None:
        """The bug this whole module exists to prevent.

        `A-` sorts before `AA+` as a string ('-' is 0x2D, 'A' is 0x41), which
        would rank a single-A credit *better* than a double-A one. Likewise
        `BB+` sorts before `BBB`, inverting those two.
        """
        assert "A-" < "AA+"  # string order says A- is "smaller", i.e. better
        assert rank("A-") > rank("AA+")  # ordinal order: A- is materially worse
        assert is_below("A-", "AA+")

        assert "BB+" < "BBB"
        assert rank("BB+") > rank("BBB")

    def test_sorting_lexically_produces_a_different_order_than_by_rank(self) -> None:
        ratings = ["A-", "AA+", "BBB", "BB+", "AAA"]
        assert sorted(ratings) != sorted(ratings, key=rank)
        assert sorted(ratings, key=rank) == ["AAA", "AA+", "A-", "BBB", "BB+"]

    def test_full_scale_is_monotonic(self) -> None:
        scale = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "D"]
        ranks = [rank(r) for r in scale]
        assert ranks == sorted(ranks)
        assert len(set(ranks)) == len(ranks)

    @pytest.mark.parametrize(
        ("rating", "threshold", "expected"),
        [
            ("A-", "A", True),
            ("BBB+", "A", True),
            ("A", "A", False),  # "below A" excludes A itself
            ("A+", "A", False),
            ("AAA", "A", False),
        ],
    )
    def test_is_below(self, rating: str, threshold: str, expected: bool) -> None:
        assert is_below(rating, threshold) is expected

    def test_is_at_or_below_includes_the_threshold(self) -> None:
        assert is_at_or_below("A", "A")
        assert not is_below("A", "A")

    def test_notches_between_is_signed(self) -> None:
        assert notches_between("A-", "A") == 1  # one notch worse
        assert notches_between("A+", "A") == -1  # one notch better
        assert notches_between("A", "A") == 0

    def test_ram_and_marc_compare_correctly_against_each_other(self) -> None:
        """A RAM-rated AA3 holding must not look worse than a MARC-rated A+."""
        assert is_below("A+", "AA3")
        assert not is_below("AA3", "A+")


class TestHelpers:
    def test_try_rank_returns_none_instead_of_raising(self) -> None:
        assert try_rank(None) is None
        assert try_rank("not-a-rating") is None
        assert try_rank("AA-") == rank("AA-")

    @pytest.mark.parametrize(
        ("rating", "expected"),
        [("AAA", True), ("BBB-", True), ("BB+", False), ("D", False)],
    )
    def test_investment_grade_boundary_is_bbb_minus(self, rating: str, expected: bool) -> None:
        assert is_investment_grade(rating) is expected

    def test_agency_argument_is_accepted_but_not_required(self) -> None:
        assert rank("AA3", RatingAgency.RAM) == rank("AA3")
