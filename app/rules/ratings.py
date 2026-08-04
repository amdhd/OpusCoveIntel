"""Credit rating normalisation and ordinal comparison.

CLAUDE.md 6: rating comparison is ordinal, not lexical. `"AA-" > "A+"` is false
as a string and true as a rating, so every comparison goes through a rank.

Rank is a notch index where **lower is better** (AAA = 0). "Below A" therefore
means `rank > rank_of("A")`.

The two Malaysian agencies use different modifier notations for the same notches:

    MARC:  AAA  AA+  AA  AA-  A+  A  A-  BBB+ ...
    RAM:   AAA  AA1  AA2  AA3  A1  A2  A3  BBB1 ...

Both are parsed onto the same rank scale, so a portfolio query spanning MARC-
and RAM-rated holdings compares like with like. Scope note: this module lands in
Phase 2 (not Phase 4 with the rest of the rules engine) because
`rating_triggers.trigger_rank` cannot be populated without it.
"""

from __future__ import annotations

import re
from typing import Final

from app.domain.enums import RatingAgency

# Canonical notch order, best to worst. Index == rank.
_NOTCHES: Final[tuple[str, ...]] = (
    "AAA",
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",
    "BB+",
    "BB",
    "BB-",
    "B+",
    "B",
    "B-",
    "CCC+",
    "CCC",
    "CCC-",
    "CC",
    "C",
    "D",
)

RANK_BY_NOTCH: Final[dict[str, int]] = {n: i for i, n in enumerate(_NOTCHES)}
NOTCH_BY_RANK: Final[dict[int, str]] = dict(enumerate(_NOTCHES))

WORST_RANK: Final[int] = len(_NOTCHES) - 1

# RAM-style numeric modifiers map onto +/ flat/- .
_RAM_MODIFIER: Final[dict[str, str]] = {"1": "+", "2": "", "3": "-"}

# Malaysian national-scale ratings carry a suffix, e.g. AA1(s), AAA(bg), A+(cg).
# Strip decorations but keep the notch.
_SUFFIX_RE: Final[re.Pattern[str]] = re.compile(
    r"\s*\((?:s|bg|cg|fg|lg|p|m|id)\)\s*$", re.IGNORECASE
)
_TRAILING_OUTLOOK_RE: Final[re.Pattern[str]] = re.compile(
    r"\s+(?:stable|positive|negative|developing|watch(?:\s+\w+)?)\s*$", re.IGNORECASE
)


class UnknownRatingError(ValueError):
    """Raised when a rating string cannot be mapped onto the notch scale."""


def normalise(rating: str, agency: RatingAgency | None = None) -> str:
    """Return the canonical notch for a raw rating string.

    Handles RAM numeric modifiers, national-scale suffixes, outlook words,
    whitespace and case. Raises `UnknownRatingError` on anything unrecognised --
    silently coercing an unparseable rating to a rank would corrupt every
    downstream breach evaluation.
    """
    if not rating or not rating.strip():
        raise UnknownRatingError("empty rating")

    text = rating.strip().upper()
    text = _SUFFIX_RE.sub("", text)
    text = _TRAILING_OUTLOOK_RE.sub("", text)
    text = text.replace(" ", "")

    if text in RANK_BY_NOTCH:
        return text

    # RAM numeric modifier, e.g. AA1 -> AA+, A3 -> A-
    match = re.fullmatch(r"(AAA|AA|A|BBB|BB|B|CCC|CC|C)([123])", text)
    if match:
        base, digit = match.groups()
        candidate = f"{base}{_RAM_MODIFIER[digit]}"
        if candidate in RANK_BY_NOTCH:
            return candidate

    raise UnknownRatingError(f"unrecognised rating {rating!r} (agency={agency})")


def rank(rating: str, agency: RatingAgency | None = None) -> int:
    """Return the notch rank. Lower is better; AAA is 0."""
    return RANK_BY_NOTCH[normalise(rating, agency)]


def try_rank(rating: str | None, agency: RatingAgency | None = None) -> int | None:
    """Rank, or None when the input is missing or unparseable."""
    if rating is None:
        return None
    try:
        return rank(rating, agency)
    except UnknownRatingError:
        return None


def is_below(rating: str, threshold: str, agency: RatingAgency | None = None) -> bool:
    """True when `rating` is strictly worse than `threshold`.

    "Downgraded below A" is `is_below(current, "A")` -- A- is below A, A is not.
    """
    return rank(rating, agency) > rank(threshold, agency)


def is_at_or_below(rating: str, threshold: str, agency: RatingAgency | None = None) -> bool:
    return rank(rating, agency) >= rank(threshold, agency)


def notches_between(rating: str, threshold: str, agency: RatingAgency | None = None) -> int:
    """Signed notch distance: positive when `rating` is worse than `threshold`."""
    return rank(rating, agency) - rank(threshold, agency)


def is_investment_grade(rating: str, agency: RatingAgency | None = None) -> bool:
    """True at BBB- or better."""
    return rank(rating, agency) <= RANK_BY_NOTCH["BBB-"]
