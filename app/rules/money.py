"""Monetary amounts in Malaysian offering documents.

CLAUDE.md 6: money is `Decimal`, never `float`. A covenant threshold that
arrives as `30000000.000000001` is not a rounding curiosity -- it is the number
a breach test compares against.

The same threshold appears in a prospectus in half a dozen forms:

    RM30,000,000 · RM30 million · RM30m · RM30.5 mil · MYR 30 million
    RM30 juta · RM1.2 bilion          (Bahasa Malaysia)
    USD10 million                     (cross-border tranches)

All of them must normalise to the same `Decimal`, because
`covenants.threshold_amount` is what portfolio-level SQL compares. Matches carry
character offsets so the extractor can cite the span they came from
(CLAUDE.md 1.2).
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Final, NamedTuple

# Scale words, English and Bahasa Malaysia. `juta` is million, `bilion` is
# billion; `ribu` is thousand and appears in older documents.
_MULTIPLIERS: Final[dict[str, Decimal]] = {
    "k": Decimal(10) ** 3,
    "thousand": Decimal(10) ** 3,
    "ribu": Decimal(10) ** 3,
    "m": Decimal(10) ** 6,
    "mm": Decimal(10) ** 6,
    "mn": Decimal(10) ** 6,
    "mil": Decimal(10) ** 6,
    "million": Decimal(10) ** 6,
    "juta": Decimal(10) ** 6,
    "b": Decimal(10) ** 9,
    "bn": Decimal(10) ** 9,
    "billion": Decimal(10) ** 9,
    "bilion": Decimal(10) ** 9,
    "bil": Decimal(10) ** 9,
}

# Currency markers. "RM" and "ringgit" are MYR; ISO codes are taken as written.
_CURRENCY_BY_MARKER: Final[dict[str, str]] = {
    "rm": "MYR",
    "ringgit": "MYR",
    "myr": "MYR",
    "usd": "USD",
    "us$": "USD",
    "$": "USD",
    "sgd": "SGD",
    "eur": "EUR",
    "jpy": "JPY",
}

_CURRENCY_ALTERNATION: Final[str] = "|".join(
    sorted((re.escape(marker) for marker in _CURRENCY_BY_MARKER), key=len, reverse=True)
)
_MULTIPLIER_ALTERNATION: Final[str] = "|".join(
    sorted((re.escape(word) for word in _MULTIPLIERS), key=len, reverse=True)
)

_MONEY_RE: Final[re.Pattern[str]] = re.compile(
    rf"(?P<currency>{_CURRENCY_ALTERNATION})\s*"
    rf"(?P<number>\d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    rf"(?:\s*(?P<multiplier>{_MULTIPLIER_ALTERNATION})\b)?",
    re.IGNORECASE,
)

DEFAULT_CURRENCY: Final[str] = "MYR"


class Money(NamedTuple):
    amount: Decimal
    currency: str


class MoneyMatch(NamedTuple):
    """A parsed amount and the span of text it was read from."""

    money: Money
    char_start: int
    char_end: int
    raw: str


def find_money(text: str) -> list[MoneyMatch]:
    """Every monetary amount in `text`, in order, with its span."""
    matches: list[MoneyMatch] = []
    for match in _MONEY_RE.finditer(text):
        parsed = _to_decimal(match.group("number"), match.group("multiplier"))
        if parsed is None:
            continue
        currency = _CURRENCY_BY_MARKER[match.group("currency").lower()]
        matches.append(
            MoneyMatch(
                money=Money(amount=parsed, currency=currency),
                char_start=match.start(),
                char_end=match.end(),
                raw=match.group(0),
            )
        )
    return matches


def parse_money(text: str) -> Money | None:
    """The first monetary amount in `text`, or None."""
    found = find_money(text)
    return found[0].money if found else None


def largest_money(text: str, *, currency: str | None = None) -> Money | None:
    """The largest amount in `text`.

    Covenant prose often names a threshold beside smaller incidental figures
    ("RM30,000,000 (or its equivalent) ... within 14 days"); the threshold is
    reliably the largest, and 14 is not money at all because it carries no
    currency marker.
    """
    candidates = [
        item.money
        for item in find_money(text)
        if currency is None or item.money.currency == currency
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.amount)


def format_myr(amount: Decimal) -> str:
    """Render an amount the way a Malaysian credit note would."""
    if amount >= Decimal(10) ** 9:
        return f"RM{_trim(amount / Decimal(10) ** 9)} billion"
    if amount >= Decimal(10) ** 6:
        return f"RM{_trim(amount / Decimal(10) ** 6)} million"
    return f"RM{amount:,.2f}".rstrip("0").rstrip(".")


def _trim(value: Decimal) -> str:
    text = f"{value:,.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _to_decimal(number: str, multiplier: str | None) -> Decimal | None:
    try:
        value = Decimal(number.replace(",", ""))
    except InvalidOperation:
        return None
    if multiplier:
        value *= _MULTIPLIERS[multiplier.lower()]
    # Normalise away the exponent so 3E+7 and 30000000 compare equal as strings
    # in JSON, where thresholds are also stored.
    return value.quantize(Decimal(1)) if value == value.to_integral_value() else value
