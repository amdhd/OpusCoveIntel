"""Display formatting for the read screens.

**These functions have a twin.** `frontend/src/app/format/` implements the same
rules for the Angular client, which receives raw JSON and has to render it
itself. Two implementations of one convention is a drift risk, so the convention
is written down once -- here -- and both suites assert the same table of cases
(`tests/test_web_format.py` and `format.spec.ts`, deliberately identical).
Change one, change both, or the same portfolio reads differently depending on
which screen someone opened.

The convention:

* Money is grouped in threes and never rounded. A trailing `.0000` off a
  Postgres `NUMERIC` is noise, but a real fractional sen is not, so zeros are
  trimmed rather than truncated to a fixed precision.
* Money never passes through a float, on either side. `HoldingRead` says why: a
  `Decimal` crosses the wire as a JSON string precisely so that a client cannot
  quietly turn RM300,000,000.05 into a double.
* An enum is a value, not a label. `financial_covenant` is what the database
  holds; `Financial covenant` is what a person reads. Acronyms survive the
  trip, because `Llm` is worse than `llm` was.
* Confidence is a percentage. `0.78` is a number an analyst has to convert
  before they can compare it to the 85% review bar; `78%` is one they can read
  against it at a glance.
* Anything that is not a bare decimal is returned verbatim. Review values are
  free text -- "RM30,000,000 or its equivalent" is a value someone typed, and a
  display helper does not get to reinterpret it as a number.
"""

from __future__ import annotations

import re
from decimal import Decimal

_BARE_DECIMAL = re.compile(r"^(-?)(\d+)(?:\.(\d*))?$")

EM_DASH = "—"


def money(value: Decimal | str | None, currency: str | None = None) -> str:
    """Group an exact amount, optionally prefixed with its currency code.

    >>> money(Decimal("300000000.0000"), "MYR")
    'MYR 300,000,000'
    >>> money("1234.5600")
    '1,234.56'
    """
    if value is None or value == "":
        return EM_DASH

    if isinstance(value, Decimal):
        # `:,f` rather than str(): a Decimal off the database can carry an
        # exponent, and "3E+8" is not a number anybody wants to read.
        amount = _trim(f"{value:,f}")
    else:
        match = _BARE_DECIMAL.match(value.strip())
        if match is None:
            return value.strip()
        sign, whole, fraction = match.group(1), match.group(2), match.group(3) or ""
        amount = _trim(f"{sign}{int(whole):,}" + (f".{fraction}" if fraction else ""))

    return f"{currency} {amount}" if currency else amount


# Words that are not words. Sentence-casing these produces `Llm`, `Spv`,
# `Isin` -- which reads as a typo rather than as an initialism.
_ACRONYMS = frozenset(
    {"llm", "vlm", "ocr", "spv", "isin", "nav", "id", "pdf", "fts", "sql", "api", "url", "vat"}
)


def label(value: object) -> str:
    """A controlled-vocabulary value, as a person would write it.

    >>> label("financial_covenant")
    'Financial covenant'
    >>> label("llm")
    'LLM'
    >>> label("rating_report")
    'Rating report'
    """
    if value is None:
        return EM_DASH
    # `StrEnum` members stringify to their value, which is what we want here.
    words = str(value).strip().replace("_", " ").split()
    if not words:
        return EM_DASH

    rendered = [w.upper() if w.lower() in _ACRONYMS else w for w in words]
    first = rendered[0]
    if first.lower() not in _ACRONYMS:
        # Sentence case, not title case: "Rating report", not "Rating Report".
        # Only the first word is lifted, and a word that is already capitalised
        # (a proper noun the data carries, like Ijarah) keeps its shape.
        first = first[0].upper() + first[1:]
    return " ".join([first, *rendered[1:]])


def confidence(value: float | None) -> str:
    """A model's confidence as a percentage, to the nearest point.

    Not two decimal places: the third significant figure of a model's
    self-reported confidence is not a real quantity, and printing it invites
    people to read 0.78 against 0.77 as though the difference meant something.
    """
    if value is None:
        return EM_DASH
    return f"{round(value * 100)}%"


def _trim(text: str) -> str:
    """Drop trailing zeros in the fraction, and the point if nothing survives."""
    if "." not in text:
        return text
    return text.rstrip("0").rstrip(".")
