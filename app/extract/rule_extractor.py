"""The deterministic extractor: chunk text in, cited extractions out.

Free, offline, and reproducible. It will not survive a real trust deed on its
own (PLAN.md is explicit about that), which is why Phase 6 runs Opus over the
same spans and treats disagreement as a review trigger at no extra model cost.

Two properties this must have to be worth anything:

* **Every extraction carries the span it came from**, so the clause it becomes
  can be cited (CLAUDE.md 1.2). A match without offsets is discarded.
* **A quantified match beats an unquantified one.** "gearing ratio of not more
  than 1.75 times" produces evaluable `CovenantTerms`; a bare mention of
  gearing produces a clause with no terms, which the rules engine will report
  as INSUFFICIENT_DATA rather than silently treat as compliant.
"""

from __future__ import annotations

import datetime as dt
import re
from decimal import Decimal, InvalidOperation
from typing import Final

from app.core.logging import get_logger
from app.domain.enums import CallType, CovenantType, RatingAgency, Severity
from app.domain.extraction import RuleExtraction
from app.domain.rules import CovenantTerms
from app.extract import patterns
from app.rules.money import largest_money
from app.rules.ratings import UnknownRatingError
from app.rules.ratings import normalise as normalise_rating

logger = get_logger(__name__)

# Part of the extraction identity (CLAUDE.md 1.7). Bumped to v2 when the rating
# agency started travelling with the trigger notch: the output of this module
# changed, and without a bump every already-extracted document would be skipped
# as "identity already satisfied" and keep the old, agency-less rows for ever.
EXTRACTOR_VERSION: Final[str] = "rules-v2"

# How far around a match to look for the number it refers to. A cross-default
# clause names its threshold in the same sentence; widening this to the chunk
# would start picking up the issue size from a neighbouring paragraph.
_CONTEXT_CHARS: Final[int] = 400

# A quote shorter than this is a heading, not evidence. See `_sentence_around`.
MIN_QUOTE_CHARS: Final[int] = 60

_AGENCY_TOKENS: Final[dict[str, RatingAgency]] = {
    "marc": RatingAgency.MARC,
    "ram": RatingAgency.RAM,
    "s&p": RatingAgency.SP,
    "standard & poor's": RatingAgency.SP,
    "moody's": RatingAgency.MOODYS,
    "fitch": RatingAgency.FITCH,
}


def extract(text: str) -> list[RuleExtraction]:
    """Every covenant this extractor can find in one chunk."""
    found: list[RuleExtraction] = []
    for pattern in patterns.ALL_PATTERNS:
        for match in pattern.regex.finditer(text):
            extraction = _build(pattern, match, text)
            if extraction is not None:
                found.append(extraction)

    deduped = _drop_overlapping(found)
    logger.debug(
        "rule extraction",
        extra={"candidates": len(found), "kept": len(deduped), "chars": len(text)},
    )
    return deduped


def extract_call_schedule(text: str) -> list[tuple[dt.date, Decimal, CallType, int, int]]:
    """Call dates, prices and types from a schedule, with their spans.

    Returned as tuples rather than `RuleExtraction` because a call schedule row
    becomes a `call_schedules` row, not a covenant -- but it still carries
    offsets, because it still needs a citation.
    """
    rows: list[tuple[dt.date, Decimal, CallType, int, int]] = []
    for match in patterns.CALL_SCHEDULE_ROW.finditer(text):
        try:
            call_date = dt.date.fromisoformat(match.group("date"))
            price = Decimal(match.group("price"))
        except (ValueError, InvalidOperation):
            continue
        rows.append(
            (
                call_date,
                price,
                _call_type(match.group("call_type")),
                match.start(),
                match.end(),
            )
        )
    return rows


# -- building an extraction ------------------------------------------------


def _build(pattern: patterns.Pattern, match: re.Match[str], text: str) -> RuleExtraction | None:
    quote = _sentence_around(text, match.start(), match.end())
    groups = match.groupdict()

    terms: CovenantTerms | None = None
    normalized: dict[str, str] = {}

    if pattern.covenant_type is not None:
        if "value" in groups and groups.get("value"):
            terms = _ratio_terms(pattern, groups)
        elif "amount" in groups and groups.get("amount"):
            terms = _amount_terms(pattern, groups["amount"])
        elif "rating" in groups and groups.get("rating"):
            terms = _rating_terms(pattern, groups["rating"], text, match.start())
        elif pattern.covenant_type is CovenantType.CROSS_DEFAULT:
            terms = _cross_default_terms(pattern, text, match)
        else:
            # Detected but not quantified: still a real clause, just not one the
            # rules engine can evaluate.
            terms = CovenantTerms(
                covenant_type=pattern.covenant_type,
                severity=_severity(pattern.covenant_type),
                description=pattern.pattern_id,
            )

    if terms is not None:
        if terms.threshold_ratio is not None:
            normalized["threshold_ratio"] = str(terms.threshold_ratio)
        if terms.threshold_amount is not None:
            normalized["threshold_amount"] = str(terms.threshold_amount)
            normalized["threshold_currency"] = terms.threshold_currency or ""
        if terms.trigger_rating:
            normalized["trigger_rating"] = terms.trigger_rating
            # The agency travels with the notch or the notch means nothing:
            # MARC's A- and RAM's AA3 are different scales (CLAUDE.md 6). It was
            # resolved here and then dropped on the way to `covenants`, so the
            # covenant row named a trigger rating no reader could place --
            # visible only in `rating_triggers`, which portfolio queries about
            # covenants do not read. UNKNOWN is left out rather than written: an
            # absent agency is a fact, "unknown" as a value is noise.
            if terms.rating_agency is not RatingAgency.UNKNOWN:
                normalized["rating_agency"] = terms.rating_agency.value
        if terms.operator:
            normalized["operator"] = terms.operator.value

    start, end = quote
    if end <= start:
        return None

    return RuleExtraction(
        clause_type=pattern.clause_type,
        covenant_type=pattern.covenant_type,
        quote=text[start:end],
        char_start=start,
        char_end=end,
        confidence=pattern.confidence,
        terms=terms,
        normalized=normalized,
        pattern_id=pattern.pattern_id,
    )


def _ratio_terms(pattern: patterns.Pattern, groups: dict[str, str | None]) -> CovenantTerms | None:
    raw = groups.get("value")
    direction = groups.get("direction") or ""
    if raw is None:
        return None
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None
    operator = pattern.operator or patterns.operator_for(direction)
    assert pattern.covenant_type is not None
    return CovenantTerms(
        covenant_type=pattern.covenant_type,
        operator=operator,
        threshold_ratio=value,
        severity=_severity(pattern.covenant_type),
        description=pattern.pattern_id,
    )


def _amount_terms(pattern: patterns.Pattern, raw: str) -> CovenantTerms | None:
    money = largest_money(raw)
    if money is None:
        return None
    assert pattern.covenant_type is not None
    return CovenantTerms(
        covenant_type=pattern.covenant_type,
        operator=pattern.operator,
        threshold_amount=money.amount,
        threshold_currency=money.currency,
        severity=_severity(pattern.covenant_type),
        description=pattern.pattern_id,
    )


def _cross_default_terms(
    pattern: patterns.Pattern, text: str, match: re.Match[str]
) -> CovenantTerms:
    """Cross-default threshold: the largest amount near the trigger language.

    "Largest" because the sentence habitually mixes the threshold with smaller
    incidental figures -- grace periods, percentages, notice days. Only figures
    carrying a currency marker are candidates at all.
    """
    window = text[max(0, match.start() - _CONTEXT_CHARS) : match.end() + _CONTEXT_CHARS]
    money = largest_money(window)
    assert pattern.covenant_type is not None
    return CovenantTerms(
        covenant_type=pattern.covenant_type,
        threshold_amount=money.amount if money else None,
        threshold_currency=money.currency if money else None,
        severity=_severity(pattern.covenant_type),
        description=pattern.pattern_id,
    )


def _rating_terms(
    pattern: patterns.Pattern, raw_rating: str, text: str, position: int
) -> CovenantTerms | None:
    try:
        notch = normalise_rating(raw_rating)
    except UnknownRatingError:
        # A rating we cannot place on the notch scale is not a trigger we can
        # evaluate. Coercing it would corrupt every downstream comparison.
        return None
    assert pattern.covenant_type is not None
    return CovenantTerms(
        covenant_type=pattern.covenant_type,
        trigger_rating=notch,
        rating_agency=_agency_near(text, position),
        severity=Severity.HIGH,
        description=pattern.pattern_id,
    )


def _agency_near(text: str, position: int) -> RatingAgency:
    window = text[max(0, position - _CONTEXT_CHARS) : position + _CONTEXT_CHARS].lower()
    for token, agency in _AGENCY_TOKENS.items():
        if token in window:
            return agency
    return RatingAgency.UNKNOWN


def _severity(covenant_type: CovenantType) -> Severity:
    match covenant_type:
        case CovenantType.SHARIAH_NON_COMPLIANCE:
            return Severity.CRITICAL
        case (
            CovenantType.CROSS_DEFAULT
            | CovenantType.RATING_TRIGGER
            | CovenantType.NEGATIVE_PLEDGE
            | CovenantType.CHANGE_OF_CONTROL
        ):
            return Severity.HIGH
        case _:
            return Severity.MEDIUM


def _call_type(raw: str | None) -> CallType:
    if not raw:
        return CallType.OPTIONAL
    key = raw.replace("-", "").replace(" ", "").lower()
    match key:
        case "mandatory":
            return CallType.MANDATORY
        case "makewhole":
            return CallType.MAKE_WHOLE
        case "cleanup":
            return CallType.CLEAN_UP
        case "tax":
            return CallType.TAX
        case "regulatory":
            return CallType.REGULATORY
        case _:
            return CallType.OPTIONAL


def _sentence_around(text: str, start: int, end: int) -> tuple[int, int]:
    """Widen a match to its sentence, so the quote reads as evidence.

    A citation of "gearing ratio of not more than 1.75" is technically accurate
    and useless to a reviewer, who needs to see what it was a covenant *of*.

    Headings get widened further. Several patterns match the heading rather than
    the operative sentence -- "CROSS DEFAULT" on its own line -- and a citation
    reading only "CROSS DEFAULT" is evidence of nothing. Pulling in the
    following paragraph also makes the heading match overlap the body match, so
    the two collapse into one extraction instead of two.
    """
    left = max(text.rfind(". ", 0, start), text.rfind("\n\n", 0, start))
    left = 0 if left < 0 else left + 2

    right = _sentence_end(text, end)
    if right - left < MIN_QUOTE_CHARS:
        right = _sentence_end(text, right)
    return left, min(right, len(text))


def _sentence_end(text: str, position: int) -> int:
    candidates = [
        found
        for found in (
            text.find(". ", position),
            text.find("\n\n", position),
            text.find(".\n", position),
        )
        if found >= 0
    ]
    return min(candidates) + 1 if candidates else len(text)


def _drop_overlapping(found: list[RuleExtraction]) -> list[RuleExtraction]:
    """Keep the most confident extraction per (clause type, covenant type, span).

    Several patterns legitimately fire on one sentence, and they are not
    duplicates: a Shariah clause also mentions a purchase undertaking, and a
    single sentence routinely carries two financial covenants --

        "a gearing ratio of not more than 1.75 times, and a finance service
         cover ratio of not less than 1.50 times"

    Both are `financial_covenant`, so keying on clause type alone silently
    drops one of them and the instrument loses a covenant. The covenant type is
    what distinguishes them.
    """
    ordered = sorted(found, key=lambda item: (-item.confidence, item.char_start))
    kept: list[RuleExtraction] = []
    for item in ordered:
        clash = any(
            other.clause_type is item.clause_type
            and other.covenant_type is item.covenant_type
            and item.char_start < other.char_end
            and other.char_start < item.char_end
            for other in kept
        )
        if not clash:
            kept.append(item)
    return sorted(kept, key=lambda item: (item.char_start, item.clause_type.value))
