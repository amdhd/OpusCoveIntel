"""The regex library behind the deterministic extractor.

docs/plan.md is blunt about the limits of this approach: regexes tuned on tidy text
"will not survive one real trust deed". They are here anyway, for the four
reasons in docs/plan.md 3 -- a free quality signal against the LLM, a fallback when
the budget guard trips, an A/B baseline, and a system that still answers
questions at $0 in CI and demos.

So the patterns are written to be *precise rather than greedy*. A pattern that
fires only on explicit covenant language and misses the paraphrase is useful; a
pattern that fires on anything containing "shall not" is worse than nothing,
because it manufactures citations that look verified.

Bilingual by necessity: Malaysian documents state the same covenant in English
and Bahasa Malaysia, sometimes on facing pages (CLAUDE.md 6).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

from app.domain.enums import ClauseType, CovenantType
from app.domain.rules import ComparisonOperator

# Rating notches as they appear in text, both agency notations.
RATING_TOKEN: Final[str] = (
    r"(?:AAA|AA[+-]?|A[+-]?|BBB[+-]?|BB[+-]?|B[+-]?|CCC[+-]?|CC|C|D)(?:[123])?"
)


@dataclass(frozen=True)
class Pattern:
    """One named regex and what a match means."""

    pattern_id: str
    regex: re.Pattern[str]
    clause_type: ClauseType
    covenant_type: CovenantType | None = None
    # Precision-weighted. A pattern that captures a number is a stronger claim
    # than one that merely recognises a heading, and confidence is what routes
    # a field to human review (CLAUDE.md 5).
    confidence: float = 0.75
    operator: ComparisonOperator | None = None


def _compile(source: str) -> re.Pattern[str]:
    return re.compile(source, re.IGNORECASE | re.VERBOSE)


# -- financial ratio covenants ---------------------------------------------
# "a consolidated gearing ratio of not more than 1.75 times"
# "a finance service cover ratio of not less than 1.50 times"

_RATIO_TAIL: Final[str] = r"""
    [^.;]{0,80}?
    of \s+ not \s+ (?P<direction> more | less ) \s+ than \s+
    (?P<value> \d+(?:\.\d+)? ) \s*
    (?: times | x \b )
"""

GEARING_RATIO = Pattern(
    pattern_id="gearing_ratio_en",
    regex=_compile(rf"(?P<subject> gearing \s+ ratio ) {_RATIO_TAIL}"),
    clause_type=ClauseType.FINANCIAL_COVENANT,
    covenant_type=CovenantType.GEARING_RATIO,
    confidence=0.90,
)

INTEREST_COVER = Pattern(
    pattern_id="interest_cover_en",
    regex=_compile(rf"(?P<subject> interest \s+ (?:cover|coverage) (?:\s+ratio)? ) {_RATIO_TAIL}"),
    clause_type=ClauseType.FINANCIAL_COVENANT,
    covenant_type=CovenantType.INTEREST_COVER,
    confidence=0.90,
)

FINANCE_SERVICE_COVER = Pattern(
    pattern_id="finance_service_cover_en",
    regex=_compile(
        r"(?P<subject> (?:finance|debt) \s+ service \s+ cover (?:age)? (?:\s+ratio)? )"
        + _RATIO_TAIL
    ),
    clause_type=ClauseType.FINANCIAL_COVENANT,
    covenant_type=CovenantType.FINANCE_SERVICE_COVER,
    confidence=0.90,
)

# Bahasa Malaysia: "nisbah gearan yang tidak melebihi 1.75 kali".
# "tidak melebihi" is "not exceeding"; "tidak kurang daripada" is "not less than".
GEARING_RATIO_MS = Pattern(
    pattern_id="gearing_ratio_ms",
    regex=_compile(
        r"""
        (?P<subject> nisbah \s+ gearan )
        [^.;]{0,60}?
        (?P<direction> tidak \s+ melebihi | tidak \s+ kurang \s+ daripada ) \s+
        (?P<value> \d+(?:\.\d+)? ) \s* kali
        """
    ),
    clause_type=ClauseType.FINANCIAL_COVENANT,
    covenant_type=CovenantType.GEARING_RATIO,
    confidence=0.85,
)

MINIMUM_NET_WORTH = Pattern(
    pattern_id="minimum_net_worth_en",
    regex=_compile(
        r"""
        (?P<subject> (?:consolidated \s+)? (?:net \s+ worth | shareholders' \s+ funds ) )
        [^.;]{0,60}?
        of \s+ not \s+ less \s+ than \s+
        (?P<amount> (?:RM|MYR) \s* [\d,]+(?:\.\d+)? (?:\s* (?:million|billion|juta|bilion|m|b))? )
        """
    ),
    clause_type=ClauseType.FINANCIAL_COVENANT,
    covenant_type=CovenantType.MINIMUM_NET_WORTH,
    confidence=0.88,
    operator=ComparisonOperator.GTE,
)

# -- event covenants -------------------------------------------------------

CROSS_DEFAULT = Pattern(
    pattern_id="cross_default_en",
    regex=_compile(
        r"""
        (?:
            cross \s+ default
          | indebtedness [^.;]{0,120}? becomes \s+ due \s+ and \s+ payable
              \s+ prior \s+ to \s+ its \s+ stated \s+ maturity
        )
        """
    ),
    clause_type=ClauseType.CROSS_DEFAULT,
    covenant_type=CovenantType.CROSS_DEFAULT,
    confidence=0.85,
)

NEGATIVE_PLEDGE = Pattern(
    pattern_id="negative_pledge_en",
    regex=_compile(
        r"""
        (?:
            negative \s+ pledge
          | shall \s+ not [^.;]{0,120}? create \s+ or \s+ permit \s+ to \s+ subsist
              [^.;]{0,60}? security \s+ interest
        )
        """
    ),
    clause_type=ClauseType.NEGATIVE_PLEDGE,
    covenant_type=CovenantType.NEGATIVE_PLEDGE,
    confidence=0.80,
)

CHANGE_OF_CONTROL = Pattern(
    pattern_id="change_of_control_en",
    regex=_compile(r"change \s+ (?:in|of) \s+ control"),
    clause_type=ClauseType.CHANGE_OF_CONTROL,
    covenant_type=CovenantType.CHANGE_OF_CONTROL,
    confidence=0.78,
)

RATING_TRIGGER = Pattern(
    pattern_id="rating_trigger_en",
    regex=_compile(
        rf"""
        rating [^.;]{{0,120}}?
        (?: is \s+ )?
        (?P<direction> downgraded | reduced | lowered | falls? )
        \s+ (?:to \s+ )? below \s+
        (?P<rating> {RATING_TOKEN} )
        """
    ),
    clause_type=ClauseType.RATING_TRIGGER,
    covenant_type=CovenantType.RATING_TRIGGER,
    confidence=0.88,
)

# Shariah non-compliance is a *dissolution event* triggering a purchase
# undertaking, not an ordinary covenant breach (CLAUDE.md 6). The clause type
# and covenant type differ deliberately.
SHARIAH_NON_COMPLIANCE = Pattern(
    pattern_id="shariah_non_compliance",
    regex=_compile(
        r"""
        (?:
            shariah \s+ non-? \s* compliance
          | non-? \s* compliance \s+ with \s+ shariah
          | ketidakpatuhan \s+ shariah
        )
        """
    ),
    clause_type=ClauseType.SHARIAH_COMPLIANCE,
    covenant_type=CovenantType.SHARIAH_NON_COMPLIANCE,
    confidence=0.85,
)

PURCHASE_UNDERTAKING = Pattern(
    pattern_id="purchase_undertaking",
    regex=_compile(r"(?: purchase \s+ undertaking | aku \s+ janji \s+ pembelian )"),
    clause_type=ClauseType.PURCHASE_UNDERTAKING,
    confidence=0.80,
)

DISSOLUTION_EVENT = Pattern(
    pattern_id="dissolution_event",
    regex=_compile(r"(?: dissolution \s+ event | kejadian \s+ pembubaran )"),
    clause_type=ClauseType.DISSOLUTION_EVENT,
    confidence=0.80,
)

CALL_OPTION = Pattern(
    pattern_id="call_option_en",
    regex=_compile(
        r"""
        (?:
            (?:may \s+ )? redeem [^.;]{0,80}?
              (?: call \s+ dates? | in \s+ whole \s+ or \s+ in \s+ part )
          | call \s+ (?:date|price|schedule)
          | redemption \s+ and \s+ call \s+ schedule
        )
        """
    ),
    clause_type=ClauseType.CALL_OPTION,
    confidence=0.75,
)

# A call schedule row: an ISO date beside a price expressed as a percentage of
# par. Restricted to plausible call prices (90-150) so a page number or a year
# cannot masquerade as one.
CALL_SCHEDULE_ROW: Final[re.Pattern[str]] = re.compile(
    r"(?P<date>\d{4}-\d{2}-\d{2})\s*\n?\s*"
    r"(?P<price>1[0-4]\d\.\d{2}|9\d\.\d{2}|150\.00)"
    r"(?:\s*\n?\s*(?P<call_type>Optional|Mandatory|Make-?whole|Clean-?up|Tax|Regulatory))?",
    re.IGNORECASE,
)

ALL_PATTERNS: Final[tuple[Pattern, ...]] = (
    GEARING_RATIO,
    GEARING_RATIO_MS,
    INTEREST_COVER,
    FINANCE_SERVICE_COVER,
    MINIMUM_NET_WORTH,
    CROSS_DEFAULT,
    NEGATIVE_PLEDGE,
    CHANGE_OF_CONTROL,
    RATING_TRIGGER,
    SHARIAH_NON_COMPLIANCE,
    PURCHASE_UNDERTAKING,
    DISSOLUTION_EVENT,
    CALL_OPTION,
)

# Direction words to comparison operators, both languages.
DIRECTION_OPERATORS: Final[dict[str, ComparisonOperator]] = {
    "more": ComparisonOperator.LTE,
    "less": ComparisonOperator.GTE,
    "tidak melebihi": ComparisonOperator.LTE,
    "tidak kurang daripada": ComparisonOperator.GTE,
}


def operator_for(direction: str) -> ComparisonOperator | None:
    """Map a captured direction word onto the compliant comparison."""
    key = re.sub(r"\s+", " ", direction.strip().lower())
    return DIRECTION_OPERATORS.get(key)
