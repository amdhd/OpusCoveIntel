"""Deterministic covenant evaluation.

This module is the load-bearing half of CLAUDE.md 1.1: **the LLM never computes
a breach.** Models turn prose into `CovenantTerms`; this turns `CovenantTerms`
plus `ObservedFacts` into a verdict, using arithmetic that can be unit-tested,
replayed and explained.

Three rules hold throughout:

1. **A missing fact is never compliance.** Every branch returns
   INSUFFICIENT_DATA rather than assuming, because "we have no gearing figure"
   and "gearing is fine" are opposite answers to a credit committee.
2. **Ratings compare ordinally.** `"AA-" > "A+"` is false as a string and true
   as a rating, so every rating test goes through `app/rules/ratings.py`.
3. **Money is Decimal.** Thresholds are compared exactly; no float ever touches
   a breach test.

Pure functions, no I/O -- `rules/` is a leaf and is held to `mypy --strict`.
"""

from __future__ import annotations

from decimal import Decimal

from app.domain.enums import CovenantType, Severity
from app.domain.rules import (
    ComparisonOperator,
    CovenantEvaluation,
    CovenantTerms,
    ObservedFacts,
    RuleStatus,
)
from app.rules.money import format_myr
from app.rules.ratings import UnknownRatingError, notches_between, rank

# Covenants whose compliant direction is "at least this much".
_MINIMUM_COVENANTS = frozenset(
    {
        CovenantType.INTEREST_COVER,
        CovenantType.FINANCE_SERVICE_COVER,
        CovenantType.MINIMUM_NET_WORTH,
    }
)


def evaluate(terms: CovenantTerms, facts: ObservedFacts) -> CovenantEvaluation:
    """Evaluate one covenant. Total over `CovenantType` -- every member is handled."""
    match terms.covenant_type:
        case CovenantType.GEARING_RATIO:
            return _ratio(terms, facts, facts.gearing_ratio, "gearing_ratio", "times")
        case CovenantType.INTEREST_COVER:
            return _ratio(terms, facts, facts.interest_cover, "interest_cover", "times")
        case CovenantType.FINANCE_SERVICE_COVER:
            return _ratio(
                terms, facts, facts.finance_service_cover, "finance_service_cover", "times"
            )
        case CovenantType.MINIMUM_NET_WORTH:
            return _amount(terms, facts, facts.net_worth, "net_worth")
        case CovenantType.CROSS_DEFAULT:
            return _cross_default(terms, facts)
        case CovenantType.RATING_TRIGGER:
            return _rating_trigger(terms, facts)
        case CovenantType.NEGATIVE_PLEDGE:
            return _negative_pledge(terms, facts)
        case CovenantType.CHANGE_OF_CONTROL:
            return _event(
                terms,
                facts.change_of_control,
                "change_of_control",
                breached="a change of control has occurred",
                clear="no change of control reported",
            )
        case CovenantType.DISPOSAL_RESTRICTION:
            return _cap(terms, facts, facts.disposals_value, "disposals_value", "disposals")
        case CovenantType.DISTRIBUTION_RESTRICTION:
            return _cap(
                terms, facts, facts.distributions_value, "distributions_value", "distributions"
            )
        case CovenantType.SHARIAH_NON_COMPLIANCE:
            return _shariah(terms, facts)
        case CovenantType.OTHER:
            # Deliberately not evaluated. A free-text obligation ("shall
            # maintain its corporate existence") has no deterministic test, and
            # inventing one would be worse than admitting the gap.
            return _verdict(
                terms,
                RuleStatus.NOT_APPLICABLE,
                explanation=(
                    "covenant has no machine-evaluable test; "
                    "route to human review rather than infer compliance"
                ),
            )


def evaluate_all(terms: list[CovenantTerms], facts: ObservedFacts) -> list[CovenantEvaluation]:
    """Evaluate a set of covenants against one issuer's facts."""
    return [evaluate(item, facts) for item in terms]


def breaches(evaluations: list[CovenantEvaluation]) -> list[CovenantEvaluation]:
    return [item for item in evaluations if item.is_breach]


# -- ratio and amount covenants -------------------------------------------


def _ratio(
    terms: CovenantTerms,
    facts: ObservedFacts,
    observed: Decimal | None,
    field: str,
    unit: str,
) -> CovenantEvaluation:
    if terms.threshold_ratio is None:
        return _verdict(terms, RuleStatus.INSUFFICIENT_DATA, explanation="no threshold extracted")
    if observed is None:
        return _verdict(
            terms,
            RuleStatus.INSUFFICIENT_DATA,
            threshold=terms.threshold_ratio,
            explanation=f"{field} not reported for {facts.as_of.isoformat()}",
        )

    operator = terms.operator or _default_operator(terms.covenant_type)
    compliant = _compare(observed, operator, terms.threshold_ratio)
    headroom = _headroom(observed, terms.threshold_ratio, operator)
    status = _status(compliant, headroom, terms)

    return _verdict(
        terms,
        status,
        observed=observed,
        threshold=terms.threshold_ratio,
        headroom=headroom,
        inputs=(field, "as_of"),
        explanation=(
            f"{field.replace('_', ' ')} of {_plain(observed)} {unit} "
            f"{'satisfies' if compliant else 'breaches'} the covenant "
            f"{_operator_phrase(operator)} {_plain(terms.threshold_ratio)} {unit} "
            f"as at {facts.as_of.isoformat()}"
        ),
    )


def _amount(
    terms: CovenantTerms,
    facts: ObservedFacts,
    observed: Decimal | None,
    field: str,
) -> CovenantEvaluation:
    if terms.threshold_amount is None:
        return _verdict(terms, RuleStatus.INSUFFICIENT_DATA, explanation="no threshold extracted")
    if observed is None:
        return _verdict(
            terms,
            RuleStatus.INSUFFICIENT_DATA,
            threshold=terms.threshold_amount,
            explanation=f"{field} not reported for {facts.as_of.isoformat()}",
        )
    if terms.threshold_currency and terms.threshold_currency != facts.currency:
        # Converting would require an FX rate we do not have, and guessing one
        # inside a breach test is exactly the kind of silent error this system
        # exists to avoid.
        return _verdict(
            terms,
            RuleStatus.INSUFFICIENT_DATA,
            threshold=terms.threshold_amount,
            explanation=(
                f"threshold is in {terms.threshold_currency} but facts are in "
                f"{facts.currency}; no conversion rate available"
            ),
        )

    operator = terms.operator or _default_operator(terms.covenant_type)
    compliant = _compare(observed, operator, terms.threshold_amount)
    headroom = _headroom(observed, terms.threshold_amount, operator)
    status = _status(compliant, headroom, terms)

    return _verdict(
        terms,
        status,
        observed=observed,
        threshold=terms.threshold_amount,
        headroom=headroom,
        inputs=(field, "as_of"),
        explanation=(
            f"{field.replace('_', ' ')} of {format_myr(observed)} "
            f"{'satisfies' if compliant else 'breaches'} the covenant "
            f"{_operator_phrase(operator)} {format_myr(terms.threshold_amount)} "
            f"as at {facts.as_of.isoformat()}"
        ),
    )


def _cap(
    terms: CovenantTerms,
    facts: ObservedFacts,
    observed: Decimal | None,
    field: str,
    noun: str,
) -> CovenantEvaluation:
    """A ceiling on activity: disposals, distributions."""
    if terms.threshold_amount is None:
        return _verdict(terms, RuleStatus.INSUFFICIENT_DATA, explanation="no cap extracted")
    if observed is None:
        return _verdict(
            terms,
            RuleStatus.INSUFFICIENT_DATA,
            threshold=terms.threshold_amount,
            explanation=f"{noun} not reported for {facts.as_of.isoformat()}",
        )

    compliant = observed <= terms.threshold_amount
    headroom = terms.threshold_amount - observed
    status = _status(compliant, headroom, terms)
    return _verdict(
        terms,
        status,
        observed=observed,
        threshold=terms.threshold_amount,
        headroom=headroom,
        inputs=(field, "as_of"),
        explanation=(
            f"{noun} of {format_myr(observed)} against a cap of "
            f"{format_myr(terms.threshold_amount)} as at {facts.as_of.isoformat()}"
        ),
    )


def _cross_default(terms: CovenantTerms, facts: ObservedFacts) -> CovenantEvaluation:
    """Cross-default: other debt accelerating above a threshold is the event.

    Note the direction. For a gearing covenant the observed value must stay
    *below* the threshold; here the threshold is the point at which acceleration
    elsewhere becomes an event of default, so reaching it is the breach.
    """
    if terms.threshold_amount is None:
        return _verdict(terms, RuleStatus.INSUFFICIENT_DATA, explanation="no threshold extracted")
    if facts.accelerated_indebtedness is None:
        return _verdict(
            terms,
            RuleStatus.INSUFFICIENT_DATA,
            threshold=terms.threshold_amount,
            explanation="no accelerated indebtedness reported",
        )

    triggered = facts.accelerated_indebtedness >= terms.threshold_amount
    headroom = terms.threshold_amount - facts.accelerated_indebtedness
    status = _status(not triggered, headroom, terms)
    return _verdict(
        terms,
        status,
        observed=facts.accelerated_indebtedness,
        threshold=terms.threshold_amount,
        headroom=headroom,
        inputs=("accelerated_indebtedness", "as_of"),
        explanation=(
            f"accelerated indebtedness of {format_myr(facts.accelerated_indebtedness)} "
            f"{'reaches' if triggered else 'stays below'} the cross-default threshold of "
            f"{format_myr(terms.threshold_amount)}"
        ),
    )


def _rating_trigger(terms: CovenantTerms, facts: ObservedFacts) -> CovenantEvaluation:
    if not terms.trigger_rating:
        return _verdict(terms, RuleStatus.INSUFFICIENT_DATA, explanation="no trigger rating")
    if not facts.current_rating:
        return _verdict(terms, RuleStatus.INSUFFICIENT_DATA, explanation="no current rating")

    agency = facts.rating_agency if facts.rating_agency else terms.rating_agency
    try:
        # Ordinal, never lexical: rank is a notch index where lower is better.
        distance = notches_between(facts.current_rating, terms.trigger_rating, agency)
        current_rank = rank(facts.current_rating, agency)
    except UnknownRatingError as exc:
        return _verdict(terms, RuleStatus.INSUFFICIENT_DATA, explanation=str(exc))

    triggered = distance > 0
    # One notch of remaining cushion is worth flagging: rating actions are
    # discrete, and the next one lands on the trigger.
    status = (
        RuleStatus.BREACH if triggered else (RuleStatus.AT_RISK if distance == 0 else RuleStatus.OK)
    )
    return _verdict(
        terms,
        status,
        observed=Decimal(current_rank),
        headroom=Decimal(-distance),
        severity=Severity.HIGH if triggered else terms.severity,
        inputs=("current_rating", "rating_agency"),
        explanation=(
            f"{facts.current_rating} is "
            f"{'below' if triggered else ('at' if distance == 0 else 'above')} "
            f"the trigger rating {terms.trigger_rating}"
            f"{f' by {abs(distance)} notch(es)' if distance else ''}"
        ),
    )


def _negative_pledge(terms: CovenantTerms, facts: ObservedFacts) -> CovenantEvaluation:
    if facts.security_created is None:
        return _verdict(
            terms,
            RuleStatus.INSUFFICIENT_DATA,
            explanation="no security-interest position reported",
        )
    if not facts.security_created:
        return _verdict(
            terms, RuleStatus.OK, inputs=("security_created",), explanation="no security created"
        )
    if facts.security_is_permitted:
        return _verdict(
            terms,
            RuleStatus.OK,
            inputs=("security_created", "security_is_permitted"),
            explanation="security created falls within the permitted-security carve-out",
        )
    return _verdict(
        terms,
        RuleStatus.BREACH,
        severity=Severity.HIGH,
        inputs=("security_created", "security_is_permitted"),
        explanation="security created over assets outside the permitted-security carve-out",
    )


def _shariah(terms: CovenantTerms, facts: ObservedFacts) -> CovenantEvaluation:
    """Shariah non-compliance is a dissolution event, not a covenant breach.

    CLAUDE.md 6 keeps the two concepts distinct. The verdict is still BREACH --
    it is the most severe outcome the engine has -- but the explanation names
    the consequence, because what follows is a purchase undertaking rather than
    a cure period.
    """
    if facts.shariah_compliant is None:
        return _verdict(
            terms, RuleStatus.INSUFFICIENT_DATA, explanation="no Shariah compliance status reported"
        )
    if facts.shariah_compliant:
        return _verdict(
            terms,
            RuleStatus.OK,
            inputs=("shariah_compliant",),
            explanation="structure reported as Shariah-compliant",
        )
    return _verdict(
        terms,
        RuleStatus.BREACH,
        severity=Severity.CRITICAL,
        inputs=("shariah_compliant",),
        explanation=(
            "Shariah non-compliance is a dissolution event; "
            "the purchase undertaking is expected to be exercised"
        ),
    )


def _event(
    terms: CovenantTerms,
    occurred: bool | None,
    field: str,
    *,
    breached: str,
    clear: str,
) -> CovenantEvaluation:
    if occurred is None:
        return _verdict(terms, RuleStatus.INSUFFICIENT_DATA, explanation=f"{field} not reported")
    return _verdict(
        terms,
        RuleStatus.BREACH if occurred else RuleStatus.OK,
        severity=Severity.HIGH if occurred else terms.severity,
        inputs=(field,),
        explanation=breached if occurred else clear,
    )


# -- helpers ---------------------------------------------------------------


def _default_operator(covenant_type: CovenantType) -> ComparisonOperator:
    return ComparisonOperator.GTE if covenant_type in _MINIMUM_COVENANTS else ComparisonOperator.LTE


def _compare(observed: Decimal, operator: ComparisonOperator, threshold: Decimal) -> bool:
    match operator:
        case ComparisonOperator.LTE:
            return observed <= threshold
        case ComparisonOperator.LT:
            return observed < threshold
        case ComparisonOperator.GTE:
            return observed >= threshold
        case ComparisonOperator.GT:
            return observed > threshold


def _operator_phrase(operator: ComparisonOperator) -> str:
    match operator:
        case ComparisonOperator.LTE:
            return "of not more than"
        case ComparisonOperator.LT:
            return "of less than"
        case ComparisonOperator.GTE:
            return "of not less than"
        case ComparisonOperator.GT:
            return "of more than"


def _headroom(observed: Decimal, threshold: Decimal, operator: ComparisonOperator) -> Decimal:
    """Signed distance from the limit: positive is compliant room to spare."""
    if operator in (ComparisonOperator.LTE, ComparisonOperator.LT):
        return threshold - observed
    return observed - threshold


def _status(compliant: bool, headroom: Decimal, terms: CovenantTerms) -> RuleStatus:
    if not compliant:
        return RuleStatus.BREACH
    limit = _margin_limit(terms)
    if limit is not None and headroom <= limit:
        return RuleStatus.AT_RISK
    return RuleStatus.OK


def _margin_limit(terms: CovenantTerms) -> Decimal | None:
    base = terms.threshold_ratio if terms.threshold_ratio is not None else terms.threshold_amount
    if base is None or terms.warning_margin <= 0:
        return None
    return abs(base) * terms.warning_margin


def _plain(value: Decimal) -> str:
    text = f"{value:f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _verdict(
    terms: CovenantTerms,
    status: RuleStatus,
    *,
    observed: Decimal | None = None,
    threshold: Decimal | None = None,
    headroom: Decimal | None = None,
    severity: Severity | None = None,
    inputs: tuple[str, ...] = (),
    explanation: str = "",
) -> CovenantEvaluation:
    return CovenantEvaluation(
        covenant_type=terms.covenant_type,
        status=status,
        severity=severity or terms.severity,
        observed=observed,
        threshold=threshold,
        headroom=headroom,
        explanation=explanation,
        inputs_used=inputs,
    )
