"""The rules engine.

PLAN.md, Phase 4 acceptance: **unit tests cover every covenant type.** The
completeness test at the bottom enforces that mechanically -- adding a new
`CovenantType` without a test for it fails the suite rather than shipping a
covenant the engine silently mishandles.

The behaviour under test is CLAUDE.md 1.1: the LLM never computes a breach.
Everything here is arithmetic and ordinal comparison, with one rule running
through all of it -- **a missing fact is never compliance.**
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest

from app.domain.enums import CovenantType, RatingAgency, Severity
from app.domain.rules import (
    ComparisonOperator,
    CovenantTerms,
    ObservedFacts,
    RuleStatus,
)
from app.rules.covenants import breaches, evaluate, evaluate_all

AS_OF = dt.date(2026, 6, 30)


def facts(**overrides: object) -> ObservedFacts:
    return ObservedFacts(as_of=AS_OF, **overrides)  # type: ignore[arg-type]


def terms(covenant_type: CovenantType, **overrides: object) -> CovenantTerms:
    return CovenantTerms(covenant_type=covenant_type, **overrides)  # type: ignore[arg-type]


# -- ratio covenants -------------------------------------------------------


def test_gearing_within_the_limit_is_compliant() -> None:
    result = evaluate(
        terms(CovenantType.GEARING_RATIO, threshold_ratio=Decimal("1.75")),
        facts(gearing_ratio=Decimal("1.20")),
    )

    assert result.status is RuleStatus.OK
    assert result.headroom == Decimal("0.55")
    assert "1.75" in result.explanation


def test_gearing_above_the_limit_is_a_breach() -> None:
    result = evaluate(
        terms(CovenantType.GEARING_RATIO, threshold_ratio=Decimal("1.75")),
        facts(gearing_ratio=Decimal("1.90")),
    )

    assert result.is_breach
    assert result.headroom == Decimal("-0.15")


def test_gearing_exactly_at_the_limit_satisfies_not_more_than() -> None:
    result = evaluate(
        terms(CovenantType.GEARING_RATIO, threshold_ratio=Decimal("1.75")),
        facts(gearing_ratio=Decimal("1.75")),
    )

    # "not more than 1.75" includes 1.75. Off-by-one here is a false default.
    assert result.status is not RuleStatus.BREACH


def test_thin_headroom_is_reported_as_at_risk_rather_than_ok() -> None:
    result = evaluate(
        terms(CovenantType.GEARING_RATIO, threshold_ratio=Decimal("1.75")),
        facts(gearing_ratio=Decimal("1.70")),
    )

    # A breach is already too late to be useful to a credit analyst.
    assert result.status is RuleStatus.AT_RISK
    assert result.needs_attention


def test_interest_cover_is_a_minimum_not_a_maximum() -> None:
    compliant = evaluate(
        terms(CovenantType.INTEREST_COVER, threshold_ratio=Decimal("3.00")),
        facts(interest_cover=Decimal("4.50")),
    )
    breached = evaluate(
        terms(CovenantType.INTEREST_COVER, threshold_ratio=Decimal("3.00")),
        facts(interest_cover=Decimal("2.10")),
    )

    assert compliant.status is RuleStatus.OK
    assert breached.is_breach


def test_finance_service_cover_is_a_minimum() -> None:
    result = evaluate(
        terms(CovenantType.FINANCE_SERVICE_COVER, threshold_ratio=Decimal("1.50")),
        facts(finance_service_cover=Decimal("1.10")),
    )

    assert result.is_breach


def test_an_explicit_operator_overrides_the_default_direction() -> None:
    result = evaluate(
        terms(
            CovenantType.GEARING_RATIO,
            threshold_ratio=Decimal("1.75"),
            operator=ComparisonOperator.LT,
        ),
        facts(gearing_ratio=Decimal("1.75")),
    )

    # "less than 1.75" excludes 1.75, unlike "not more than".
    assert result.is_breach


# -- monetary covenants ----------------------------------------------------


def test_minimum_net_worth_below_the_floor_is_a_breach() -> None:
    result = evaluate(
        terms(
            CovenantType.MINIMUM_NET_WORTH,
            threshold_amount=Decimal("500000000"),
            threshold_currency="MYR",
        ),
        facts(net_worth=Decimal("420000000")),
    )

    assert result.is_breach
    assert "RM500 million" in result.explanation


def test_a_currency_mismatch_refuses_to_convert() -> None:
    result = evaluate(
        terms(
            CovenantType.MINIMUM_NET_WORTH,
            threshold_amount=Decimal("100000000"),
            threshold_currency="USD",
        ),
        facts(net_worth=Decimal("450000000"), currency="MYR"),
    )

    # Inventing an FX rate inside a breach test is exactly the silent error
    # this system exists to avoid.
    assert result.status is RuleStatus.INSUFFICIENT_DATA
    assert "conversion" in result.explanation


def test_cross_default_triggers_when_accelerated_debt_reaches_the_threshold() -> None:
    triggered = evaluate(
        terms(
            CovenantType.CROSS_DEFAULT,
            threshold_amount=Decimal("30000000"),
            threshold_currency="MYR",
        ),
        facts(accelerated_indebtedness=Decimal("30000000")),
    )
    clear = evaluate(
        terms(
            CovenantType.CROSS_DEFAULT,
            threshold_amount=Decimal("30000000"),
            threshold_currency="MYR",
        ),
        facts(accelerated_indebtedness=Decimal("29999999")),
    )

    # Direction is inverted relative to a maintenance covenant: reaching the
    # threshold is the event, not the limit.
    assert triggered.is_breach
    assert clear.status is not RuleStatus.BREACH


@pytest.mark.parametrize(
    ("covenant_type", "field"),
    [
        (CovenantType.DISPOSAL_RESTRICTION, "disposals_value"),
        (CovenantType.DISTRIBUTION_RESTRICTION, "distributions_value"),
    ],
)
def test_activity_caps_breach_when_exceeded(covenant_type: CovenantType, field: str) -> None:
    over = evaluate(
        terms(covenant_type, threshold_amount=Decimal("50000000"), threshold_currency="MYR"),
        facts(**{field: Decimal("60000000")}),
    )
    under = evaluate(
        terms(covenant_type, threshold_amount=Decimal("50000000"), threshold_currency="MYR"),
        facts(**{field: Decimal("10000000")}),
    )

    assert over.is_breach
    assert under.status is RuleStatus.OK


# -- rating triggers -------------------------------------------------------


def test_a_rating_trigger_compares_ordinally_not_lexically() -> None:
    result = evaluate(
        terms(CovenantType.RATING_TRIGGER, trigger_rating="A"),
        facts(current_rating="A-", rating_agency=RatingAgency.MARC),
    )

    # "A-" sorts after "A" as a string and is one notch worse as a rating.
    assert result.is_breach
    assert result.severity is Severity.HIGH


def test_a_rating_above_the_trigger_is_compliant() -> None:
    result = evaluate(
        terms(CovenantType.RATING_TRIGGER, trigger_rating="BBB+"),
        facts(current_rating="A-", rating_agency=RatingAgency.MARC),
    )

    assert result.status is RuleStatus.OK


def test_sitting_exactly_on_the_trigger_is_at_risk() -> None:
    result = evaluate(
        terms(CovenantType.RATING_TRIGGER, trigger_rating="A-"),
        facts(current_rating="A-", rating_agency=RatingAgency.MARC),
    )

    # Rating actions are discrete: the next one lands on the trigger.
    assert result.status is RuleStatus.AT_RISK


def test_ram_and_marc_notations_compare_on_one_scale() -> None:
    result = evaluate(
        terms(CovenantType.RATING_TRIGGER, trigger_rating="A+"),
        facts(current_rating="AA3", rating_agency=RatingAgency.RAM),
    )

    # RAM's AA3 is MARC's AA-, which is better than A+.
    assert result.status is RuleStatus.OK


def test_an_unparseable_rating_yields_insufficient_data_not_a_guess() -> None:
    result = evaluate(
        terms(CovenantType.RATING_TRIGGER, trigger_rating="A"),
        facts(current_rating="not-a-rating"),
    )

    assert result.status is RuleStatus.INSUFFICIENT_DATA


# -- event covenants -------------------------------------------------------


def test_negative_pledge_breaches_only_outside_the_permitted_carve_out() -> None:
    breached = evaluate(
        terms(CovenantType.NEGATIVE_PLEDGE),
        facts(security_created=True, security_is_permitted=False),
    )
    permitted = evaluate(
        terms(CovenantType.NEGATIVE_PLEDGE),
        facts(security_created=True, security_is_permitted=True),
    )
    none_created = evaluate(terms(CovenantType.NEGATIVE_PLEDGE), facts(security_created=False))

    assert breached.is_breach
    assert permitted.status is RuleStatus.OK
    assert none_created.status is RuleStatus.OK


def test_change_of_control_is_an_event() -> None:
    occurred = evaluate(terms(CovenantType.CHANGE_OF_CONTROL), facts(change_of_control=True))
    clear = evaluate(terms(CovenantType.CHANGE_OF_CONTROL), facts(change_of_control=False))

    assert occurred.is_breach
    assert clear.status is RuleStatus.OK


def test_shariah_non_compliance_names_the_purchase_undertaking() -> None:
    result = evaluate(terms(CovenantType.SHARIAH_NON_COMPLIANCE), facts(shariah_compliant=False))

    assert result.is_breach
    assert result.severity is Severity.CRITICAL
    # CLAUDE.md 6: it is a dissolution event, and what follows is a purchase
    # undertaking rather than a cure period.
    assert "dissolution event" in result.explanation
    assert "purchase undertaking" in result.explanation


def test_a_covenant_with_no_machine_test_is_not_applicable_not_compliant() -> None:
    result = evaluate(terms(CovenantType.OTHER), facts())

    assert result.status is RuleStatus.NOT_APPLICABLE
    assert "human review" in result.explanation


# -- the rule that runs through everything ---------------------------------


@pytest.mark.parametrize("covenant_type", list(CovenantType))
def test_no_covenant_type_ever_reports_ok_on_empty_facts(covenant_type: CovenantType) -> None:
    """Silence is not compliance.

    With no facts at all, every covenant must report INSUFFICIENT_DATA or
    NOT_APPLICABLE. An OK here would mean a breach report that quietly claims
    compliance for issuers who reported nothing.
    """
    result = evaluate(terms(covenant_type), facts())

    assert result.status in (RuleStatus.INSUFFICIENT_DATA, RuleStatus.NOT_APPLICABLE)
    assert not result.is_breach


def test_evaluate_all_and_breaches_filter_together() -> None:
    given = [
        terms(CovenantType.GEARING_RATIO, threshold_ratio=Decimal("1.75")),
        terms(CovenantType.INTEREST_COVER, threshold_ratio=Decimal("3.00")),
    ]
    results = evaluate_all(
        given, facts(gearing_ratio=Decimal("1.90"), interest_cover=Decimal("4.00"))
    )

    assert len(results) == 2
    assert [item.covenant_type for item in breaches(results)] == [CovenantType.GEARING_RATIO]


def test_every_covenant_type_has_a_dedicated_test() -> None:
    """PLAN.md, Phase 4 acceptance, enforced mechanically.

    Adding a `CovenantType` without a test for it fails here rather than
    shipping a covenant the engine handles silently and wrongly.
    """
    source = __import__("pathlib").Path(__file__).read_text()
    untested = [
        covenant_type.name
        for covenant_type in CovenantType
        if f"CovenantType.{covenant_type.name}" not in source
    ]

    assert untested == []
