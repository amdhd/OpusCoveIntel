"""Types the rules engine consumes and produces.

CLAUDE.md 1.1 is the reason this module exists: **the LLM never computes a
breach.** A model may read "a consolidated gearing ratio of not more than 1.75
times" and turn it into `CovenantTerms(operator=LTE, threshold=1.75)`, but the
comparison against an observed 1.9 is arithmetic, and arithmetic belongs in
Python where it can be unit-tested and replayed.

So the boundary is drawn here. Everything above this line is extraction;
everything below it is deterministic evaluation.

`domain/` is a pure leaf -- no db, no llm, no I/O (CLAUDE.md 3).
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import CovenantType, RatingAgency, Severity


class RuleStatus(StrEnum):
    """Outcome of evaluating one covenant against one set of facts."""

    OK = "ok"
    # Inside the covenant but within the warning margin -- the state a credit
    # analyst actually wants to hear about, since a breach is already too late.
    AT_RISK = "at_risk"
    BREACH = "breach"
    # A fact was missing. Emphatically not "ok": silence is not compliance
    # (CLAUDE.md 1.5 applied to the deterministic path).
    INSUFFICIENT_DATA = "insufficient_data"
    # The covenant does not admit a deterministic test -- free-text obligations
    # a rules engine must not pretend to evaluate.
    NOT_APPLICABLE = "not_applicable"


class ComparisonOperator(StrEnum):
    """How an observed value must relate to its threshold to stay compliant."""

    LTE = "lte"
    LT = "lt"
    GTE = "gte"
    GT = "gt"


class CovenantTerms(BaseModel):
    """The machine-evaluable form of a covenant.

    Mirrors `covenants.conditions_json` / `thresholds_json`, so a row round-trips
    through this type without loss.
    """

    model_config = ConfigDict(frozen=True)

    covenant_type: CovenantType
    operator: ComparisonOperator | None = None

    # Ratio covenants (gearing, interest cover). Decimal, not float: these are
    # compared for breach, and 1.7500000000000002 <= 1.75 is False.
    threshold_ratio: Decimal | None = None
    # Monetary covenants (cross-default, minimum net worth, disposal caps).
    threshold_amount: Decimal | None = None
    threshold_currency: str | None = None

    # Rating triggers. The rating is kept as written; comparison goes through
    # the ordinal rank in app/rules/ratings.py, never string comparison.
    trigger_rating: str | None = None
    rating_agency: RatingAgency = RatingAgency.UNKNOWN

    severity: Severity = Severity.MEDIUM
    # Fraction of the threshold within which OK becomes AT_RISK. 0.05 means
    # "warn once within 5% of the limit".
    warning_margin: Decimal = Decimal("0.05")
    description: str | None = None

    @model_validator(mode="after")
    def _threshold_amount_has_currency(self) -> CovenantTerms:
        if self.threshold_amount is not None and not self.threshold_currency:
            raise ValueError("a monetary threshold must name its currency")
        return self


class ObservedFacts(BaseModel):
    """What is known about an issuer at a point in time.

    Every field is optional and defaults to None, because "we do not know" is a
    first-class answer here. A missing fact yields INSUFFICIENT_DATA rather than
    a cheerful OK.
    """

    model_config = ConfigDict(frozen=True)

    as_of: dt.date

    # Financial ratios, as reported.
    gearing_ratio: Decimal | None = None
    interest_cover: Decimal | None = None
    finance_service_cover: Decimal | None = None
    net_worth: Decimal | None = None

    current_rating: str | None = None
    rating_agency: RatingAgency = RatingAgency.UNKNOWN

    # Event-driven covenants.
    accelerated_indebtedness: Decimal | None = None
    disposals_value: Decimal | None = None
    distributions_value: Decimal | None = None
    security_created: bool | None = None
    security_is_permitted: bool = False
    change_of_control: bool | None = None
    shariah_compliant: bool | None = None

    currency: str = "MYR"


class CovenantEvaluation(BaseModel):
    """A breach decision, with everything needed to defend it.

    `explanation` states the arithmetic in words; `inputs_used` names the facts
    that produced it. Together they make an answer auditable without re-running
    the engine -- which is the whole point of keeping this out of a model.
    """

    model_config = ConfigDict(frozen=True)

    covenant_type: CovenantType
    status: RuleStatus
    severity: Severity = Severity.MEDIUM

    observed: Decimal | None = None
    threshold: Decimal | None = None
    # Signed distance from the limit, in the covenant's own units. Positive is
    # compliant headroom; negative is the size of the breach.
    headroom: Decimal | None = None
    explanation: str = ""
    inputs_used: tuple[str, ...] = ()
    covenant_id: str | None = None
    instrument_id: str | None = None

    @property
    def is_breach(self) -> bool:
        return self.status is RuleStatus.BREACH

    @property
    def needs_attention(self) -> bool:
        return self.status in (RuleStatus.BREACH, RuleStatus.AT_RISK)


class Citation(BaseModel):
    """A pointer back to the text a claim came from (CLAUDE.md 1.2)."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    chunk_id: str | None = None
    clause_id: str | None = None
    page_number: int = Field(ge=1)
    quote: str
    char_start: int | None = None
    char_end: int | None = None
    section_title: str | None = None
