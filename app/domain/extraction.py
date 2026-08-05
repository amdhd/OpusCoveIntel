"""What an extractor produces, before it becomes a row.

PLAN.md 3 runs a rule-based extractor and an LLM extractor over the same spans,
deliberately. This is the type they both produce, which is what makes them
comparable at all: where they disagree, the field goes to human review at no
extra model cost.

Phase 4 fills these from regex. Phase 6 fills the same shape from Opus.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import ClauseType, CovenantType, ExtractionMethod
from app.domain.rules import CovenantTerms


class RuleExtraction(BaseModel):
    """One clause found in one chunk, with the span it was read from.

    `quote` is a verbatim slice of the chunk: `chunk_text[char_start:char_end]`.
    A rule extraction cannot invent a quote the way a model can, but it is
    verified on the same code path anyway (CLAUDE.md 1.3) -- an extractor that
    is trusted because of what it is, rather than because of what it produced,
    is exactly the assumption this system refuses to make.
    """

    model_config = ConfigDict(frozen=True)

    clause_type: ClauseType
    covenant_type: CovenantType | None = None
    quote: str
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)
    method: ExtractionMethod = ExtractionMethod.RULE
    # Machine-evaluable terms, when the pattern captured enough to build them.
    # A covenant detected but not quantified ("shall not create any security
    # interest") has none, and that is a real distinction: it can be reported
    # but not evaluated.
    terms: CovenantTerms | None = None
    normalized: dict[str, str] = Field(default_factory=dict)
    pattern_id: str = ""

    @model_validator(mode="after")
    def _span_is_ordered(self) -> RuleExtraction:
        if self.char_end <= self.char_start:
            raise ValueError("extraction span must be non-empty and ordered")
        return self

    @property
    def threshold_amount(self) -> Decimal | None:
        return self.terms.threshold_amount if self.terms else None
