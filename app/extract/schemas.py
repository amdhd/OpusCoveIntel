"""Pydantic models that the LLM extractor fills in via structured output.

CLAUDE.md 1.2 requires every fact to be traceable to a span. The LLM output
carries `source_quote` as a schema field — not via the Anthropic citations API
(which is incompatible with `output_config.format`, per CLAUDE.md §2). We verify
the quote against the chunk ourselves in `app/extract/citations.py`.

**The response is a list, because a span is not one covenant.** This schema
returned a single covenant until the eval harness made the cost of that visible:
LLM recall 0.70 against the rule extractor's 0.98. The candidate spans were
never the problem — they contained 15 of 15 labelled covenants — but a span
routinely holds two or three:

    prospectus  gearing ratio + finance service cover, one sentence
    trust deed  interest cover + gearing + minimum net worth

The model had to pick one and the rest were lost silently, which was exactly
four false negatives. `app/extract/rule_extractor.py::_drop_overlapping` has
always keyed on `(clause_type, covenant_type)` for this reason; the LLM path
simply had no way to say "and also".
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.logging import get_logger
from app.domain.enums import ClauseType, CovenantType, RatingAgency, Severity
from app.domain.rules import ComparisonOperator

logger = get_logger(__name__)

# Bounds $ref inlining on a self-referential schema. Ours nests three levels.
_MAX_REF_DEPTH = 20


class LLMCovenantExtraction(BaseModel):
    """One covenant the LLM found in a candidate span.

    This is the Pydantic model that validates the LLM's structured output.
    It is deliberately separate from `RuleExtraction` — the LLM does not
    produce span offsets (those come from citation verification) and it
    returns a shape that is closer to the DB row it will become.

    Extra fields are forbidden for the same reason as on the response wrapper:
    a covenant object carrying keys this model does not know is not a covenant
    this model understands, and quietly dropping them would persist a partial
    reading as if it were a complete one.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    clause_type: ClauseType
    covenant_type: CovenantType | None = Field(
        default=None, description="None when the clause does not create an obligation"
    )
    source_quote: str = Field(
        ...,
        min_length=1,
        description="Verbatim text from the chunk — will be verified before persistence",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="0.0 = guessing, 1.0 = certain")
    summary: str = Field(
        default="", description="One-sentence summary of what this clause requires"
    )

    # --- Financial covenant fields ---
    threshold_amount: Decimal | None = Field(
        default=None, description="Monetary threshold in the covenant's currency (e.g. 30000000)"
    )
    threshold_currency: str | None = Field(
        default=None, max_length=3, description="ISO 4217 currency code (e.g. MYR)"
    )
    threshold_ratio: Decimal | None = Field(
        default=None, description="Ratio threshold (e.g. 1.75 for a gearing covenant)"
    )
    operator: ComparisonOperator | None = Field(
        default=None,
        description="How the observed value must relate to the threshold to be compliant",
    )

    # --- Rating trigger fields ---
    trigger_rating: str | None = Field(
        default=None, description="The rating notch that triggers the clause (e.g. BBB+)"
    )
    rating_agency: RatingAgency | None = Field(
        default=None, description="Agency that assigns the rating (MARC, RAM, etc.)"
    )

    # --- Meta ---
    severity: Severity = Field(
        default=Severity.MEDIUM,
        description="How critical a breach of this covenant would be",
    )

    @model_validator(mode="after")
    def _threshold_amount_requires_currency(self) -> LLMCovenantExtraction:
        if self.threshold_amount is not None and not self.threshold_currency:
            raise ValueError("a monetary threshold must name its currency")
        return self

    @model_validator(mode="after")
    def _rating_trigger_requires_agency(self) -> LLMCovenantExtraction:
        if self.trigger_rating is not None and self.rating_agency is None:
            raise ValueError("a rating trigger must name the rating agency")
        return self


# JSON Schema validation keywords that Anthropic's `output_config.format`
# rejects outright ("For 'number' type, properties maximum, minimum are not
# supported"). Dropping them costs nothing real: the schema sent to the model
# is a generation constraint, while `LLMCovenantExtraction.model_validate` is
# the authoritative gate and still enforces every one of them on the way in.
_UNSUPPORTED_SCHEMA_KEYWORDS: frozenset[str] = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)


def _strip_unsupported(node: Any) -> None:
    """Remove validation keywords the structured-output endpoint refuses."""
    if isinstance(node, dict):
        for keyword in _UNSUPPORTED_SCHEMA_KEYWORDS:
            node.pop(keyword, None)
        for value in node.values():
            _strip_unsupported(value)
    elif isinstance(node, list):
        for item in node:
            _strip_unsupported(item)


class LLMExtractionResponse(BaseModel):
    """Every covenant the LLM found in one candidate span.

    An empty list is a valid and useful answer: the candidate detector is tuned
    for recall, so a span reaching the model may hold no covenant at all. That
    reads better than the previous convention of a single
    `clause_type="covenant_other"` row at confidence 0.0, which was a covenant
    record asserting there was no covenant.

    **Extra fields are forbidden, and that is load-bearing here.** `covenants`
    defaults to `[]`, and Pydantic ignores unknown keys by default — so a
    response in some other shape entirely (the old single-covenant one, or
    nonsense) would validate happily as "no covenants found". Since an empty
    list is a *meaningful* answer under this schema, a malformed response must
    not be able to impersonate it: that would turn a broken call into a silent
    "nothing here" rather than the retry-then-review it deserves.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    covenants: list[LLMCovenantExtraction] = Field(
        default_factory=list,
        description=(
            "One entry per distinct covenant in the text. Two covenants in one "
            "sentence are two entries. Empty when the text states none."
        ),
    )


def _close_objects(node: Any) -> None:
    """Set `additionalProperties: false` on every object in the schema.

    Anthropic's structured-output endpoint rejects the request outright
    without it:

        output_config.format.schema: For 'object' type,
        'additionalProperties' must be explicitly set to false

    A 400 before inference, so it costs nothing but blocks everything -- every
    extraction call failed this way the first time the pipeline met the real
    API. Applied recursively because a nested object is checked too.
    """
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
        for value in node.values():
            _close_objects(value)
    elif isinstance(node, list):
        for item in node:
            _close_objects(item)


def _inline_refs(schema: dict[str, Any], defs: dict[str, Any], _depth: int = 0) -> None:
    """Recursively replace $ref nodes with their definitions.

    **Recurses into what it just inlined.** It did not, which went unnoticed
    while the top-level model was `LLMCovenantExtraction`: its enum `$ref`s sat
    one level down and were resolved on the way in. Wrapping the response in
    `{"covenants": [...]}` put them one level deeper, behind a `$ref` of their
    own — so the covenant object was inlined and every enum inside it was left
    as an unresolved `#/$defs/ClauseType`, which no provider can follow once
    `$defs` has been stripped.

    `_depth` bounds a schema that refers to itself. Ours does not, but a
    self-referential model would otherwise recurse until the stack ran out, and
    a schema builder is not the place to find that out.
    """
    if _depth > _MAX_REF_DEPTH:
        logger.warning("schema $ref inlining hit its depth limit; leaving the rest as-is")
        return
    if isinstance(schema, dict):
        schema.pop("title", None)
        if "$ref" in schema:
            ref_key = schema["$ref"].split("/")[-1]
            if ref_key in defs:
                resolved = dict(defs[ref_key])
                resolved.pop("title", None)
                schema.clear()
                schema.update(resolved)
                # The definition just inlined may itself hold $refs.
                for value in schema.values():
                    _inline_refs(value, defs, _depth + 1)
        else:
            for value in schema.values():
                _inline_refs(value, defs, _depth + 1)
    elif isinstance(schema, list):
        for item in schema:
            _inline_refs(item, defs, _depth + 1)


def extraction_jsonschema() -> dict[str, Any]:
    """The JSON Schema the adapter sends as `response_schema` for extraction.

    Cached here so the same bytes flow through every call — a requirement for
    prompt caching (CLAUDE.md §2: prefix must be byte-stable, and json.dumps
    with sort_keys=True on the same model always produces the same output).

    $defs/$ref are inlined recursively so the schema is a single flat object —
    some provider structured-output endpoints reject $ref.
    """
    raw = LLMExtractionResponse.model_json_schema()
    defs = raw.get("$defs", {})

    raw.pop("title", None)
    raw.pop("$defs", None)

    _inline_refs(raw, defs)
    _strip_unsupported(raw)
    _close_objects(raw)

    # Ensure the JSON is stable — sort_keys is what makes this cacheable.
    result: dict[str, Any] = json.loads(json.dumps(raw, sort_keys=True))
    return result


# Build once at import time so every call shares the identical bytes.
EXTRACTION_JSON_SCHEMA: dict[str, Any] = extraction_jsonschema()
