"""Pydantic models that the LLM extractor fills in via structured output.

CLAUDE.md 1.2 requires every fact to be traceable to a span. The LLM output
carries `source_quote` as a schema field — not via the Anthropic citations API
(which is incompatible with `output_config.format`, per CLAUDE.md §2). We verify
the quote against the chunk ourselves in `app/extract/citations.py`.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.domain.enums import ClauseType, CovenantType, RatingAgency, Severity
from app.domain.rules import ComparisonOperator


class LLMCovenantExtraction(BaseModel):
    """One covenant the LLM found in a candidate span.

    This is the Pydantic model that validates the LLM's structured output.
    It is deliberately separate from `RuleExtraction` — the LLM does not
    produce span offsets (those come from citation verification) and it
    returns a shape that is closer to the DB row it will become.
    """

    model_config = ConfigDict(frozen=True)

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


def _inline_refs(schema: dict[str, Any], defs: dict[str, Any]) -> None:
    """Recursively replace $ref nodes with their definitions."""
    if isinstance(schema, dict):
        schema.pop("title", None)
        if "$ref" in schema:
            ref_key = schema["$ref"].split("/")[-1]
            if ref_key in defs:
                resolved = dict(defs[ref_key])
                resolved.pop("title", None)
                schema.clear()
                schema.update(resolved)
        else:
            for value in schema.values():
                _inline_refs(value, defs)
    elif isinstance(schema, list):
        for item in schema:
            _inline_refs(item, defs)


def extraction_jsonschema() -> dict[str, Any]:
    """The JSON Schema the adapter sends as `response_schema` for extraction.

    Cached here so the same bytes flow through every call — a requirement for
    prompt caching (CLAUDE.md §2: prefix must be byte-stable, and json.dumps
    with sort_keys=True on the same model always produces the same output).

    $defs/$ref are inlined recursively so the schema is a single flat object —
    some provider structured-output endpoints reject $ref.
    """
    raw = LLMCovenantExtraction.model_json_schema()
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
