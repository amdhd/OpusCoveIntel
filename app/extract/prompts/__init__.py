"""Prompt builder for LLM covenant extraction.

PLAN.md Phase 6: "Versioned Jinja2 prompts." We achieve the same byte-stable,
version-controlled prompt construction with Python string formatting so no new
dependency is required. What makes this cacheable is:

1. The system prompt is a pure function of the prompt version — no timestamps,
   no UUIDs, no dynamic data.
2. The JSON schema is serialised with sort_keys=True and embedded in the prefix.
3. Every call for the same version produces identical bytes.

Usage:
    from app.extract.prompts import build_system_prompt, PROMPT_VERSION
    system = build_system_prompt()  # cacheable prefix
    messages = [{"role": "user", "content": candidate_text}]
"""

from __future__ import annotations

import json

from app.extract.schemas import EXTRACTION_JSON_SCHEMA

PROMPT_VERSION: str = "extract-covenant-v1"

# -- Schema rendering (extracted once, byte-stable) --------------------------

_SCHEMA_JSON: str = json.dumps(EXTRACTION_JSON_SCHEMA, sort_keys=True, indent=2)


# -- Few-shot examples -------------------------------------------------------

_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "text": (
            "The Issuer shall at all times maintain a consolidated gearing ratio of "
            "not more than 1.75 times, tested semi-annually on each financial half-year "
            "end by reference to the latest audited consolidated financial statements."
        ),
        "output": json.dumps(
            {
                "clause_type": "financial_covenant",
                "covenant_type": "gearing_ratio",
                "source_quote": (
                    "The Issuer shall at all times maintain a consolidated gearing ratio "
                    "of not more than 1.75 times"
                ),
                "confidence": 0.95,
                "summary": "Issuer must maintain gearing ratio ≤ 1.75x",
                "threshold_amount": None,
                "threshold_currency": None,
                "threshold_ratio": 1.75,
                "operator": "lte",
                "trigger_rating": None,
                "rating_agency": None,
                "severity": "high",
            },
            sort_keys=True,
        ),
    },
    {
        "text": (
            "An event of default shall occur if any indebtedness of the Issuer in an "
            "aggregate principal amount exceeding RM30,000,000 becomes due and payable "
            "prior to its stated maturity."
        ),
        "output": json.dumps(
            {
                "clause_type": "cross_default",
                "covenant_type": "cross_default",
                "source_quote": (
                    "any indebtedness of the Issuer in an aggregate principal amount "
                    "exceeding RM30,000,000 becomes due and payable prior to its stated maturity"
                ),
                "confidence": 0.95,
                "summary": "Cross-default triggered at RM30m of accelerated indebtedness",
                "threshold_amount": 30000000,
                "threshold_currency": "MYR",
                "threshold_ratio": None,
                "operator": None,
                "trigger_rating": None,
                "rating_agency": None,
                "severity": "high",
            },
            sort_keys=True,
        ),
    },
    {
        "text": (
            "In the event the rating assigned to the Sukuk by MARC is downgraded below "
            "BBB+, the Issuer shall notify the Trustee within five business days and "
            "shall procure additional security acceptable to the Trustee."
        ),
        "output": json.dumps(
            {
                "clause_type": "rating_trigger",
                "covenant_type": "rating_trigger",
                "source_quote": (
                    "the rating assigned to the Sukuk by MARC is downgraded below BBB+"
                ),
                "confidence": 0.95,
                "summary": "Downgrade below BBB+ by MARC triggers additional security requirement",
                "threshold_amount": None,
                "threshold_currency": None,
                "threshold_ratio": None,
                "operator": None,
                "trigger_rating": "BBB+",
                "rating_agency": "MARC",
                "severity": "high",
            },
            sort_keys=True,
        ),
    },
    {
        "text": (
            "The Issuer shall not create or permit to subsist any security interest "
            "over its assets without the prior written consent of the Trustee."
        ),
        "output": json.dumps(
            {
                "clause_type": "negative_pledge",
                "covenant_type": "negative_pledge",
                "source_quote": (
                    "The Issuer shall not create or permit to subsist any security "
                    "interest over its assets"
                ),
                "confidence": 0.90,
                "summary": "Issuer is restricted from creating security interests without consent",
                "threshold_amount": None,
                "threshold_currency": None,
                "threshold_ratio": None,
                "operator": None,
                "trigger_rating": None,
                "rating_agency": None,
                "severity": "high",
            },
            sort_keys=True,
        ),
    },
    {
        "text": (
            "Sekiranya berlaku ketidakpatuhan Shariah, ia adalah suatu kejadian "
            "pembubaran dan Penerbit hendaklah melaksanakan aku janji pembelian."
        ),
        "output": json.dumps(
            {
                "clause_type": "shariah_compliance",
                "covenant_type": "shariah_non_compliance",
                "source_quote": (
                    "Sekiranya berlaku ketidakpatuhan Shariah, ia adalah suatu kejadian pembubaran"
                ),
                "confidence": 0.90,
                "summary": (
                    "Shariah non-compliance is a dissolution event requiring purchase undertaking"
                ),
                "threshold_amount": None,
                "threshold_currency": None,
                "threshold_ratio": None,
                "operator": None,
                "trigger_rating": None,
                "rating_agency": None,
                "severity": "critical",
            },
            sort_keys=True,
        ),
    },
    {
        "text": (
            "The Issuer shall maintain a consolidated net worth of not less than "
            "RM500,000,000 at all times."
        ),
        "output": json.dumps(
            {
                "clause_type": "financial_covenant",
                "covenant_type": "minimum_net_worth",
                "source_quote": (
                    "The Issuer shall maintain a consolidated net worth of not less "
                    "than RM500,000,000 at all times"
                ),
                "confidence": 0.95,
                "summary": "Minimum consolidated net worth of RM500m",
                "threshold_amount": 500000000,
                "threshold_currency": "MYR",
                "threshold_ratio": None,
                "operator": "gte",
                "trigger_rating": None,
                "rating_agency": None,
                "severity": "high",
            },
            sort_keys=True,
        ),
    },
]


def build_system_prompt() -> str:
    """The extraction system prompt, stable for a given PROMPT_VERSION.

    Returns a string that is byte-identical across calls with the same version,
    making it suitable as the prompt-cached prefix.
    """
    few_shot_text = _render_few_shots()
    return _SYSTEM_TEMPLATE.format(
        schema_json=_SCHEMA_JSON,
        few_shot_examples=few_shot_text,
    )


def build_user_message(candidate_text: str) -> str:
    """The user message carrying one candidate span to extract from."""
    return f"Extract covenants from the following clause text:\n\n{candidate_text}"


def _render_few_shots() -> str:
    parts: list[str] = []
    for i, example in enumerate(_FEW_SHOT_EXAMPLES, start=1):
        parts.append(f"Example {i}:\nText: {example['text']}\nOutput: {example['output']}")
    return "\n\n".join(parts)


_SYSTEM_TEMPLATE: str = """\
You are a legal document analyst specialised in Malaysian sukuk and bond covenants.

Your task is to read a clause from a trust deed or prospectus and extract structured
covenant data. Return ONLY valid JSON matching the schema below. Every field must
be populated from the text -- never invent values.

CRITICAL RULES:
1. source_quote MUST be a verbatim slice of the input text. Copy it character for character.
2. If the text does not contain a covenant, return a JSON object with
   clause_type="covenant_other" and confidence=0.0.
3. For monetary amounts, convert to the base unit:
   "RM30 million" -> 30000000, "RM500,000,000" -> 500000000.
4. For ratios, extract the numeric value: "1.75 times" -> 1.75.
5. For rating triggers, capture the exact notch (e.g. "BBB+") and the agency
   (MARC, RAM, S&P, Moody's, Fitch).
6. operator: use "lte" for "not more than"/"not exceeding", "gte" for "not less than".
7. Set confidence based on how clearly the covenant is stated:
   0.95+ for explicit thresholds, 0.80-0.94 for implied obligations,
   below 0.80 only when genuinely ambiguous.
8. If the text is in Bahasa Malaysia, extract the covenant in its original language
   but use English enum values for clause_type and covenant_type.

JSON SCHEMA:
{schema_json}

FEW-SHOT EXAMPLES:
{few_shot_examples}

Now extract the covenant from the text provided in the user message. Return ONLY the JSON object."""


# Stable representation for cache-key computation.
SYSTEM_PROMPT_BYTES: bytes = build_system_prompt().encode("utf-8")
