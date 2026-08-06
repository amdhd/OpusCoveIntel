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

# v3: rule 11, redemption-table rows are not covenants.
# v2: the response became a list of covenants. The version is part of the
# response-cache key and the extraction identity (CLAUDE.md 1.7), so bumping it
# is what stops a v1 single-covenant answer being replayed against a schema that
# now expects many -- and what makes re-extraction happen at all.
PROMPT_VERSION: str = "extract-covenant-v3"

# -- Schema rendering (extracted once, byte-stable) --------------------------

_SCHEMA_JSON: str = json.dumps(EXTRACTION_JSON_SCHEMA, sort_keys=True, indent=2)


# -- Few-shot examples -------------------------------------------------------

_FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    # First, deliberately: it is the case the single-covenant schema could not
    # express, and the one the eval harness caught the model getting wrong.
    {
        "text": (
            "The Issuer shall at all times maintain a consolidated gearing ratio of "
            "not more than 1.75 times, and shall further maintain a finance service "
            "cover ratio of not less than 1.50 times, in each case tested "
            "semi-annually."
        ),
        "output": json.dumps(
            {
                "covenants": [
                    {
                        "clause_type": "financial_covenant",
                        "covenant_type": "gearing_ratio",
                        "source_quote": (
                            "The Issuer shall at all times maintain a consolidated gearing "
                            "ratio of not more than 1.75 times"
                        ),
                        "confidence": 0.96,
                        "summary": "Gearing ratio must not exceed 1.75x",
                        "threshold_amount": None,
                        "threshold_currency": None,
                        "threshold_ratio": 1.75,
                        "operator": "lte",
                        "trigger_rating": None,
                        "rating_agency": None,
                        "severity": "high",
                    },
                    {
                        "clause_type": "financial_covenant",
                        "covenant_type": "finance_service_cover",
                        "source_quote": (
                            "shall further maintain a finance service cover ratio of not "
                            "less than 1.50 times"
                        ),
                        "confidence": 0.96,
                        "summary": "Finance service cover must be at least 1.50x",
                        "threshold_amount": None,
                        "threshold_currency": None,
                        "threshold_ratio": 1.50,
                        "operator": "gte",
                        "trigger_rating": None,
                        "rating_agency": None,
                        "severity": "high",
                    },
                ]
            },
            sort_keys=True,
        ),
    },
    # A passage that states no covenant. The empty list is the answer.
    {
        "text": (
            "PARTIES TO THE TRANSACTION The Issuer is a special purpose vehicle "
            "incorporated in Malaysia. The Trustee is Synthetic Trustees Berhad, "
            "acting for and on behalf of the holders of the Sukuk."
        ),
        "output": json.dumps({"covenants": []}, sort_keys=True),
    },
    {
        "text": (
            "The Issuer shall at all times maintain a consolidated gearing ratio of "
            "not more than 1.75 times, tested semi-annually on each financial half-year "
            "end by reference to the latest audited consolidated financial statements."
        ),
        "output": json.dumps(
            {
                "covenants": [
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
                    }
                ]
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
                "covenants": [
                    {
                        "clause_type": "cross_default",
                        "covenant_type": "cross_default",
                        "source_quote": (
                            "any indebtedness of the Issuer in an aggregate principal amount "
                            "exceeding RM30,000,000 becomes due and payable prior to its "
                            "stated maturity"
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
                    }
                ]
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
                "covenants": [
                    {
                        "clause_type": "rating_trigger",
                        "covenant_type": "rating_trigger",
                        "source_quote": (
                            "the rating assigned to the Sukuk by MARC is downgraded below BBB+"
                        ),
                        "confidence": 0.95,
                        "summary": ("Downgrade below BBB+ by MARC triggers additional security"),
                        "threshold_amount": None,
                        "threshold_currency": None,
                        "threshold_ratio": None,
                        "operator": None,
                        "trigger_rating": "BBB+",
                        "rating_agency": "MARC",
                        "severity": "high",
                    }
                ]
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
                "covenants": [
                    {
                        "clause_type": "negative_pledge",
                        "covenant_type": "negative_pledge",
                        "source_quote": (
                            "The Issuer shall not create or permit to subsist any security "
                            "interest over its assets"
                        ),
                        "confidence": 0.90,
                        "summary": (
                            "Issuer is restricted from creating security interests without consent"
                        ),
                        "threshold_amount": None,
                        "threshold_currency": None,
                        "threshold_ratio": None,
                        "operator": None,
                        "trigger_rating": None,
                        "rating_agency": None,
                        "severity": "high",
                    }
                ]
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
                "covenants": [
                    {
                        "clause_type": "shariah_compliance",
                        "covenant_type": "shariah_non_compliance",
                        "source_quote": (
                            "Sekiranya berlaku ketidakpatuhan Shariah, ia adalah suatu "
                            "kejadian pembubaran"
                        ),
                        "confidence": 0.90,
                        "summary": (
                            "Shariah non-compliance is a dissolution event requiring "
                            "purchase undertaking"
                        ),
                        "threshold_amount": None,
                        "threshold_currency": None,
                        "threshold_ratio": None,
                        "operator": None,
                        "trigger_rating": None,
                        "rating_agency": None,
                        "severity": "critical",
                    }
                ]
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
                "covenants": [
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
                    }
                ]
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

Your task is to read a passage from a trust deed or prospectus and extract structured
covenant data. Return ONLY valid JSON matching the schema below. Every field must
be populated from the text -- never invent values.

CRITICAL RULES:
1. Return EVERY covenant in the passage, as separate entries in "covenants".
   A passage often states several. One sentence can state several:
   "a gearing ratio of not more than 1.75 times, and a finance service cover
   ratio of not less than 1.50 times" is TWO covenants, not one. Extracting only
   the first silently loses the rest, so read to the end of the passage before
   you answer.
2. source_quote MUST be a verbatim slice of the input text. Copy it character for
   character. Each covenant quotes the part of the passage that states IT, not
   the whole passage.
3. If the passage states no covenant at all, return {{"covenants": []}}.
   An empty list is a correct answer. Do not invent a placeholder entry.
4. For monetary amounts, convert to the base unit:
   "RM30 million" -> 30000000, "RM500,000,000" -> 500000000.
5. For ratios, extract the numeric value: "1.75 times" -> 1.75.
6. For rating triggers, capture the exact notch (e.g. "BBB+") and the agency
   (MARC, RAM, S&P, Moody's, Fitch).
7. operator: use "lte" for "not more than"/"not exceeding", "gte" for "not less than".
8. Set confidence based on how clearly the covenant is stated:
   0.95+ for explicit thresholds, 0.80-0.94 for implied obligations,
   below 0.80 only when genuinely ambiguous.
9. If the text is in Bahasa Malaysia, extract the covenant in its original language
   but use English enum values for clause_type and covenant_type.
10. A threshold governing something other than the covenant itself is not the
   covenant's threshold. "consent of holders representing not less than 66 per
   cent of the nominal value" is a consent requirement, not a covenant ratio --
   leave threshold_ratio null.
11. A row of a redemption or call schedule -- a date beside a price, such as
   "2028-06-15 102.00 Optional" -- is NOT a covenant. Do not return one. The
   schedule is read separately as structured data. The sentence granting the
   redemption right may be a call_option clause; the table rows under it are not.

JSON SCHEMA:
{schema_json}

FEW-SHOT EXAMPLES:
{few_shot_examples}

Now extract every covenant from the passage in the user message. Return ONLY the JSON object."""


# Stable representation for cache-key computation.
SYSTEM_PROMPT_BYTES: bytes = build_system_prompt().encode("utf-8")
