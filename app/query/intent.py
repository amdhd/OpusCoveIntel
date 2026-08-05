"""Deterministic intent classification.

Phase 7 replaces this with a classifier node in the LangGraph agent. Keyword
rules are enough for Phase 4 and have one property the model version will not:
they are free, and their mistakes are inspectable.

The important class is UNSUPPORTED. This system is decision support, not an
oracle -- it must refuse forecasts, recommendations and anything else the
corpus cannot evidence (PLAN.md 7, CLAUDE.md 1.5). Refusal is checked *first*,
because "should I buy this sukuk?" contains the word "sukuk" and would
otherwise classify as a perfectly answerable instrument lookup.
"""

from __future__ import annotations

import re
from typing import Final

from app.domain.enums import QueryIntent

# Asking for a prediction, a recommendation, or an opinion. None of these are
# answerable from a corpus of offering documents at any confidence.
_UNSUPPORTED: Final[tuple[str, ...]] = (
    r"\bshould\s+(?:i|we)\b",
    r"\brecommend",
    r"\bwill\b.*\b(?:rally|fall|rise|outperform|default)\b",
    r"\bforecast",
    r"\bpredict",
    r"\bgoing\s+to\s+happen\b",
    r"\bworth\s+buying\b",
    r"\bgood\s+investment\b",
    r"\bfair\s+value\b",
    r"\bprice\s+target\b",
)

_BREACH: Final[tuple[str, ...]] = (
    r"\bbreach",
    r"\bviolat",
    r"\bin\s+compliance\b",
    r"\bcomply\b",
    r"\bcompliant\b",
    r"\bheadroom\b",
    r"\btrip(?:ped|s)?\b",
    r"\btrigger(?:ed)?\s+(?:by|at)\b",
    r"\bwould\s+be\s+triggered\b",
)

_PORTFOLIO: Final[tuple[str, ...]] = (
    r"\bportfolio",
    r"\bexposure\b",
    r"\bholding",
    r"\bnav\b",
    r"\bweight",
    r"\bfund\b",
    r"\bhow\s+much\s+do\s+we\s+(?:hold|own)\b",
)

_COVENANT: Final[tuple[str, ...]] = (
    r"\bcovenant",
    r"\bnegative\s+pledge\b",
    r"\bcross[\s-]?default\b",
    r"\bgearing\b",
    r"\bnisbah\s+gearan\b",
    r"\binterest\s+cover",
    r"\bfinance\s+service\s+cover",
    r"\brating\s+trigger\b",
    r"\bthreshold\b",
    r"\bcall\s+(?:date|price|schedule)\b",
    r"\bredeem\b",
    r"\bredemption\b",
    r"\bshariah\b",
    r"\bdissolution\b",
    r"\bpurchase\s+undertaking\b",
    r"\bevent\s+of\s+default\b",
    r"\bchange\s+of\s+control\b",
)

_INSTRUMENT: Final[tuple[str, ...]] = (
    r"\bissuer\b",
    r"\bmaturity\b",
    r"\bissue\s+size\b",
    r"\bisin\b",
    r"\brated\b",
    r"\brating\s+of\b",
    r"\bwhich\s+instruments?\b",
    r"\bsukuk\s+structure\b",
    r"\bijarah\b|\bwakalah\b|\bmusharakah\b|\bmudharabah\b|\bmurabahah\b|\bistisna",
)


def _matches(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def classify(question: str) -> QueryIntent:
    """Map a question onto the intent space (PLAN.md 5)."""
    text = question.lower().strip()
    if not text:
        return QueryIntent.UNSUPPORTED

    # Order matters: refusal first, then the most specific answerable intents.
    if _matches(_UNSUPPORTED, text):
        return QueryIntent.UNSUPPORTED
    if _matches(_BREACH, text):
        return QueryIntent.COVENANT_BREACH_CHECK
    if _matches(_PORTFOLIO, text):
        return QueryIntent.PORTFOLIO_QUERY
    if _matches(_COVENANT, text):
        return QueryIntent.COVENANT_LOOKUP
    if _matches(_INSTRUMENT, text):
        return QueryIntent.INSTRUMENT_LOOKUP
    return QueryIntent.DOCUMENT_SEARCH


def mentioned_entities(question: str, candidates: list[str]) -> list[str]:
    """Which known names the question names.

    Substring matching on registered names, deliberately literal: attaching an
    answer to the wrong issuer produces a confident, wrong portfolio number,
    which is the failure mode this system exists to prevent.
    """
    text = question.lower()
    return [name for name in candidates if name.lower() in text]
