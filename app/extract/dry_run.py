"""Pricing a document before spending anything on it.

PLAN.md 2: "a `--dry-run` estimator that prices a document via `count_tokens`
before spending anything." This is that estimator, and it is free: candidate
detection is regex over text already in the database, and token counts come
from the offline estimator in `app/llm/cost.py`.

**The figure is a ceiling, not a forecast.** Output is priced at the full
`max_tokens` budget because that is what the budget guard itself assumes when
it decides whether to allow a call -- so the number here is the one that
matters for "will this be rejected", not the one that will appear on an
invoice. Real cost lands well below it: completions are a few hundred tokens
rather than 8000, and the cached prefix bills at 0.1x from the second call on.
Both effects are reported separately so the gap is visible rather than
surprising.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.extract.candidates import CandidateDetectionService
from app.extract.llm_extractor import EXTRACTION_MAX_TOKENS
from app.extract.prompts import build_system_prompt, build_user_message
from app.llm.cost import count_tokens_estimate, estimate_cost


@dataclass(frozen=True)
class CandidateCostEstimate:
    """The worst-case cost of extracting a set of candidate spans.

    This is the shared unit of estimation: both the CLI `--dry-run` (via
    `estimate_document`) and the pipeline's refuse-before-spending preflight
    price the same way, so the number an operator sees and the number the guard
    acts on can never disagree.
    """

    candidates: int
    prompt_tokens: int
    prefix_tokens: int
    ceiling_usd: Decimal

    @property
    def cached_prefix_tokens(self) -> int:
        """Prefix tokens that bill at 0.1x after the first call.

        The system prompt is byte-stable across calls, so every call after the
        first reads it from cache instead of paying full input rate.
        """
        return self.prefix_tokens * max(0, self.candidates - 1)


def estimate_candidate_cost(
    candidate_texts: list[str],
    *,
    settings: Settings | None = None,
) -> CandidateCostEstimate:
    """Price extracting these candidate spans. Dispatches nothing.

    Prices output at the full `max_tokens` budget per candidate, matching what
    the budget guard assumes — so this is the ceiling that decides "will this be
    rejected", not a forecast of the invoice.
    """
    config = settings or get_settings()

    prefix_tokens = count_tokens_estimate(build_system_prompt())
    prompt_tokens = sum(
        prefix_tokens + count_tokens_estimate(build_user_message(text)) for text in candidate_texts
    )
    ceiling = estimate_cost(
        provider="anthropic",
        model_id=config.EXTRACTION_MODEL,
        prompt_tokens=prompt_tokens,
        max_output_tokens=EXTRACTION_MAX_TOKENS * len(candidate_texts),
    )
    return CandidateCostEstimate(
        candidates=len(candidate_texts),
        prompt_tokens=prompt_tokens,
        prefix_tokens=prefix_tokens,
        ceiling_usd=ceiling.total,
    )


@dataclass(frozen=True)
class DocumentEstimate:
    """What one document would cost to extract, before anything is dispatched."""

    document_id: uuid.UUID
    candidates: int
    prompt_tokens: int
    cached_prefix_tokens: int
    ceiling_usd: Decimal
    per_document_cap: Decimal

    @property
    def over_cap(self) -> bool:
        """Whether the ceiling alone would exhaust the per-document budget."""
        return self.ceiling_usd > self.per_document_cap

    def describe(self) -> str:
        if self.candidates == 0:
            return (
                f"{self.document_id}: no candidate spans — nothing to send, $0. "
                f"(Either the document holds no covenant language the patterns "
                f"recognise, or it has not been chunked yet.)"
            )
        # What the operator is told must match what the pipeline will do. This
        # said "the guard will stop mid-document", which was true until the
        # preflight landed and is now the opposite of the behaviour: an
        # over-cap document is refused before the first call and costs $0.
        note = (
            "  ** exceeds the per-document cap; refused before the first call, $0 **"
            if self.over_cap
            else ""
        )
        return (
            f"{self.document_id}: {self.candidates} candidate span(s), "
            f"~{self.prompt_tokens:,} prompt tokens, "
            f"worst case ${self.ceiling_usd:.4f} "
            f"(cap ${self.per_document_cap}){note}\n"
            f"    Worst case prices every completion at the full "
            f"{EXTRACTION_MAX_TOKENS:,}-token budget, which is what the budget guard "
            f"assumes. Actual spend is typically far lower, and "
            f"~{self.cached_prefix_tokens:,} tokens of prompt prefix bill at 0.1x "
            f"after the first call."
        )


async def estimate_document(
    session: AsyncSession,
    document_id: uuid.UUID,
    *,
    settings: Settings | None = None,
) -> DocumentEstimate:
    """Price the LLM half of extraction for one document. Dispatches nothing."""
    config = settings or get_settings()

    candidates = await CandidateDetectionService(session).detect(document_id)
    estimate = estimate_candidate_cost([c.text for c in candidates], settings=config)

    return DocumentEstimate(
        document_id=document_id,
        candidates=estimate.candidates,
        prompt_tokens=estimate.prompt_tokens,
        cached_prefix_tokens=estimate.cached_prefix_tokens,
        ceiling_usd=estimate.ceiling_usd,
        per_document_cap=config.MAX_COST_PER_DOCUMENT_USD,
    )
