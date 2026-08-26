"""Cost estimation and token pricing for every provider/model combination.

docs/plan.md 2: the budget guard checks cost *before* dispatch. This module provides
the prices so the guard doesn't guess. Money is Decimal, never float (CLAUDE.md 6).

Prices are per million tokens (MTok). Model IDs are read from settings, not
hardcoded, so a model change doesn't require touching this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Final

# -- Pricing (per million tokens, USD) ---------------------------------------
# Verified against provider docs as of 2026-08. These are configuration, not
# constants: a price change is a settings change in practice, but they live here
# because they're provider facts rather than deployment choices.

_ProviderPricing = dict[str, dict[str, Decimal]]

# Model → {input_per_MTok, output_per_MTok, cache_read_per_MTok, cache_write_per_MTok}
_CLAUDE_PRICING: Final[_ProviderPricing] = {
    "claude-opus-5": {
        "input": Decimal("5.00"),
        "output": Decimal("25.00"),
        "cache_read": Decimal("0.50"),
        "cache_write": Decimal("6.25"),
    },
}

_OPENAI_PRICING: Final[_ProviderPricing] = {
    "gpt-4o": {
        "input": Decimal("2.50"),
        "output": Decimal("10.00"),
    },
    "gpt-4o-mini": {
        "input": Decimal("0.15"),
        "output": Decimal("0.60"),
    },
}

_QWEN_PRICING: Final[_ProviderPricing] = {
    "qwen-plus": {
        "input": Decimal("0.80"),
        "output": Decimal("2.00"),
    },
    "text-embedding-v4": {
        "input": Decimal("0.07"),
    },
}

# Priced at zero *on purpose*: these providers never bill, so a zero here is a
# fact rather than a missing entry. Every other unknown model raises -- see
# `_get_pricing`.
_FREE_PRICING: Final[dict[str, Decimal]] = {
    "input": Decimal("0.00"),
    "output": Decimal("0.00"),
}

_MOCK_PRICING: Final[_ProviderPricing] = {
    # The CI provider. Zero because it makes no network call, not because we
    # failed to look up a price.
    "mock-v1": _FREE_PRICING,
}

# USD per VLM page. GPT-4o images: ~$0.00213 per 512x512 tile at low-res;
# a typical scanned A4 page at detail:high is ~4 tiles, but we cap it
# conservatively. This is an estimate until real-page tuning happens.
_ESTIMATED_COST_PER_VLM_PAGE: Final[Decimal] = Decimal("0.02")

# Dimension of embedding vectors — needed for Qwen pricing estimates.
_QWEN_EMBEDDING_PRICE_PER_1K_TOKENS: Final[Decimal] = Decimal("0.00007")


class UnknownModelPricingError(LookupError):
    """Raised when a provider/model pair has no entry in the price table.

    Failing closed is the whole point. If an unpriced model returned $0, the
    per-call, per-document and global guards would all pass it, and a single
    typo in `EXTRACTION_MODEL` -- or a provider shipping a new model id --
    would silently disable every ceiling in docs/plan.md 2. CLAUDE.md 1.4 says
    there is no silent LLM spend; an unknown price is the loudest possible
    version of silent.
    """

    def __init__(self, provider: str, model_id: str) -> None:
        self.provider = provider
        self.model_id = model_id
        super().__init__(
            f"no pricing for provider={provider!r} model={model_id!r}; "
            f"add it to app/llm/cost.py before routing spend through it"
        )


@dataclass(frozen=True)
class TokenCost:
    """The estimated cost of a call, broken down by token category."""

    input_cost: Decimal
    output_cost: Decimal
    cache_read_cost: Decimal
    cache_write_cost: Decimal
    total: Decimal

    def __bool__(self) -> bool:
        return self.total > 0


@dataclass(frozen=True)
class CallEstimate:
    """Pre-dispatch estimate so the budget guard can reject before spending."""

    model_id: str
    provider: str
    prompt_tokens: int
    max_output_tokens: int
    estimated_cost: Decimal
    would_use_cache: bool = False


def estimate_cost(
    *,
    provider: str,
    model_id: str,
    prompt_tokens: int,
    max_output_tokens: int = 0,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> TokenCost:
    """Price a call before it happens. Returns USD as Decimal.

    The three input categories are **disjoint**, matching how every provider
    reports usage: Anthropic's `input_tokens` already excludes
    `cache_read_input_tokens` and `cache_creation_input_tokens`. Subtracting
    cache reads from `prompt_tokens` here would double-discount them and
    understate real spend, so nothing is subtracted.

    Args:
        provider: One of "anthropic", "openai", "qwen", "mock".
        model_id: The exact model string (e.g. "claude-opus-5").
        prompt_tokens: Input tokens billed at the full input rate — i.e. those
            *not* served from or written to the prompt cache.
        max_output_tokens: The completion budget — we charge for the max, not
            the actual, because the guard stops *before* dispatch.
        cache_read_tokens: Tokens read from prompt cache (charged at 0.1x).
        cache_write_tokens: Tokens written to prompt cache (charged at 1.25x).

    Raises:
        UnknownModelPricingError: when the provider/model has no price entry.
    """
    pricing = _get_pricing(provider, model_id)

    def _mtok(tokens: int) -> Decimal:
        return Decimal(max(0, tokens)) / Decimal(1_000_000)

    input_cost = _mtok(prompt_tokens) * pricing.get("input", Decimal("0"))
    output_cost = _mtok(max_output_tokens) * pricing.get("output", Decimal("0"))
    cache_read_cost = _mtok(cache_read_tokens) * pricing.get("cache_read", Decimal("0"))
    cache_write_cost = _mtok(cache_write_tokens) * pricing.get("cache_write", Decimal("0"))

    total = input_cost + output_cost + cache_read_cost + cache_write_cost

    return TokenCost(
        input_cost=input_cost,
        output_cost=output_cost,
        cache_read_cost=cache_read_cost,
        cache_write_cost=cache_write_cost,
        total=total,
    )


def estimate_vlm_page_cost() -> Decimal:
    """Conservative per-page estimate for GPT-4o vision OCR."""
    return _ESTIMATED_COST_PER_VLM_PAGE


def estimate_embedding_cost(token_count: int) -> Decimal:
    """Estimate cost for embedding `token_count` tokens via Qwen."""
    return (Decimal(token_count) / Decimal(1_000)) * _QWEN_EMBEDDING_PRICE_PER_1K_TOKENS


def _get_pricing(provider: str, model_id: str) -> dict[str, Decimal]:
    """Resolve pricing for a provider+model, or raise.

    There is deliberately no zero-priced default: see `UnknownModelPricingError`.
    """
    tables: dict[str, _ProviderPricing] = {
        "anthropic": _CLAUDE_PRICING,
        "openai": _OPENAI_PRICING,
        "qwen": _QWEN_PRICING,
        "mock": _MOCK_PRICING,
    }
    provider_table = tables.get(provider)
    if provider_table is None:
        raise UnknownModelPricingError(provider, model_id)
    pricing = provider_table.get(model_id)
    if pricing is None:
        raise UnknownModelPricingError(provider, model_id)
    return pricing


def is_priced(provider: str, model_id: str) -> bool:
    """Whether spend through this provider/model can be estimated at all."""
    try:
        _get_pricing(provider, model_id)
    except UnknownModelPricingError:
        return False
    return True


def count_tokens_estimate(text: str) -> int:
    """Fast, offline token-count estimate. ~4 chars per token for English text.

    This is a budget-guard input, not a billing metric. It must be fast enough
    to run before every call (sub-ms), and conservative enough that we never
    underestimate by more than 15%. The actual count comes back in the provider
    response and is written to `llm_calls`.

    For languages with different tokenisation ratios (BM mixes English and
    Malay, which occupy different parts of the Unicode space), the estimate
    will be off — but in the direction of overestimation for non-Latin scripts,
    which is safe for a budget guard.
    """
    if not text:
        return 0
    # Count whitespace-delimited words, then inflate: tokenisers break rare
    # words into subwords, and punctuation into separate tokens.
    words = text.split()
    if not words:
        return 0
    char_count = len(text)
    # Average: 3 chars/token for dense text, 1.3 tokens/word. Take the larger.
    char_estimate = char_count // 3
    word_estimate = int(len(words) * 1.35)
    return max(char_estimate, word_estimate, 1)
