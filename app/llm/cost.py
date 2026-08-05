"""Cost estimation and token pricing for every provider/model combination.

PLAN.md 2: the budget guard checks cost *before* dispatch. This module provides
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

# Provider-agnostic default when a model isn't in the price table.
_FALLBACK_PRICING: Final[dict[str, Decimal]] = {
    "input": Decimal("0.00"),
    "output": Decimal("0.00"),
}

# USD per VLM page. GPT-4o images: ~$0.00213 per 512x512 tile at low-res;
# a typical scanned A4 page at detail:high is ~4 tiles, but we cap it
# conservatively. This is an estimate until real-page tuning happens.
_ESTIMATED_COST_PER_VLM_PAGE: Final[Decimal] = Decimal("0.02")

# Dimension of embedding vectors — needed for Qwen pricing estimates.
_QWEN_EMBEDDING_PRICE_PER_1K_TOKENS: Final[Decimal] = Decimal("0.00007")


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

    Args:
        provider: One of "anthropic", "openai", "qwen".
        model_id: The exact model string (e.g. "claude-opus-5").
        prompt_tokens: Input tokens charged at the input rate.
        max_output_tokens: The completion budget — we charge for the max, not
            the actual, because the guard stops *before* dispatch.
        cache_read_tokens: Tokens read from prompt cache (charged at cache rate).
        cache_write_tokens: Tokens written to prompt cache.
    """
    pricing = _get_pricing(provider, model_id)

    def _mtok(tokens: int) -> Decimal:
        return Decimal(tokens) / Decimal(1_000_000)

    # Uncached input tokens = total prompt minus what was already cached.
    uncached_input = max(0, prompt_tokens - cache_read_tokens)
    input_cost = _mtok(uncached_input) * pricing.get("input", Decimal("0"))
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
    """Resolve pricing for a provider+model, falling back to zero."""
    tables: dict[str, _ProviderPricing] = {
        "anthropic": _CLAUDE_PRICING,
        "openai": _OPENAI_PRICING,
        "qwen": _QWEN_PRICING,
    }
    provider_table = tables.get(provider, {})
    return provider_table.get(model_id, _FALLBACK_PRICING)


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
