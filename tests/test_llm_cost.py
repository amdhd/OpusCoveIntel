"""Cost estimation tests.

Verify that pricing is correct, token counting is conservative, and Decimal
is used throughout (never float — CLAUDE.md 6).
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.llm.cost import (
    UnknownModelPricingError,
    count_tokens_estimate,
    estimate_cost,
    estimate_embedding_cost,
    estimate_vlm_page_cost,
    is_priced,
)


class TestTokenCounting:
    def test_empty_text_is_zero(self) -> None:
        assert count_tokens_estimate("") == 0
        assert count_tokens_estimate("   ") == 0

    def test_short_text_counts_tokens(self) -> None:
        n = count_tokens_estimate("hello world")
        assert n > 0

    def test_long_text_counts_more_tokens(self) -> None:
        short = count_tokens_estimate("hello world")
        long = count_tokens_estimate("hello world " * 100)
        assert long > short * 50  # approximately linear

    def test_returns_int(self) -> None:
        assert isinstance(count_tokens_estimate("hello world"), int)


class TestCostEstimation:
    def test_claude_opus_5_pricing(self) -> None:
        cost = estimate_cost(
            provider="anthropic",
            model_id="claude-opus-5",
            prompt_tokens=1_000_000,
            max_output_tokens=100_000,
        )
        # Input: $5/MTok → $5. Output: $25/MTok → $2.50. Total ~$7.50
        assert cost.input_cost > Decimal("4.50")
        assert cost.output_cost > Decimal("2.00")
        assert cost.total > Decimal("7.00")
        assert isinstance(cost.total, Decimal)

    def test_gpt_4o_pricing(self) -> None:
        cost = estimate_cost(
            provider="openai",
            model_id="gpt-4o",
            prompt_tokens=1_000_000,
            max_output_tokens=100_000,
        )
        # Input: $2.50/MTok, Output: $10/MTok → ~$3.50
        assert cost.total > Decimal("3.00")

    def test_qwen_plus_pricing(self) -> None:
        cost = estimate_cost(
            provider="qwen",
            model_id="qwen-plus",
            prompt_tokens=1_000_000,
            max_output_tokens=100_000,
        )
        # Input: $0.80/MTok, Output: $2.00/MTok → ~$1.00
        assert cost.total > Decimal("0.80")
        assert cost.total < Decimal("2.00")

    def test_cache_read_is_cheaper(self) -> None:
        """The same 10k-token prompt costs less when 9k of it comes from cache.

        The three input categories are disjoint (see `estimate_cost`), so the
        cached call splits the same 10k tokens rather than adding to them.
        """
        no_cache = estimate_cost(
            provider="anthropic",
            model_id="claude-opus-5",
            prompt_tokens=10_000,
            max_output_tokens=1_000,
        )
        with_cache = estimate_cost(
            provider="anthropic",
            model_id="claude-opus-5",
            prompt_tokens=1_000,
            max_output_tokens=1_000,
            cache_read_tokens=9_000,
        )
        assert with_cache.total < no_cache.total

    def test_zero_tokens_is_zero_cost(self) -> None:
        cost = estimate_cost(
            provider="anthropic",
            model_id="claude-opus-5",
            prompt_tokens=0,
            max_output_tokens=0,
        )
        assert cost.total == Decimal("0")

    def test_unknown_model_raises_rather_than_pricing_at_zero(self) -> None:
        """An unpriced model must fail closed, not cost $0.

        A zero here would pass every budget ceiling in docs/plan.md 2, so one typo
        in `EXTRACTION_MODEL` would disable the guards entirely.
        """
        with pytest.raises(UnknownModelPricingError):
            estimate_cost(
                provider="anthropic",
                model_id="nonexistent-model-v99",
                prompt_tokens=1_000_000,
                max_output_tokens=100_000,
            )

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(UnknownModelPricingError):
            estimate_cost(provider="acme", model_id="claude-opus-5", prompt_tokens=1_000)

    def test_mock_provider_is_priced_at_zero(self) -> None:
        """The CI provider is free because it makes no call, not by omission."""
        cost = estimate_cost(
            provider="mock",
            model_id="mock-v1",
            prompt_tokens=1_000_000,
            max_output_tokens=100_000,
        )
        assert cost.total == Decimal("0")
        assert is_priced("mock", "mock-v1")


class TestVLMPageCost:
    def test_is_positive(self) -> None:
        assert estimate_vlm_page_cost() > Decimal("0")

    def test_is_decimal(self) -> None:
        assert isinstance(estimate_vlm_page_cost(), Decimal)


class TestEmbeddingCost:
    def test_estimates_embedding_cost(self) -> None:
        cost = estimate_embedding_cost(1000)
        assert cost > Decimal("0")
        assert isinstance(cost, Decimal)
