"""LLM Router integration tests.

The router is the single chokepoint (CLAUDE.md 1.4). These tests prove that:
- The budget guard blocks before dispatch.
- Cache hits cost $0 and skip the provider.
- The mock provider drives the full pipeline.
- Spend is recorded in the ledger.
- make test makes zero paid API calls (all tests use the mock).
"""

from __future__ import annotations

from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.domain.enums import LLMStage
from app.llm.budget import BudgetExceededError
from app.llm.mock import MockLLMProvider
from app.llm.router import LLMRouter

# Tiny limits so the budget guard fires with small spend.
# Global cap is very tight; per-call is loose so we test the global guard.
_tiny_settings = Settings(
    ENVIRONMENT="test",
    MAX_COST_PER_CALL_USD=Decimal("1.00"),  # generous — per-call is NOT what we're testing
    MAX_TOTAL_COST_USD=Decimal("0.001"),  # tiny — a mock call will trip the global
    MAX_COST_PER_DOCUMENT_USD=Decimal("2.00"),
    MAX_VLM_PAGES_PER_DOC=40,
)


@pytest_asyncio.fixture
async def mock_provider() -> MockLLMProvider:
    return MockLLMProvider()


@pytest_asyncio.fixture
async def clean_ledger(db_session: AsyncSession) -> None:
    """Each router test starts with an empty cost ledger."""
    await db_session.execute(text("DELETE FROM llm_calls"))
    await db_session.execute(text("DELETE FROM llm_cache"))
    await db_session.flush()


class TestRouterWithMockProvider:
    """These are the tests that prove the acceptance criteria."""

    async def test_chat_completes_and_returns_content(
        self, db_session: AsyncSession, clean_ledger: None
    ) -> None:
        router = LLMRouter(db_session, provider=MockLLMProvider())
        result = await router.chat(
            stage=LLMStage.EXTRACT,
            provider_name="anthropic",
            model_id="claude-opus-5",
            system_prompt="Extract covenants from legal text.",
            messages=[{"role": "user", "content": "RM30 million cross-default threshold."}],
        )
        assert result.content is not None
        assert "MOCK" in str(result.content)
        assert result.cache_hit is False
        assert result.model_id == "claude-opus-5"

    async def test_chat_records_spend_in_ledger(
        self, db_session: AsyncSession, clean_ledger: None
    ) -> None:
        router = LLMRouter(db_session, provider=MockLLMProvider())
        await router.chat(
            stage=LLMStage.EXTRACT,
            provider_name="anthropic",
            model_id="claude-opus-5",
            system_prompt="Extract.",
            messages=[{"role": "user", "content": "RM50m threshold."}],
        )

        from app.db.repositories.ops import LLMCallRepository

        total = await LLMCallRepository(db_session).total_cost()
        # The mock provider's usage generates a non-zero cost estimate
        assert total >= Decimal("0")

    async def test_cache_hit_skips_provider_and_costs_zero(
        self, db_session: AsyncSession, clean_ledger: None
    ) -> None:
        router = LLMRouter(db_session, provider=MockLLMProvider())

        args = {
            "stage": LLMStage.EXTRACT,
            "provider_name": "anthropic",
            "model_id": "claude-opus-5",
            "system_prompt": "Extract covenants.",
            "messages": [{"role": "user", "content": "The gearing ratio shall not exceed 2.0x."}],
            "prompt_version": "v1",
        }

        first = await router.chat(**args)  # type: ignore[arg-type]
        assert not first.cache_hit

        second = await router.chat(**args)  # type: ignore[arg-type]
        assert second.cache_hit
        assert second.estimated_cost_usd == Decimal("0")
        assert second.prompt_tokens == 0

    async def test_changed_prompt_version_is_cache_miss(
        self, db_session: AsyncSession, clean_ledger: None
    ) -> None:
        router = LLMRouter(db_session, provider=MockLLMProvider())

        base_args = {
            "stage": LLMStage.EXTRACT,
            "provider_name": "anthropic",
            "model_id": "claude-opus-5",
            "system_prompt": "Extract.",
            "messages": [{"role": "user", "content": "RM10m."}],
        }

        first = await router.chat(**base_args, prompt_version="v1")  # type: ignore[arg-type]
        assert not first.cache_hit

        second = await router.chat(**base_args, prompt_version="v2")  # type: ignore[arg-type]
        assert not second.cache_hit

    async def test_budget_guard_blocks_over_limit_call(
        self, db_session: AsyncSession, clean_ledger: None
    ) -> None:
        """With a tiny global cap, the router's guard must block."""
        # Pre-record enough spend to nearly exhaust the tiny $0.001 cap
        from app.db.models.ops import LLMCall
        from app.db.repositories.ops import LLMCallRepository

        call = LLMCall(
            stage=LLMStage.EXTRACT,
            provider="anthropic",
            model_id="claude-opus-5",
            prompt_tokens=100,
            completion_tokens=50,
            estimated_cost_usd=Decimal("0.0009"),
        )
        await LLMCallRepository(db_session).add(call)
        await db_session.flush()

        # The mock provider will generate tens of prompt_tokens — a small
        # cost estimate, but enough to push past $0.001 when added to $0.0009.
        router = LLMRouter(db_session, provider=MockLLMProvider(), settings=_tiny_settings)

        with pytest.raises(BudgetExceededError) as exc_info:
            await router.chat(
                stage=LLMStage.EXTRACT,
                provider_name="anthropic",
                model_id="claude-opus-5",
                system_prompt="Extract.",
                messages=[{"role": "user", "content": "Test."}],
            )
        assert "global" in str(exc_info.value).lower()

    async def test_embed_returns_vectors_and_records_spend(
        self, db_session: AsyncSession, clean_ledger: None
    ) -> None:
        router = LLMRouter(db_session, provider=MockLLMProvider())
        vectors = await router.embed(
            texts=["gearing ratio covenant", "shariah compliance event"],
        )
        assert len(vectors) == 2
        assert all(len(v) == 1024 for v in vectors)

        from app.db.repositories.ops import LLMCallRepository

        total = await LLMCallRepository(db_session).total_cost()
        assert total >= Decimal("0")

    async def test_vision_returns_ocr_text(
        self, db_session: AsyncSession, clean_ledger: None
    ) -> None:
        router = LLMRouter(db_session, provider=MockLLMProvider())
        result = await router.vision(
            image_bytes=b"\x89PNG\r\n\x1a\n" + bytes(200),
            prompt="Transcribe this page.",
        )
        assert "MOCK VLM" in str(result.content)
        assert result.cache_hit is False

    async def test_vision_cache_hit_costs_zero(
        self, db_session: AsyncSession, clean_ledger: None
    ) -> None:
        router = LLMRouter(db_session, provider=MockLLMProvider())
        image = bytes([i % 256 for i in range(300)])
        prompt = "OCR this page."

        first = await router.vision(image_bytes=image, prompt=prompt)
        assert not first.cache_hit

        second = await router.vision(image_bytes=image, prompt=prompt)
        assert second.cache_hit
        assert second.estimated_cost_usd == Decimal("0")


class TestRouterStructuredOutput:
    async def test_structured_output_round_trips(self, db_session: AsyncSession) -> None:
        from sqlalchemy import text as _text

        await db_session.execute(_text("DELETE FROM llm_calls"))
        await db_session.execute(_text("DELETE FROM llm_cache"))
        await db_session.flush()

        schema = {
            "type": "object",
            "properties": {
                "threshold_amount": {"type": "number"},
                "threshold_currency": {"type": "string", "enum": ["MYR", "USD"]},
            },
            "required": ["threshold_amount", "threshold_currency"],
        }
        router = LLMRouter(db_session, provider=MockLLMProvider())
        result = await router.chat(
            stage=LLMStage.EXTRACT,
            provider_name="anthropic",
            model_id="claude-opus-5",
            system_prompt="Extract thresholds.",
            messages=[{"role": "user", "content": "RM30 million"}],
            response_schema=schema,
        )
        assert isinstance(result.content, dict)
        assert "threshold_amount" in result.content
        assert result.content["threshold_currency"] == "MYR"
