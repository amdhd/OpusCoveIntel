"""Central LLM dispatch — budget guard → cache → adapter.

CLAUDE.md 1.4: "No silent LLM spend. Every model call goes through
app/llm/router.py, which enforces the budget guard and cache. Direct
anthropic.Anthropic() / openai.OpenAI() calls outside app/llm/adapters/
are forbidden."

This is the single chokepoint that enforces that invariant. Every call:
1. Checks the budget guard (reject before dispatch).
2. Checks the response cache (cache hit → $0).
3. Routes to the correct adapter.
4. Records the spend in llm_calls.
5. Stores the response in the cache.

Phase 5 ordering (PLAN.md): guards land before adapters. The router can be
constructed with a MockLLMProvider for CI — same interface, zero spend.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.enums import LLMStage
from app.llm.budget import BudgetDecision, BudgetExceededError, BudgetGuard
from app.llm.cache import ResponseCache, build_cache_key
from app.llm.cost import TokenCost, estimate_cost

logger = get_logger(__name__)


class LLMProvider(Protocol):
    """The interface every provider (real or mock) must satisfy.

    Narrow by design: the router only needs chat, embed, and vision.
    """

    @property
    def provider_name(self) -> str: ...

    async def chat(
        self,
        *,
        model_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        response_schema: dict[str, Any] | None = None,
    ) -> Any: ...

    async def embed(self, texts: list[str], *, model_id: str) -> list[list[float]]: ...

    async def vision(
        self,
        *,
        model_id: str,
        image_bytes: bytes,
        prompt: str,
        max_tokens: int = 4096,
    ) -> Any: ...


@dataclass
class LLMCallResult:
    """Everything the caller needs after dispatch."""

    content: str | dict[str, Any]
    model_id: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    estimated_cost_usd: Decimal
    cache_hit: bool
    budget_decision: BudgetDecision | None = None
    call_id: uuid.UUID | None = None


class LLMRouter:
    """The single chokepoint for all LLM calls.

    Usage:
        router = LLMRouter(session)
        result = await router.chat(
            stage=LLMStage.EXTRACT,
            provider_name="anthropic",
            model_id="claude-opus-5",
            system_prompt="Extract covenants from the following clause...",
            messages=[{"role": "user", "content": "..."}],
            prompt_version="v1",
            document_id=doc_id,
        )
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        provider: LLMProvider | None = None,
        settings: object | None = None,
    ) -> None:
        self._session = session
        self._budget = BudgetGuard(session, settings=settings)
        self._cache = ResponseCache(session)
        if settings is not None:
            self._settings = settings
        else:
            self._settings = get_settings()
        # Provider is injected — real in prod, mock in CI.
        self._provider = provider

    async def chat(
        self,
        *,
        stage: LLMStage,
        provider_name: str,
        model_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        response_schema: dict[str, Any] | None = None,
        prompt_version: str = "v0",
        document_id: uuid.UUID | None = None,
        enable_prompt_caching: bool = False,
    ) -> LLMCallResult:
        """Chat completion through the full guard → cache → adapter pipeline."""

        # 1. Estimate cost
        estimated_prompt_tokens = _estimate_prompt_tokens(system_prompt, messages)
        cost = estimate_cost(
            provider=provider_name,
            model_id=model_id,
            prompt_tokens=estimated_prompt_tokens,
            max_output_tokens=max_tokens,
        )

        # 2. Budget guard
        decision = await self._budget.check_call(
            estimated_cost=cost.total,
            document_id=document_id,
        )
        if not decision.allowed:
            self._record_rejected_call(
                stage=stage,
                provider_name=provider_name,
                model_id=model_id,
                document_id=document_id,
                decision=decision,
            )
            raise BudgetExceededError(decision)

        # 3. Cache check
        content_hash = _content_hash(system_prompt, messages, response_schema)
        cache_key = build_cache_key(
            prompt_version=prompt_version,
            model_id=model_id,
            content=content_hash,
        )
        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.info(
                "llm.cache_hit",
                extra={
                    "stage": stage.value,
                    "provider": provider_name,
                    "model_id": model_id,
                    "cache_key": cache_key[:16],
                },
            )
            return LLMCallResult(
                content=cached,
                model_id=model_id,
                provider=provider_name,
                prompt_tokens=0,
                completion_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                estimated_cost_usd=Decimal("0"),
                cache_hit=True,
                budget_decision=decision,
            )

        # 4. Dispatch to provider
        provider = await self._resolve_provider(provider_name)
        response = await provider.chat(
            model_id=model_id,
            system_prompt=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            response_schema=response_schema,
        )

        # 5. Record spend
        actual_cost = estimate_cost(
            provider=provider_name,
            model_id=model_id,
            prompt_tokens=response.usage.prompt_tokens,
            max_output_tokens=response.usage.completion_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_tokens", 0),
            cache_write_tokens=getattr(response.usage, "cache_write_tokens", 0),
        )

        call_id = await self._record_call(
            stage=stage,
            provider_name=provider_name,
            model_id=model_id,
            document_id=document_id,
            usage=response.usage,
            cost=actual_cost,
            cache_hit=False,
        )

        # 6. Store in cache
        response_content: dict[str, Any]
        if isinstance(response.content, dict):
            response_content = response.content
        else:
            response_content = {"text": response.content}

        await self._cache.put(
            cache_key=cache_key,
            prompt_hash=content_hash,
            model_id=model_id,
            prompt_version=prompt_version,
            response_json=response_content,
            estimated_cost_usd=actual_cost.total,
        )

        return LLMCallResult(
            content=response.content,
            model_id=model_id,
            provider=provider_name,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_tokens", 0),
            cache_write_tokens=getattr(response.usage, "cache_write_tokens", 0),
            estimated_cost_usd=actual_cost.total,
            cache_hit=False,
            budget_decision=decision,
            call_id=call_id,
        )

    async def embed(
        self,
        *,
        texts: list[str],
        provider_name: str = "qwen",
        model_id: str | None = None,
        document_id: uuid.UUID | None = None,
    ) -> list[list[float]]:
        """Embed texts through the guard → adapter pipeline."""
        settings = get_settings()
        model = model_id or settings.EMBEDDING_MODEL

        # Estimate token count and cost
        estimated_tokens = sum(len(t.split()) for t in texts) * 2  # rough: ~2 tokens/word
        cost = estimate_cost(
            provider=provider_name,
            model_id=model,
            prompt_tokens=estimated_tokens,
            max_output_tokens=0,
        )

        # Budget guard
        decision = await self._budget.check_call(
            estimated_cost=cost.total,
            document_id=document_id,
        )
        if not decision.allowed:
            raise BudgetExceededError(decision)

        # Embeddings are not cached — they're deterministic for the same text
        # and the retrieval layer handles model-switching via embedding_model.

        provider = await self._resolve_provider(provider_name)
        vectors = await provider.embed(texts, model_id=model)

        # Record spend
        await self._record_call(
            stage=LLMStage.EMBED,
            provider_name=provider_name,
            model_id=model,
            document_id=document_id,
            usage=_EmbedUsage(prompt_tokens=estimated_tokens, completion_tokens=0),
            cost=cost,
            cache_hit=False,
        )

        return vectors

    async def vision(
        self,
        *,
        stage: LLMStage = LLMStage.VLM_OCR,
        model_id: str | None = None,
        image_bytes: bytes,
        prompt: str,
        max_tokens: int = 4096,
        prompt_version: str = "v0",
        document_id: uuid.UUID | None = None,
    ) -> LLMCallResult:
        """Vision (VLM OCR) through the guard → cache → adapter pipeline."""
        settings = get_settings()
        model = model_id or settings.VLM_MODEL

        # Estimate cost — VLM is priced per image
        cost = estimate_cost(
            provider="openai",
            model_id=model,
            prompt_tokens=len(prompt.split()) + 500,  # rough image token estimate
            max_output_tokens=max_tokens,
        )

        # Budget guard
        decision = await self._budget.check_call(
            estimated_cost=cost.total,
            document_id=document_id,
        )
        if not decision.allowed:
            self._record_rejected_call(
                stage=stage,
                provider_name="openai",
                model_id=model,
                document_id=document_id,
                decision=decision,
            )
            raise BudgetExceededError(decision)

        # Cache check — VLM responses are cacheable by image bytes
        from app.llm.cache import build_cache_key_bytes

        cache_key = build_cache_key_bytes(
            prompt_version=prompt_version,
            model_id=model,
            content_bytes=image_bytes + prompt.encode("utf-8"),
        )
        cached = await self._cache.get(cache_key)
        if cached is not None:
            logger.info(
                "llm.vision_cache_hit",
                extra={
                    "model_id": model,
                    "image_bytes": len(image_bytes),
                },
            )
            return LLMCallResult(
                content=cached,
                model_id=model,
                provider="openai",
                prompt_tokens=0,
                completion_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
                estimated_cost_usd=Decimal("0"),
                cache_hit=True,
                budget_decision=decision,
            )

        # Dispatch
        provider = await self._resolve_provider("openai")
        response = await provider.vision(
            model_id=model,
            image_bytes=image_bytes,
            prompt=prompt,
            max_tokens=max_tokens,
        )

        # Record
        actual_cost = estimate_cost(
            provider="openai",
            model_id=model,
            prompt_tokens=response.usage.prompt_tokens,
            max_output_tokens=response.usage.completion_tokens,
        )

        call_id = await self._record_call(
            stage=stage,
            provider_name="openai",
            model_id=model,
            document_id=document_id,
            usage=response.usage,
            cost=actual_cost,
            cache_hit=False,
        )

        # Cache
        response_content: dict[str, Any]
        if isinstance(response.content, dict):
            response_content = response.content
        else:
            response_content = {"text": response.content}

        await self._cache.put(
            cache_key=cache_key,
            prompt_hash=hashlib.sha256(image_bytes).hexdigest(),
            model_id=model,
            prompt_version=prompt_version,
            response_json=response_content,
            estimated_cost_usd=actual_cost.total,
        )

        return LLMCallResult(
            content=response.content,
            model_id=model,
            provider="openai",
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cache_read_tokens=0,
            cache_write_tokens=0,
            estimated_cost_usd=actual_cost.total,
            cache_hit=False,
            budget_decision=decision,
            call_id=call_id,
        )

    # -- internals -----------------------------------------------------------

    async def _resolve_provider(self, provider_name: str) -> LLMProvider:
        """Return the injected provider, or lazily construct a real one."""
        if self._provider is not None:
            return self._provider

        # Lazy import so real adapter SDKs are only imported when used.
        if provider_name == "anthropic":
            from app.llm.adapters.anthropic import AnthropicAdapter

            return AnthropicAdapter()  # type: ignore[return-value]
        elif provider_name == "openai":
            from app.llm.adapters.openai import OpenAIAdapter

            return OpenAIAdapter()  # type: ignore[return-value]
        elif provider_name == "qwen":
            from app.llm.adapters.qwen import QwenAdapter

            return QwenAdapter()  # type: ignore[return-value]
        else:
            raise ValueError(f"unknown provider: {provider_name}")

    async def _record_call(
        self,
        *,
        stage: LLMStage,
        provider_name: str,
        model_id: str,
        document_id: uuid.UUID | None,
        usage: Any,
        cost: TokenCost,
        cache_hit: bool,
    ) -> uuid.UUID:
        """Write an LLMCall row to the cost ledger."""

        from app.db.models.ops import LLMCall
        from app.db.repositories.ops import LLMCallRepository

        call = LLMCall(
            document_id=document_id,
            stage=stage,
            provider=provider_name,
            model_id=model_id,
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            cache_read_tokens=getattr(usage, "cache_read_tokens", 0),
            cache_write_tokens=getattr(usage, "cache_write_tokens", 0),
            estimated_cost_usd=cost.total,
            cache_hit=cache_hit,
        )
        await LLMCallRepository(self._session).add(call)
        return call.id

    def _record_rejected_call(
        self,
        *,
        stage: LLMStage,
        provider_name: str,
        model_id: str,
        document_id: uuid.UUID | None,
        decision: BudgetDecision,
    ) -> None:
        """Log a rejected call. No DB row — nothing was dispatched."""
        logger.warning(
            "llm.call_rejected",
            extra={
                "stage": stage.value,
                "provider": provider_name,
                "model_id": model_id,
                "document_id": str(document_id) if document_id else None,
                "outcome": decision.outcome.value,
                "reason": decision.reason,
            },
        )


def _estimate_prompt_tokens(
    system_prompt: str,
    messages: list[dict[str, str]],
) -> int:
    """Rough token count for budget estimation."""
    from app.llm.cost import count_tokens_estimate

    total = count_tokens_estimate(system_prompt)
    for msg in messages:
        total += count_tokens_estimate(msg.get("content", ""))
    return total


def _content_hash(
    system_prompt: str,
    messages: list[dict[str, str]],
    response_schema: dict[str, Any] | None,
) -> str:
    """Deterministic hash of the full request content."""
    material = system_prompt + json.dumps(messages, sort_keys=True)
    if response_schema is not None:
        material += json.dumps(response_schema, sort_keys=True)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class _EmbedUsage:
    """Minimal usage for embedding calls."""

    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cache_read_tokens = 0
        self.cache_write_tokens = 0
