"""Qwen adapter: chat (qwen-plus) and embeddings (text-embedding-v4).

CLAUDE.md routing table:
- Document classification, section detection: Qwen (qwen-plus) — high volume, low stakes
- Embeddings: Qwen text-embedding-v4, 1024 dims — strong multilingual (EN + BM)

Uses the DashScope international endpoint via OpenAI-compatible API.
httpx only — no DashScope SDK dependency.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class QwenAdapter:
    """Qwen provider via the OpenAI-compatible DashScope endpoint.

    Handles both chat (qwen-plus) and embeddings (text-embedding-v4).
    """

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key or (
            settings.QWEN_API_KEY.get_secret_value() if settings.QWEN_API_KEY else None
        )
        if not key:
            raise ValueError("QWEN_API_KEY is not set")
        self._api_key = key
        self._base_url = settings.QWEN_BASE_URL.rstrip("/")

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0),
        )

    @property
    def provider_name(self) -> str:
        return "qwen"

    async def close(self) -> None:
        await self._client.aclose()

    async def chat(
        self,
        *,
        model_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int = 4096,
        response_schema: dict[str, Any] | None = None,
    ) -> QwenResponse:
        """Send a chat completion.

        Args:
            model_id: e.g. "qwen-plus"
            system_prompt: System instruction.
            messages: Conversation messages.
            max_tokens: Completion budget.
            response_schema: Optional JSON Schema for structured output.
        """
        qwen_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        for msg in messages:
            qwen_messages.append({"role": msg["role"], "content": msg["content"]})

        body: dict[str, Any] = {
            "model": model_id,
            "messages": qwen_messages,
            "max_tokens": max_tokens,
        }

        if response_schema is not None:
            body["response_format"] = {
                "type": "json_object",
            }

        logger.info(
            "qwen.chat.request",
            extra={
                "model_id": model_id,
                "max_tokens": max_tokens,
                "system_chars": len(system_prompt),
                "message_count": len(messages),
                "has_schema": response_schema is not None,
            },
        )

        response = await self._client.post("/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()

        choice = data.get("choices", [{}])[0]
        raw_content = choice.get("message", {}).get("content", "")

        content: str | dict[str, Any] = raw_content
        if response_schema is not None and isinstance(raw_content, str):
            try:
                content = json.loads(raw_content)
            except json.JSONDecodeError:
                logger.warning("qwen.invalid_json", extra={"raw": raw_content[:500]})

        return QwenResponse(
            content=content,
            model_id=data.get("model", model_id),
            usage=QwenUsage(
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            ),
        )

    async def embed(self, texts: list[str], *, model_id: str) -> list[list[float]]:
        """Embed a batch of texts via Qwen text-embedding-v4.

        Returns vectors in input order, each of settings.VECTOR_DIMENSION
        (1024) dimensions.
        """
        settings = get_settings()

        body = {
            "model": model_id,
            "input": texts,
            "dimensions": settings.VECTOR_DIMENSION,
            "encoding_format": "float",
        }

        response = await self._client.post("/embeddings", json=body)
        response.raise_for_status()
        data = response.json()

        embeddings = data.get("data", [])

        # Sort by index to preserve input order
        embeddings.sort(key=lambda e: e.get("index", 0))

        vectors: list[list[float]] = [e.get("embedding", []) for e in embeddings]

        if len(vectors) != len(texts):
            raise RuntimeError(f"Qwen returned {len(vectors)} embeddings for {len(texts)} inputs")

        logger.info(
            "qwen.embed.response",
            extra={
                "model_id": model_id,
                "batch_size": len(texts),
                "dimensions": settings.VECTOR_DIMENSION,
            },
        )

        return vectors


class QwenResponse:
    def __init__(
        self,
        content: str | dict[str, Any],
        model_id: str,
        usage: QwenUsage,
    ) -> None:
        self.content = content
        self.model_id = model_id
        self.usage = usage


class QwenUsage:
    def __init__(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cache_read_tokens: int = 0
        self.cache_write_tokens: int = 0
