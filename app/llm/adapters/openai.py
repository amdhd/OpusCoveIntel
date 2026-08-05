"""OpenAI adapter: GPT-4o for vision (VLM OCR).

PLAN.md: scanned / low-confidence pages route to GPT-4o for OCR. This adapter
also supports regular chat for potential future use, but its primary role in
Phase 5 is the vision model behind the VLM fallback.

Uses httpx directly against the OpenAI chat completions API.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIAdapter:
    """Thin wrapper over the OpenAI Chat Completions API.

    Primary use in Phase 5: `vision()` for OCR of scanned/low-confidence pages.
    """

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key or (
            settings.OPENAI_API_KEY.get_secret_value() if settings.OPENAI_API_KEY else None
        )
        if not key:
            raise ValueError("OPENAI_API_KEY is not set")
        self._api_key = key
        self._client = httpx.AsyncClient(
            base_url=OPENAI_BASE_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0),
        )

    @property
    def provider_name(self) -> str:
        return "openai"

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
    ) -> OpenAIResponse:
        """Send a chat completion request."""
        openai_messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]
        for msg in messages:
            openai_messages.append({"role": msg["role"], "content": msg["content"]})

        body: dict[str, Any] = {
            "model": model_id,
            "messages": openai_messages,
            "max_tokens": max_tokens,
        }

        if response_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": response_schema,
                    "strict": True,
                },
            }

        logger.info(
            "openai.request",
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
                logger.warning("openai.invalid_json", extra={"raw": raw_content[:500]})

        return OpenAIResponse(
            content=content,
            model_id=data.get("model", model_id),
            usage=OpenAIUsage(
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            ),
        )

    async def vision(
        self,
        *,
        model_id: str,
        image_bytes: bytes,
        prompt: str,
        max_tokens: int = 4096,
    ) -> OpenAIResponse:
        """OCR a page image via GPT-4o vision.

        Sends the image as a base64-encoded data URL alongside the OCR prompt.
        """
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        image_url = f"data:image/png;base64,{image_b64}"

        body = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "high",  # needed for small text on A4 pages
                            },
                        },
                    ],
                }
            ],
            "max_tokens": max_tokens,
        }

        logger.info(
            "openai.vision.request",
            extra={
                "model_id": model_id,
                "image_bytes": len(image_bytes),
                "prompt_chars": len(prompt),
            },
        )

        response = await self._client.post("/chat/completions", json=body)
        response.raise_for_status()
        data = response.json()

        choice = data.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")

        return OpenAIResponse(
            content=content,
            model_id=data.get("model", model_id),
            usage=OpenAIUsage(
                prompt_tokens=data.get("usage", {}).get("prompt_tokens", 0),
                completion_tokens=data.get("usage", {}).get("completion_tokens", 0),
            ),
        )


class OpenAIResponse:
    def __init__(
        self,
        content: str | dict[str, Any],
        model_id: str,
        usage: OpenAIUsage,
    ) -> None:
        self.content = content
        self.model_id = model_id
        self.usage = usage


class OpenAIUsage:
    def __init__(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        # OpenAI doesn't have prompt caching in the same way
        self.cache_read_tokens: int = 0
        self.cache_write_tokens: int = 0
