"""Anthropic adapter: claude-opus-5 via the Messages API.

CLAUDE.md 2 has the verified API rules:
- Model ID is `claude-opus-5` — no date suffix.
- temperature, top_p, top_k are rejected (400). Steer via prompt.
- thinking: {"type": "adaptive"} — control depth with output_config.effort.
- Assistant-turn prefills return 400 — use structured outputs instead.
- max_tokens caps thinking + response together — extraction ≥8000.
- citations: {enabled: true} incompatible with output_config.format (400).
- cache_control: {"type": "ephemeral"} on the system+schema+few-shot prefix.
  Minimum cacheable prefix on claude-opus-5 is 512 tokens.

Uses httpx (already a project dependency) rather than the anthropic SDK,
since the API is a thin JSON-over-HTTP layer.

CLAUDE.md 1.4: this module is the ONLY place anthropic.* API calls originate.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.adapters._http import post_with_retry

logger = get_logger(__name__)

ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


class AnthropicAdapter:
    """Thin wrapper over the Anthropic Messages API.

    Does NOT do budget checks, caching, or cost tracking — the router does
    those. This adapter only speaks the Anthropic HTTP protocol.
    """

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        key = api_key or (
            settings.ANTHROPIC_API_KEY.get_secret_value() if settings.ANTHROPIC_API_KEY else None
        )
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self._api_key = key
        self._provider_name = "anthropic"
        self._client = httpx.AsyncClient(
            base_url=ANTHROPIC_BASE_URL,
            headers={
                "x-api-key": self._api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(120.0),
        )

    @property
    def provider_name(self) -> str:
        return self._provider_name

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
        effort: str | None = None,
        enable_prompt_caching: bool = False,
    ) -> AnthropicResponse:
        """Send a chat completion request.

        Args:
            model_id: e.g. "claude-opus-5"
            system_prompt: The system-level instruction (cached if enable_prompt_caching).
            messages: List of {"role": "user"|"assistant", "content": "..."}
            max_tokens: Must be ≥8000 for extraction calls.
            response_schema: If provided, the response is constrained to this
                JSON Schema via output_config.format. Mutually exclusive with
                citations — we carry citations as schema fields.
            effort: "low" | "medium" | "high" | "xhigh" | "max". Controls
                thinking depth. Default is "medium".
            enable_prompt_caching: Adds cache_control to the system prompt block.
        """
        body: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "messages": messages,
        }

        # System prompt with optional prompt caching
        system_block: dict[str, Any] = {
            "type": "text",
            "text": system_prompt,
        }
        if enable_prompt_caching:
            system_block["cache_control"] = {"type": "ephemeral"}
        body["system"] = [system_block]

        # Structured output — mutually exclusive with citations per CLAUDE.md 2
        if response_schema is not None:
            body["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": response_schema,
                },
            }

        # Thinking / effort control
        if effort is not None:
            if "output_config" not in body:
                body["output_config"] = {}
            body["output_config"]["effort"] = effort

        logger.info(
            "anthropic.request",
            extra={
                "model_id": model_id,
                "max_tokens": max_tokens,
                "system_chars": len(system_prompt),
                "message_count": len(messages),
                "caching": enable_prompt_caching,
                "has_schema": response_schema is not None,
                "effort": effort,
            },
        )

        response = await post_with_retry(self._client, "/messages", body, provider="anthropic")
        data = response.json()

        content = _extract_content(data, expects_json=response_schema is not None)

        usage = AnthropicUsage(
            prompt_tokens=data.get("usage", {}).get("input_tokens", 0),
            completion_tokens=data.get("usage", {}).get("output_tokens", 0),
            cache_read_tokens=data.get("usage", {}).get("cache_read_input_tokens", 0),
            cache_write_tokens=data.get("usage", {}).get("cache_creation_input_tokens", 0),
        )

        logger.info(
            "anthropic.response",
            extra={
                "model_id": model_id,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cache_read_tokens": usage.cache_read_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
            },
        )

        return AnthropicResponse(
            content=content,
            model_id=data.get("model", model_id),
            usage=usage,
        )


class AnthropicResponse:
    """Uniform response shape, matching MockChatResponse."""

    def __init__(
        self,
        content: str | dict[str, Any],
        model_id: str,
        usage: AnthropicUsage,
    ) -> None:
        self.content = content
        self.model_id = model_id
        self.usage = usage


class AnthropicUsage:
    def __init__(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_write_tokens = cache_write_tokens


def _extract_content(data: dict[str, Any], *, expects_json: bool = False) -> str | dict[str, Any]:
    """Pull text or structured JSON from the Messages response.

    With `output_config.format` the model does *not* return a tool_use block --
    it returns ordinary text blocks whose content is the constrained JSON. The
    tool_use branch is kept because a tool-calling request still produces one,
    but when a schema was requested the text is parsed. Skipping that parse is
    what used to make every structured extraction look like "the model returned
    prose", costing one wasted retry per candidate before landing in review.
    """
    content_blocks = data.get("content", [])

    # Tool-calling shape: the arguments object is already parsed JSON.
    for block in content_blocks:
        if block.get("type") == "tool_use":
            return block.get("input", {})  # type: ignore[no-any-return]

    text_blocks = [block.get("text", "") for block in content_blocks if block.get("type") == "text"]
    text = "\n".join(text_blocks)

    if expects_json and text.strip():
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Hand the raw text back so the caller's validation path reports
            # "not valid JSON" with the actual response attached.
            logger.warning("anthropic.structured_output_not_json", extra={"raw": text[:500]})
            return text
        if isinstance(parsed, dict):
            return parsed
        logger.warning(
            "anthropic.structured_output_not_an_object",
            extra={"json_type": type(parsed).__name__},
        )
        return text

    return text
