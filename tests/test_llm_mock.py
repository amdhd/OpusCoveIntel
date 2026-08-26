"""Mock provider tests.

docs/plan.md Phase 5 acceptance: "mock provider drives the whole pipeline in CI."
These tests prove the mock is deterministic, returns plausible responses,
satisfies the provider interface, and — critically — costs $0.
"""

from __future__ import annotations

from app.llm.mock import MockChatResponse, MockLLMProvider


class TestMockProvider:
    @property
    def provider(self) -> MockLLMProvider:
        return MockLLMProvider()

    async def test_provider_name_is_mock(self) -> None:
        assert self.provider.provider_name == "mock"

    async def test_chat_returns_plausible_response(self) -> None:
        response = await self.provider.chat(
            model_id="claude-opus-5",
            system_prompt="You are a legal document extractor.",
            messages=[{"role": "user", "content": "Extract the gearing ratio."}],
        )
        assert isinstance(response, MockChatResponse)
        assert isinstance(response.content, str)
        assert "MOCK" in response.content
        assert response.model_id == "claude-opus-5"

    async def test_chat_is_deterministic(self) -> None:
        args = {
            "model_id": "claude-opus-5",
            "system_prompt": "Extract covenants.",
            "messages": [{"role": "user", "content": "Find cross-default thresholds."}],
        }
        first = await self.provider.chat(**args)  # type: ignore[arg-type]
        second = await self.provider.chat(**args)  # type: ignore[arg-type]
        assert first.content == second.content

    async def test_different_inputs_give_different_responses(self) -> None:
        a = await self.provider.chat(
            model_id="claude-opus-5",
            system_prompt="A",
            messages=[{"role": "user", "content": "X"}],
        )
        b = await self.provider.chat(
            model_id="claude-opus-5",
            system_prompt="B",
            messages=[{"role": "user", "content": "Y"}],
        )
        assert a.content != b.content

    async def test_chat_with_schema_returns_structured_json(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "threshold_amount": {"type": "number"},
                "threshold_currency": {"type": "string", "enum": ["MYR", "USD"]},
            },
            "required": ["threshold_amount", "threshold_currency"],
        }
        response = await self.provider.chat(
            model_id="claude-opus-5",
            system_prompt="Extract.",
            messages=[{"role": "user", "content": "RM30 million"}],
            response_schema=schema,
        )
        assert isinstance(response.content, dict)
        assert "threshold_amount" in response.content
        assert response.content["threshold_currency"] == "MYR"

    async def test_chat_returns_usage_stats(self) -> None:
        response = await self.provider.chat(
            model_id="claude-opus-5",
            system_prompt="Test.",
            messages=[{"role": "user", "content": "Hello world."}],
        )
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.cache_read_tokens == 0

    async def test_embed_is_imported_from_hashing_embedder(self) -> None:
        vectors = await self.provider.embed(
            ["gearing ratio covenant", "shariah compliance"],
            model_id="text-embedding-v4",
        )
        assert len(vectors) == 2
        assert all(len(v) == 1024 for v in vectors)
        # The hashing embedder is deterministic
        again = await self.provider.embed(
            ["gearing ratio covenant", "shariah compliance"],
            model_id="text-embedding-v4",
        )
        assert vectors == again

    async def test_vision_returns_ocr_like_response(self) -> None:
        image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # fake PNG header
        response = await self.provider.vision(
            model_id="gpt-4o",
            image_bytes=image,
            prompt="Transcribe this page.",
        )
        assert isinstance(response.content, str)
        assert "MOCK VLM" in response.content
        assert response.usage.prompt_tokens > 0

    async def test_vision_is_deterministic_for_same_image(self) -> None:
        image = bytes(200)
        a = await self.provider.vision(
            model_id="gpt-4o",
            image_bytes=image,
            prompt="OCR this.",
        )
        b = await self.provider.vision(
            model_id="gpt-4o",
            image_bytes=image,
            prompt="OCR this.",
        )
        assert a.content == b.content


class TestMockMinimalValidInstance:
    """The mock schema validator must produce structurally valid objects."""

    async def test_enum_property_uses_first_value(self) -> None:
        from app.llm.mock import _minimal_valid_instance

        obj = _minimal_valid_instance(
            {
                "type": "object",
                "properties": {
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            }
        )
        assert obj["severity"] == "low"

    async def test_nested_object(self) -> None:
        from app.llm.mock import _minimal_valid_instance

        obj = _minimal_valid_instance(
            {
                "type": "object",
                "properties": {
                    "threshold": {
                        "type": "object",
                        "properties": {
                            "amount": {"type": "number"},
                            "currency": {"type": "string"},
                        },
                    },
                },
            }
        )
        assert isinstance(obj["threshold"], dict)
        assert obj["threshold"]["amount"] == 0.0
