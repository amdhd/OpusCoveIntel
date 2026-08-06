"""Mock LLM provider — drives the full pipeline in CI for $0.

PLAN.md Phase 5 acceptance: "mock provider drives the whole pipeline in CI"
and "make test makes zero paid API calls."

This is not a stub that returns noise. It returns deterministic, plausible
responses keyed on the input content, so:
- The extraction pipeline can run end-to-end in tests.
- The query agent can answer golden questions without a real model.
- CI stays fast and free.

For extraction calls, it returns structured JSON matching the expected schema.
For chat calls, it returns a plausible response echoing the system prompt role.
For embedding calls, it delegates to HashingEmbedder — already tested and free.

Usage in tests:
    from app.llm.mock import MockLLMProvider
    mock = MockLLMProvider()
    router = LLMRouter(session, provider=mock)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

from app.core.logging import get_logger
from app.llm.embeddings import HashingEmbedder

logger = get_logger(__name__)

MOCK_MODEL_ID = "mock-v1"


class MockLLMProvider:
    """A provider that returns deterministic, free responses.

    Satisfies the same interface as real adapters so the router doesn't know
    the difference. Zero API calls, zero cost, zero latency.
    """

    @property
    def provider_name(self) -> str:
        return "mock"

    async def chat(
        self,
        *,
        model_id: str,
        system_prompt: str,
        messages: list[dict[str, str]],
        max_tokens: int = 1024,
        response_schema: dict[str, Any] | None = None,
        **provider_options: Any,
    ) -> MockChatResponse:
        """Return a deterministic response keyed on the system prompt + messages.

        When a `response_schema` is provided, returns a minimal valid instance
        of that schema. Otherwise returns a plausible text response.

        `provider_options` (Anthropic's `effort`, `enable_prompt_caching`) are
        accepted and ignored: the mock has no cache and no thinking budget, but
        refusing them would make the router's forwarding untestable in CI.
        """
        content_hash = _hash(system_prompt + json.dumps(messages, sort_keys=True))

        content: str | dict[str, Any]
        if response_schema is not None:
            content = _minimal_valid_instance(response_schema)
            if isinstance(content, dict):
                _ground_in_input(content, response_schema, messages)
        else:
            content = (
                f"[MOCK RESPONSE — model={model_id}, hash={content_hash[:12]}]\n"
                f"This is a deterministic mock. The real {model_id} would answer "
                f"based on the provided system prompt and messages."
            )

        return MockChatResponse(
            content=content,
            model_id=model_id,
            usage=MockUsage(
                prompt_tokens=_est_tokens(system_prompt, messages),
                completion_tokens=_est_output_tokens(content),
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
        )

    async def embed(self, texts: list[str], *, model_id: str) -> list[list[float]]:
        """Delegate to HashingEmbedder for deterministic free vectors."""
        embedder = HashingEmbedder()
        return await embedder.embed(texts)

    async def vision(
        self,
        *,
        model_id: str,
        image_bytes: bytes,
        prompt: str,
        max_tokens: int = 1024,
    ) -> MockChatResponse:
        """Return a deterministic OCR-like response for VLM testing."""
        img_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        content = (
            f"[MOCK VLM OCR — model={model_id}, image_hash={img_hash}]\n"
            f"Prompt: {prompt[:200]}\n"
            f"The real VLM would transcribe text from the page image."
        )
        return MockChatResponse(
            content=content,
            model_id=model_id,
            usage=MockUsage(
                prompt_tokens=len(prompt.split()) + 200,  # rough image token estimate
                completion_tokens=len(content.split()),
                cache_read_tokens=0,
                cache_write_tokens=0,
            ),
        )


class MockChatResponse:
    """The shape every adapter returns, so the router can treat them uniformly."""

    def __init__(
        self,
        content: str | dict[str, Any],
        model_id: str,
        usage: MockUsage,
    ) -> None:
        self.content = content
        self.model_id = model_id
        self.usage = usage


class MockUsage:
    """Token counts that the cost tracker and ledger consume."""

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


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _est_tokens(system_prompt: str, messages: list[dict[str, str]]) -> int:
    """Rough token count for mock usage stats."""
    return len(system_prompt.split()) + sum(len(json.dumps(m).split()) for m in messages)


def _est_output_tokens(content: str | dict[str, Any]) -> int:
    """Rough token count for mock output stats."""
    if isinstance(content, dict):
        return len(json.dumps(content).split())
    return len(content.split())


# Fields a mock response must take verbatim from its input rather than invent.
# A citation is verified against the source chunk before anything is persisted
# (CLAUDE.md 1.3), so a placeholder here fails that check every time -- which
# meant the mock could never exercise the extraction *success* path, only the
# review-queue path, and PLAN.md Phase 5's "mock provider drives the whole
# pipeline in CI" was true only of its unhappy half.
_QUOTE_FIELDS: Final[tuple[str, ...]] = ("source_quote",)

# Long enough to clear the fuzzy leg's minimum-quote floor, short enough to be
# a slice of any realistic candidate span.
_MOCK_QUOTE_CHARS: Final[int] = 120

# Optional fields that gate a whole branch of the pipeline, so leaving them
# None (the correct default for anything `anyOf [T, null]`) makes that branch
# unreachable in CI. `covenant_type` is the gate on covenant persistence: while
# it was None, no mock run ever wrote a covenant, and everything downstream of
# that -- instrument linking, threshold review triggers, the high-value cap --
# went untested against the mock that supposedly drives the whole pipeline.
_FORCED_ENUM_FIELDS: Final[tuple[str, ...]] = ("covenant_type",)


def _ground_in_input(
    instance: dict[str, Any],
    schema: dict[str, Any],
    messages: list[dict[str, str]],
) -> None:
    """Replace placeholder quote fields with a verbatim slice of the input.

    Deterministic: always the first `_MOCK_QUOTE_CHARS` characters of the last
    user message, trimmed to a whitespace boundary so the slice is a run of
    whole words. That makes the mock's citation genuinely verifiable against
    the chunk the candidate came from.
    """
    properties = schema.get("properties", {})

    # A response whose payload is a list of objects (the extraction schema is
    # `{"covenants": [...]}`) is grounded element-wise. `_minimal_valid_instance`
    # builds an empty list for an array, which validates but exercises nothing:
    # the pipeline would see "no covenants here" for every span and CI would
    # never reach covenant persistence at all.
    for name, prop in properties.items():
        if prop.get("type") != "array":
            continue
        item_schema = prop.get("items")
        if not isinstance(item_schema, dict) or "properties" not in item_schema:
            continue
        element = _minimal_valid_instance(item_schema)
        if isinstance(element, dict):
            _ground_in_input(element, item_schema, messages)
            instance[name] = [element]

    for field in _FORCED_ENUM_FIELDS:
        if field in properties and instance.get(field) is None:
            choice = _first_enum_value(properties[field])
            if choice is not None:
                instance[field] = choice

    source = _last_user_content(messages)
    if not source:
        return

    quote = _leading_slice(source, _MOCK_QUOTE_CHARS)
    if not quote:
        return

    for field in _QUOTE_FIELDS:
        if field in properties and field in instance:
            instance[field] = quote


def _first_enum_value(prop: dict[str, Any]) -> Any:
    """The first enum member of a property, looking through `anyOf`."""
    if "enum" in prop:
        return prop["enum"][0]
    for branch in prop.get("anyOf", ()):
        if "enum" in branch:
            return branch["enum"][0]
    return None


def _last_user_content(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _leading_slice(text: str, limit: int) -> str:
    """A whole-word prefix of `text`, at most `limit` characters.

    The candidate text arrives inside a framing sentence from
    `build_user_message`; the blank line separates it, so anything after the
    first one is the document's own words and is what a citation should quote.
    """
    body = text.split("\n\n", 1)[-1].strip()
    if len(body) <= limit:
        return body
    cut = body.rfind(" ", 0, limit)
    return body[: cut if cut > 0 else limit].strip()


def _minimal_valid_instance(schema: dict[str, Any]) -> Any:
    """Build a minimal object that satisfies a JSON Schema.

    This is not a full schema validator — it handles the patterns our
    extraction schemas use: objects with string/number/boolean/array
    properties, enum constraints, anyOf (for Optional fields), and $refs.
    It returns the first valid value for each property so the pipeline can
    flow through Pydantic validation.

    The returned object is deliberately minimal — the point is structural
    validity, not correctness. The rules extractor provides the correctness
    baseline; the mock shows that the LLM path *would* have run.
    """
    if "properties" in schema:
        obj: dict[str, Any] = {}
        for prop_name, prop_schema in schema["properties"].items():
            obj[prop_name] = _mock_value(prop_schema)
        return obj

    if "items" in schema:
        return [_mock_value(schema["items"])]

    if "enum" in schema:
        return schema["enum"][0]

    if "anyOf" in schema:
        # anyOf [T, {"type": "null"}] → Optional[T] → return None.
        non_null = [s for s in schema["anyOf"] if s.get("type") != "null"]
        if non_null and len(non_null) < len(schema["anyOf"]):
            return None  # It's optional — None is valid.
        if non_null:
            return _mock_value(non_null[0])

    if "type" in schema:
        type_name = schema["type"]
        if isinstance(type_name, list):
            # e.g. ["number", "null"] → return 0.0
            type_name = next((t for t in type_name if t != "null"), "string")
        type_map = {
            "string": "[MOCK]",
            "number": 0.0,
            "integer": 0,
            "boolean": False,
            "array": [],
            "object": {},
        }
        return type_map.get(type_name, "[MOCK]")

    return "[MOCK]"


def _mock_value(prop: dict[str, Any]) -> Any:
    """Return a type-appropriate mock value for a single schema property."""
    if "anyOf" in prop:
        # Optional[T] represented as anyOf: [T, {type: "null"}]
        non_null = [s for s in prop["anyOf"] if s.get("type") != "null"]
        if non_null and len(non_null) < len(prop["anyOf"]):
            return None  # Optional — return None for structural validity.
        if non_null:
            return _mock_value(non_null[0])

    if "enum" in prop:
        return prop["enum"][0]
    if "const" in prop:
        return prop["const"]

    type_name = prop.get("type", "string")
    if isinstance(type_name, list):
        type_name = next((t for t in type_name if t != "null"), "string")

    if type_name == "string":
        if "format" in prop:
            format_map: dict[str, str] = {
                "date": "2025-01-15",
                "date-time": "2025-01-15T00:00:00Z",
                "uuid": "00000000-0000-0000-0000-000000000000",
                "email": "mock@example.com",
                "uri": "https://example.com",
            }
            return format_map.get(prop["format"], "[MOCK]")
        # String with minLength=1 — use a meaningful mock value.
        if prop.get("minLength", 0) >= 1:
            return "[MOCK TEXT]"
        return "[MOCK]"

    if type_name == "number":
        return 0.0

    if type_name == "integer":
        return 0

    if type_name == "boolean":
        return False

    if type_name == "array":
        return []

    if type_name == "object":
        if "properties" in prop:
            return _minimal_valid_instance(prop)
        return {}

    return "[MOCK]"
