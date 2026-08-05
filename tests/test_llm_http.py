"""Shared adapter HTTP behaviour: retry, and the two meanings of a 429.

Found live: the VLM's first real call came back 429, and the adapter reported
only "429 Too Many Requests". The body said "You have no credits remaining" —
a billing state, not a rate limit. Retrying it would have burned three attempts
and two seconds of backoff to reach the same failure, with the one useful
sentence still hidden.
"""

from __future__ import annotations

import httpx
import pytest

from app.llm.adapters._http import (
    ProviderQuotaExhaustedError,
    post_with_retry,
    retry_delay,
)


def _client(handler: httpx.MockTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=handler, base_url="https://example.test")


class TestQuotaIsNotARateLimit:
    async def test_an_exhausted_quota_raises_immediately(self) -> None:
        attempts = 0

        def handle(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "You have no credits remaining. Add credits to continue.",
                        "type": "insufficient_quota",
                        "code": "credit_balance_exhausted",
                    }
                },
            )

        async with _client(httpx.MockTransport(handle)) as client:
            with pytest.raises(ProviderQuotaExhaustedError) as excinfo:
                await post_with_retry(client, "/v1/x", {}, provider="openai")

        assert attempts == 1, "an out-of-credit account must not be retried"
        assert "no credits remaining" in excinfo.value.detail
        assert excinfo.value.provider == "openai"

    async def test_a_genuine_rate_limit_is_retried(self) -> None:
        attempts = 0

        def handle(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                return httpx.Response(
                    429,
                    json={
                        "error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}
                    },
                    headers={"retry-after": "0"},
                )
            return httpx.Response(200, json={"ok": True})

        async with _client(httpx.MockTransport(handle)) as client:
            response = await post_with_retry(client, "/v1/x", {}, provider="openai")

        assert attempts == 3
        assert response.json() == {"ok": True}


class TestErrorsCarryTheirExplanation:
    async def test_a_400_reports_the_provider_message(self) -> None:
        """The failure that cost a debugging session the first time."""

        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={
                    "error": {
                        "type": "invalid_request_error",
                        "message": (
                            "output_config.format.schema: For 'object' type, "
                            "'additionalProperties' must be explicitly set to false"
                        ),
                    }
                },
            )

        async with _client(httpx.MockTransport(handle)) as client:
            with pytest.raises(httpx.HTTPStatusError) as excinfo:
                await post_with_retry(client, "/v1/messages", {}, provider="anthropic")

        assert "additionalProperties" in str(excinfo.value)

    async def test_a_400_is_never_retried(self) -> None:
        attempts = 0

        def handle(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(400, json={"error": {"message": "bad schema"}})

        async with _client(httpx.MockTransport(handle)) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await post_with_retry(client, "/v1/x", {}, provider="anthropic")

        assert attempts == 1, "a rejected parameter does not become valid on retry"

    async def test_a_non_json_body_still_surfaces(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, text="<html>bad gateway</html>")

        async with _client(httpx.MockTransport(handle)) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await post_with_retry(client, "/v1/x", {}, provider="openai")


class TestRetryDelay:
    def test_retry_after_is_honoured(self) -> None:
        response = httpx.Response(429, headers={"retry-after": "7"})
        assert retry_delay(response, attempt=0) == 7.0

    def test_a_nonsense_retry_after_falls_back_to_backoff(self) -> None:
        response = httpx.Response(429, headers={"retry-after": "soon"})
        assert retry_delay(response, attempt=0) == 1.0

    def test_backoff_is_capped(self) -> None:
        response = httpx.Response(429)
        assert retry_delay(response, attempt=20) == 30.0
