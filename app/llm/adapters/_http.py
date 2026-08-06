"""Shared HTTP behaviour for provider adapters.

Retry policy and error reporting belong in one place. They started in the
Anthropic adapter alone, and the OpenAI adapter -- which is the one that
actually meets a rate limit, because it is billed per page image -- had
neither: a single transient 429 killed the page, and the exception said only
"429 Too Many Requests" while the body explained exactly what was wrong.

**A 429 is two different failures.** Rate limiting is transient and worth
retrying; an exhausted quota is a billing state that no amount of backoff will
change. Retrying the second wastes the operator's time and buries the one
sentence that would have told them to add credits, so the two are separated
here rather than lumped under the status code they share.
"""

from __future__ import annotations

import asyncio
from typing import Any, Final

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

# Transient. 529 is Anthropic's "overloaded"; the rest are the usual rate-limit
# and gateway failures.
RETRYABLE_STATUS: Final[frozenset[int]] = frozenset({429, 500, 502, 503, 529})

# Provider error codes that arrive as a retryable status but are not retryable:
# the account is out of money. Backing off changes nothing.
QUOTA_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "insufficient_quota",
        "credit_balance_exhausted",
        "billing_not_active",
    }
)

MAX_RETRIES: Final[int] = 3
BASE_RETRY_DELAY_SECONDS: Final[float] = 1.0
MAX_RETRY_DELAY_SECONDS: Final[float] = 30.0


class ProviderQuotaExhaustedError(RuntimeError):
    """The provider account is out of credit.

    Distinct from a rate limit so callers can say "add credits" instead of
    "temporarily unavailable", and so the retry loop stops immediately.
    """

    def __init__(self, provider: str, detail: str) -> None:
        self.provider = provider
        self.detail = detail
        super().__init__(f"{provider} quota exhausted: {detail}")


async def post_with_retry(
    client: httpx.AsyncClient,
    path: str,
    body: dict[str, Any],
    *,
    provider: str,
) -> httpx.Response:
    """POST with bounded backoff on the retryable statuses.

    Everything else raises immediately: a 400 from a rejected parameter must
    surface loudly rather than be retried three times.
    """
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES):
        response = await client.post(path, json=body)

        if response.status_code not in RETRYABLE_STATUS:
            raise_with_body(response, provider=provider)
            return response

        # A retryable status that is really a billing state: stop now.
        detail, code = _error_detail(response)
        if code in QUOTA_ERROR_CODES:
            logger.error(
                "llm.quota_exhausted",
                extra={
                    "provider": provider,
                    "status": response.status_code,
                    "detail": detail[:300],
                },
            )
            raise ProviderQuotaExhaustedError(provider, detail)

        last_error = httpx.HTTPStatusError(
            f"{provider} returned {response.status_code}: {detail}"
            if detail
            else f"{provider} returned {response.status_code}",
            request=response.request,
            response=response,
        )
        if attempt == MAX_RETRIES - 1:
            break

        delay = retry_delay(response, attempt)
        logger.warning(
            "llm.retrying",
            extra={
                "provider": provider,
                "status": response.status_code,
                "attempt": attempt + 1,
                "delay_seconds": delay,
            },
        )
        await asyncio.sleep(delay)

    assert last_error is not None
    raise last_error


def raise_with_body(response: httpx.Response, *, provider: str) -> None:
    """`raise_for_status()`, but carrying the provider's explanation.

    Providers put the actual fault in the response body -- "for 'object' type,
    'additionalProperties' must be explicitly set to false", "You have no
    credits remaining" -- while the status line says only "400 Bad Request" or
    "429 Too Many Requests". Losing that turns a one-line fix into a debugging
    session, which is what it cost the first time each of these adapters met
    its real API.
    """
    if response.is_success:
        return

    detail, code = _error_detail(response)
    if code in QUOTA_ERROR_CODES:
        raise ProviderQuotaExhaustedError(provider, detail)

    request_id = response.headers.get("request-id") or response.headers.get("x-request-id", "")
    logger.error(
        "llm.error_response",
        extra={
            "provider": provider,
            "status": response.status_code,
            "detail": detail[:500],
            "request_id": request_id,
        },
    )

    message = f"{provider} returned {response.status_code}"
    if detail:
        message += f": {detail}"
    if request_id:
        message += f" (request-id {request_id})"
    raise httpx.HTTPStatusError(message, request=response.request, response=response)


def retry_delay(response: httpx.Response, attempt: int) -> float:
    """Honour `retry-after` when the server sends one; otherwise exponential."""
    header = response.headers.get("retry-after")
    if header:
        try:
            return min(float(header), MAX_RETRY_DELAY_SECONDS)
        except ValueError:
            pass
    backoff: float = BASE_RETRY_DELAY_SECONDS * (2**attempt)
    return min(backoff, MAX_RETRY_DELAY_SECONDS)


def _error_detail(response: httpx.Response) -> tuple[str, str]:
    """`(message, code)` from a provider error body. Empty strings when absent."""
    try:
        payload = response.json()
    except ValueError:
        return response.text[:500], ""

    if not isinstance(payload, dict):
        return "", ""

    error = payload.get("error")
    if isinstance(error, dict):
        return str(error.get("message", "") or error), str(error.get("code", "") or "")
    if error:
        return str(error), ""
    return "", ""
