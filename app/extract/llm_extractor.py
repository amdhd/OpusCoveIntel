"""LLM-based covenant extractor — calls the model, validates, retries once.

CLAUDE.md 1.4: every model call goes through `app/llm/router.py`. This extractor
calls `router.chat()` with:
- The extraction system prompt (prompt-cached prefix)
- Structured output (JSON Schema matching LLMCovenantExtraction)
- `max_tokens ≥ 8000` (thinking + response together, per CLAUDE.md §2)

Pydantic validation + one feedback retry, per PLAN.md Phase 6. After the retry,
a still-invalid extraction goes to the human review queue — never silently dropped.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from decimal import Decimal

from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.enums import ExtractionStatus, LLMStage
from app.extract.candidates import Candidate
from app.extract.citations import CitationCheck, verify_quote
from app.extract.prompts import PROMPT_VERSION, build_system_prompt, build_user_message
from app.extract.schemas import LLMCovenantExtraction, LLMExtractionResponse
from app.llm.router import LLMCallResult, LLMRouter

logger = get_logger(__name__)

# Minimum max_tokens for extraction: thinking + structured response together.
EXTRACTION_MAX_TOKENS: int = 8000

# CLAUDE.md 2: covenant extraction is the highest-stakes call in the system and
# runs at `effort: high`. Depth is steered here, not with `thinking.budget_tokens`
# (a 400) and not with `temperature` (also a 400).
EXTRACTION_EFFORT: str = "high"


@dataclass
class LLMExtraction:
    """Everything the LLM extractor produced from one candidate span.

    `outputs` holds every covenant the model found in the span — a list,
    because a span routinely states more than one and the previous
    single-covenant shape silently dropped all but the first.

    An empty `outputs` with `extraction_status == EXTRACTED` is a real answer:
    the span held no covenant. That is different from a failure, where the
    status says why and `validation_errors` carries the detail.
    """

    candidate: Candidate
    outputs: list[LLMCovenantExtraction] = field(default_factory=list)
    extraction_status: ExtractionStatus = ExtractionStatus.PENDING
    validation_errors: list[str] = field(default_factory=list)
    retry_attempted: bool = False
    # One citation check per covenant, in the same order as `outputs`.
    citation_checks: list[CitationCheck] = field(default_factory=list)
    cost_usd: Decimal = Decimal("0")
    model_id: str = ""
    cache_hit: bool = False

    @property
    def succeeded(self) -> bool:
        """The call produced a valid answer — including a valid empty one."""
        return self.extraction_status is ExtractionStatus.EXTRACTED


class LLMExtractor:
    """Extract covenants from candidate spans using the configured LLM.

    Constructed with the async session and an optional pre-built router.
    When `router` is None, one is built from the session (real adapters).
    In tests, inject a router backed by MockLLMProvider for zero-cost CI.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        router: LLMRouter | None = None,
    ) -> None:
        self._router = router or LLMRouter(session)
        self._settings = get_settings()

    async def extract(
        self,
        candidate: Candidate,
        *,
        document_id: uuid.UUID | None = None,
    ) -> LLMExtraction:
        """Run extraction on one candidate span.

        The flow:
        1. Call the LLM with structured output.
        2. Validate the response against LLMCovenantExtraction.
        3. On validation failure: ONE retry with the error appended.
        4. Quick citation check against the candidate text.
        5. Return the result — the caller handles persistence and the full
           citation verification against the chunk.
        """
        system_prompt = build_system_prompt()
        user_message = build_user_message(candidate.text)

        # -- First attempt ---------------------------------------------------
        result = await self._call_llm(
            document_id=document_id,
            system_prompt=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )

        extraction = self._validate(result, candidate)
        if extraction.succeeded:
            return extraction
        if extraction.retry_attempted:
            return extraction  # Already a retry (text-instead-of-JSON edge case handled)

        # -- Retry with validation error ------------------------------------
        error_detail = "; ".join(extraction.validation_errors)
        retry_result = await self._call_llm(
            document_id=document_id,
            system_prompt=system_prompt,
            messages=[
                {"role": "user", "content": user_message},
                {
                    "role": "assistant",
                    "content": (
                        result.content if isinstance(result.content, str) else str(result.content)
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Your previous response failed validation with these errors:\n"
                        f"{error_detail}\n\n"
                        f"Please fix the errors and return a valid JSON object matching the schema."
                    ),
                },
            ],
        )

        retry_extraction = self._validate(retry_result, candidate, retry_attempted=True)

        # Combine costs from both attempts.
        retry_extraction.cost_usd += extraction.cost_usd

        if not retry_extraction.succeeded:
            logger.warning(
                "llm extraction failed after retry; routing to review",
                extra={
                    "candidate_chunk_id": str(candidate.chunk_id),
                    "candidate_page": candidate.page_number,
                    "errors": retry_extraction.validation_errors,
                },
            )

        return retry_extraction

    # -- internals -----------------------------------------------------------

    async def _call_llm(
        self,
        *,
        document_id: uuid.UUID | None,
        system_prompt: str,
        messages: list[dict[str, str]],
    ) -> LLMCallResult:
        """Dispatch one chat call through the router."""
        from app.extract.schemas import EXTRACTION_JSON_SCHEMA

        try:
            return await self._router.chat(
                stage=LLMStage.EXTRACT,
                provider_name="anthropic",
                model_id=self._settings.EXTRACTION_MODEL,
                system_prompt=system_prompt,
                messages=messages,
                max_tokens=EXTRACTION_MAX_TOKENS,
                response_schema=EXTRACTION_JSON_SCHEMA,
                prompt_version=PROMPT_VERSION,
                document_id=document_id,
                enable_prompt_caching=True,
                effort=EXTRACTION_EFFORT,
            )
        except Exception as exc:
            logger.error(
                "llm extraction call failed",
                extra={
                    "document_id": str(document_id),
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise

    def _validate(
        self,
        result: LLMCallResult,
        candidate: Candidate,
        *,
        retry_attempted: bool = False,
    ) -> LLMExtraction:
        """Validate the LLM response and quick-check each covenant's citation.

        Always returns an LLMExtraction. The caller checks `succeeded`:
        - succeeded → persist `outputs`, which may legitimately be empty.
        - not succeeded and not `retry_attempted` → retry.
        - not succeeded and `retry_attempted` → final failure, route to review.
        """
        content = result.content
        if isinstance(content, str):
            # The model returned free text instead of structured JSON.
            return LLMExtraction(
                candidate=candidate,
                extraction_status=ExtractionStatus.VALIDATION_FAILED,
                validation_errors=["LLM returned text instead of structured JSON"],
                retry_attempted=retry_attempted,
                cost_usd=result.estimated_cost_usd,
                model_id=result.model_id,
                cache_hit=result.cache_hit,
            )

        # content is a dict from structured output — validate via Pydantic.
        try:
            response = LLMExtractionResponse.model_validate(content)
        except ValidationError as exc:
            errors = [str(e) for e in exc.errors()]
            logger.info(
                "llm extraction validation failed",
                extra={
                    "candidate_chunk_id": str(candidate.chunk_id),
                    "errors": errors,
                    "retry_attempted": retry_attempted,
                },
            )
            return LLMExtraction(
                candidate=candidate,
                extraction_status=ExtractionStatus.VALIDATION_FAILED,
                validation_errors=errors,
                retry_attempted=retry_attempted,
                cost_usd=result.estimated_cost_usd,
                model_id=result.model_id,
                cache_hit=result.cache_hit,
            )

        # One quick citation check per covenant, against the candidate text.
        # Kept positional with `outputs` so the caller can pair them; the
        # authoritative check is still the caller's, against the chunk.
        checks = [verify_quote(item.source_quote, candidate.text) for item in response.covenants]
        for item, check in zip(response.covenants, checks, strict=True):
            if not check.verified:
                logger.info(
                    "quick citation check failed against candidate text",
                    extra={
                        "candidate_chunk_id": str(candidate.chunk_id),
                        "covenant_type": item.covenant_type.value if item.covenant_type else None,
                        "source_quote": item.source_quote[:200],
                        "method": check.method,
                    },
                )

        if len(response.covenants) > 1:
            logger.info(
                "span held several covenants",
                extra={
                    "candidate_chunk_id": str(candidate.chunk_id),
                    "covenants": [
                        item.covenant_type.value if item.covenant_type else "none"
                        for item in response.covenants
                    ],
                },
            )

        return LLMExtraction(
            candidate=candidate,
            outputs=list(response.covenants),
            extraction_status=ExtractionStatus.EXTRACTED,
            retry_attempted=retry_attempted,
            citation_checks=checks,
            cost_usd=result.estimated_cost_usd,
            model_id=result.model_id,
            cache_hit=result.cache_hit,
        )
