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
from app.extract.schemas import LLMCovenantExtraction
from app.llm.router import LLMCallResult, LLMRouter

logger = get_logger(__name__)

# Minimum max_tokens for extraction: thinking + structured response together.
_EXTRACTION_MAX_TOKENS: int = 8000


@dataclass
class LLMExtraction:
    """Everything the LLM extractor produced from one candidate span.

    `output` is None when extraction failed (validation, budget, or error).
    `extraction_status` says why, and `validation_errors` carries the details.
    """

    candidate: Candidate
    output: LLMCovenantExtraction | None = None
    extraction_status: ExtractionStatus = ExtractionStatus.PENDING
    validation_errors: list[str] = field(default_factory=list)
    retry_attempted: bool = False
    citation_check: CitationCheck | None = None
    cost_usd: Decimal = Decimal("0")
    model_id: str = ""
    cache_hit: bool = False

    @property
    def succeeded(self) -> bool:
        return self.output is not None and self.extraction_status is ExtractionStatus.EXTRACTED


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
        if extraction.output is not None:
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

        if retry_extraction.output is None:
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
                max_tokens=_EXTRACTION_MAX_TOKENS,
                response_schema=EXTRACTION_JSON_SCHEMA,
                prompt_version=PROMPT_VERSION,
                document_id=document_id,
                enable_prompt_caching=True,
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
        """Validate the LLM response and perform a quick citation check.

        Always returns an LLMExtraction. The caller checks `output`:
        - `output is not None` → success, persist it.
        - `output is None and not retry_attempted` → retry needed.
        - `output is None and retry_attempted` → final failure, route to review.
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
            output = LLMCovenantExtraction.model_validate(content)
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

        # Quick citation check against the candidate text.
        citation = verify_quote(output.source_quote, candidate.text)
        if not citation.verified:
            logger.info(
                "quick citation check failed against candidate text",
                extra={
                    "candidate_chunk_id": str(candidate.chunk_id),
                    "source_quote": output.source_quote[:200],
                    "method": citation.method,
                },
            )

        return LLMExtraction(
            candidate=candidate,
            output=output,
            extraction_status=ExtractionStatus.EXTRACTED,
            retry_attempted=retry_attempted,
            citation_check=citation,
            cost_usd=result.estimated_cost_usd,
            model_id=result.model_id,
            cache_hit=result.cache_hit,
        )
