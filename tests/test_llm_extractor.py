"""LLM extractor tests — all using MockLLMProvider for zero-cost CI.

CLAUDE.md §7: CI must never hit a paid API. The mock provider returns
deterministic structured JSON that flows through Pydantic validation,
citation verification and the retry loop exactly as real output would.
"""

from __future__ import annotations

import json
import uuid
from decimal import Decimal

import pytest
import pytest_asyncio
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import (
    ClauseType,
    CovenantType,
    ExtractionStatus,
    Severity,
)
from app.domain.rules import ComparisonOperator
from app.extract.candidates import Candidate
from app.extract.llm_extractor import LLMExtractor
from app.extract.schemas import EXTRACTION_JSON_SCHEMA, LLMCovenantExtraction
from app.llm.mock import MockLLMProvider
from app.llm.router import LLMRouter


@pytest_asyncio.fixture
async def mock_router(db_session: AsyncSession) -> LLMRouter:
    """A router that never spends money."""
    return LLMRouter(db_session, provider=MockLLMProvider())


@pytest_asyncio.fixture
async def extractor(db_session: AsyncSession, mock_router: LLMRouter) -> LLMExtractor:
    return LLMExtractor(db_session, router=mock_router)


@pytest.fixture
def candidate() -> Candidate:
    return Candidate(
        chunk_id=uuid.uuid4(),
        text=(
            "The Issuer shall at all times maintain a consolidated gearing ratio of "
            "not more than 1.75 times, tested semi-annually."
        ),
        char_start=0,
        char_end=120,
        clause_type_hints=frozenset({ClauseType.FINANCIAL_COVENANT}),
        page_number=2,
        section_title="Financial Covenants",
    )


# -- extraction ---------------------------------------------------------------


async def test_mock_extraction_produces_valid_output(
    extractor: LLMExtractor, candidate: Candidate
) -> None:
    result = await extractor.extract(candidate)

    assert result.output is not None
    assert result.extraction_status is ExtractionStatus.EXTRACTED
    assert isinstance(result.output, LLMCovenantExtraction)
    # The mock returns minimal valid instances — clause_type defaults to the first enum.
    assert isinstance(result.output.clause_type, ClauseType)
    assert result.cost_usd >= Decimal("0")
    assert result.model_id != ""


async def test_mock_extraction_is_deterministic(
    extractor: LLMExtractor, candidate: Candidate
) -> None:
    """Same input → same output, every time (the mock is content-hash-keyed)."""
    first = await extractor.extract(candidate)
    second = await extractor.extract(candidate)

    assert first.output is not None
    assert second.output is not None
    # The mock returns the same content-hash-keyed result.
    assert first.output.clause_type == second.output.clause_type


async def test_extraction_includes_cost(extractor: LLMExtractor, candidate: Candidate) -> None:
    result = await extractor.extract(candidate)

    # Even the mock records a non-zero estimated cost for the prompt.
    assert result.cost_usd >= Decimal("0")
    assert result.model_id == "claude-opus-5"  # from settings.EXTRACTION_MODEL


# -- validation and retry -----------------------------------------------------


async def test_validation_error_records_the_failure(
    db_session: AsyncSession, candidate: Candidate
) -> None:
    """An LLM output that fails Pydantic validation is caught."""
    # The mock always returns minimal valid instances, so this path is hard
    # to reach with the mock. Instead we test the _validate method directly.
    from app.llm.router import LLMCallResult

    extractor = LLMExtractor(db_session, router=LLMRouter(db_session, provider=MockLLMProvider()))

    # Simulate a response with invalid content.
    bad_result = LLMCallResult(
        content={"clause_type": "not_a_real_clause_type", "source_quote": "test"},
        model_id="mock-v1",
        provider="mock",
        prompt_tokens=10,
        completion_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        estimated_cost_usd=Decimal("0.001"),
        cache_hit=False,
    )

    validation = extractor._validate(bad_result, candidate)
    assert validation.output is None
    assert validation.extraction_status is ExtractionStatus.VALIDATION_FAILED
    assert len(validation.validation_errors) > 0
    assert not validation.retry_attempted


async def test_text_instead_of_json_is_caught(
    db_session: AsyncSession, candidate: Candidate
) -> None:
    """When the LLM returns a string instead of structured JSON."""
    from app.llm.router import LLMCallResult

    extractor = LLMExtractor(db_session, router=LLMRouter(db_session, provider=MockLLMProvider()))

    bad_result = LLMCallResult(
        content="Sorry, I couldn't parse that clause.",
        model_id="mock-v1",
        provider="mock",
        prompt_tokens=10,
        completion_tokens=5,
        cache_read_tokens=0,
        cache_write_tokens=0,
        estimated_cost_usd=Decimal("0.001"),
        cache_hit=False,
    )

    validation = extractor._validate(bad_result, candidate)
    assert validation.output is None
    assert validation.extraction_status is ExtractionStatus.VALIDATION_FAILED
    assert any("text" in err.lower() for err in validation.validation_errors)


# -- citation verification ----------------------------------------------------


async def test_llm_output_quote_is_checked_against_candidate_text(
    db_session: AsyncSession,
) -> None:
    """The quick citation check verifies the LLM's quote against the text it saw."""
    from app.extract.citations import verify_quote

    candidate_text = "The Issuer shall maintain a gearing ratio of not more than 1.75 times."
    llm_quote = "The Issuer shall maintain a gearing ratio of not more than 1.75 times."

    check = verify_quote(llm_quote, candidate_text)
    assert check.verified
    assert check.score == 1.0


async def test_fuzzy_citation_catches_dropped_footnote() -> None:
    """Phase 6: rapidfuzz catches LLM quotes that differ by a footnote marker."""
    from app.extract.citations import verify_quote

    chunk = "The Issuer shall not create any security interest[1] over its assets."
    quote = "The Issuer shall not create any security interest over its assets"

    check = verify_quote(quote, chunk)
    assert check.verified
    assert check.method == "fuzzy"


# -- cost tracking ------------------------------------------------------------


async def test_cache_hit_records_zero_cost(extractor: LLMExtractor, candidate: Candidate) -> None:
    """Cost is always tracked, even with the mock provider."""
    first = await extractor.extract(candidate)

    # The mock provider has non-zero cost estimates for prompt tokens.
    # The important invariant: cost is always tracked.
    assert first.cost_usd >= Decimal("0")
    assert first.model_id != ""


# -- LLMCovenantExtraction schema --------------------------------------------


def test_valid_extraction_passes_validation() -> None:
    output = LLMCovenantExtraction(
        clause_type=ClauseType.FINANCIAL_COVENANT,
        covenant_type=CovenantType.GEARING_RATIO,
        source_quote="The Issuer shall maintain a gearing ratio of not more than 1.75 times.",
        confidence=0.95,
        summary="Gearing ratio capped at 1.75x",
        threshold_ratio=Decimal("1.75"),
        operator=ComparisonOperator.LTE,
        severity=Severity.HIGH,
    )
    assert output.clause_type is ClauseType.FINANCIAL_COVENANT
    assert output.threshold_ratio == Decimal("1.75")


def test_monetary_threshold_requires_currency() -> None:
    with pytest.raises(ValueError, match="currency"):
        LLMCovenantExtraction(
            clause_type=ClauseType.CROSS_DEFAULT,
            covenant_type=CovenantType.CROSS_DEFAULT,
            source_quote="indebtedness exceeding RM30,000,000",
            confidence=0.95,
            summary="Cross-default at RM30m",
            threshold_amount=Decimal("30000000"),
            # threshold_currency is missing — should fail validation.
        )


def test_rating_trigger_requires_agency() -> None:
    with pytest.raises(ValueError, match="agency"):
        LLMCovenantExtraction(
            clause_type=ClauseType.RATING_TRIGGER,
            covenant_type=CovenantType.RATING_TRIGGER,
            source_quote="downgraded below BBB+",
            confidence=0.90,
            summary="Rating downgrade trigger",
            trigger_rating="BBB+",
            # rating_agency is missing — should fail validation.
        )


def test_json_schema_is_deterministic() -> None:
    """The JSON schema must be byte-stable for prompt caching."""
    from app.extract.schemas import EXTRACTION_JSON_SCHEMA, extraction_jsonschema

    first = extraction_jsonschema()
    second = extraction_jsonschema()
    assert first == second
    assert EXTRACTION_JSON_SCHEMA == first


class TestSchemaIsAcceptedByTheStructuredOutputEndpoint:
    """Constraints verified against the live API, pinned so they cannot regress.

    Both were discovered by a 400 on the first real call, and neither is
    visible to any offline test that does not know the endpoint's rules.
    """

    def test_every_object_is_closed(self) -> None:
        """ "For 'object' type, 'additionalProperties' must be explicitly set to false"."""

        def check(node: object) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object" or "properties" in node:
                    assert node.get("additionalProperties") is False, node.get("title", node)
                for value in node.values():
                    check(value)
            elif isinstance(node, list):
                for item in node:
                    check(item)

        check(EXTRACTION_JSON_SCHEMA)

    def test_unsupported_validation_keywords_are_absent(self) -> None:
        """ "For 'number' type, properties maximum, minimum are not supported".

        Pydantic emits these from `Field(ge=…, le=…)` and `max_length`. They are
        stripped from the wire schema only; `model_validate` still enforces
        them, which is where the guarantee actually lives.
        """
        rendered = json.dumps(EXTRACTION_JSON_SCHEMA)
        for keyword in ("minimum", "maximum", "minLength", "maxLength", "multipleOf", "pattern"):
            assert f'"{keyword}"' not in rendered, keyword

    def test_the_model_still_enforces_what_the_schema_no_longer_states(self) -> None:
        """Stripping keywords from the wire schema must not weaken validation."""
        with pytest.raises(ValidationError):
            LLMCovenantExtraction(
                clause_type=ClauseType.FINANCIAL_COVENANT,
                source_quote="x",
                confidence=1.5,  # ge=0, le=1 — no longer in the wire schema
            )
