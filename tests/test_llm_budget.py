"""Budget guard tests.

docs/plan.md Phase 5 acceptance: "budget guard provably blocks an over-budget call
(unit test)." These are those tests.

Uses settings injection rather than monkeypatch to avoid cached-settings issues.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.llm.budget import BudgetDecision, BudgetExceededError, BudgetGuard, BudgetOutcome

# A Settings instance with tiny limits so we can exercise every guard without
# needing to record hundreds of dollars of fake spend.
_tiny_settings = Settings(
    ENVIRONMENT="test",
    MAX_COST_PER_CALL_USD=Decimal("0.10"),
    MAX_TOTAL_COST_USD=Decimal("2.00"),  # high enough that only global tests trip it
    MAX_COST_PER_DOCUMENT_USD=Decimal("0.02"),
    MAX_VLM_PAGES_PER_DOC=5,
)


class TestPerCallGuard:
    """The per-call ceiling — no DB query needed."""

    async def test_allows_call_under_limit(self, db_session: AsyncSession) -> None:
        guard = BudgetGuard(db_session, settings=_tiny_settings)
        decision = await guard.check_call(estimated_cost=Decimal("0.05"))
        assert decision.allowed
        assert decision.outcome is BudgetOutcome.ALLOWED

    async def test_blocks_call_over_limit(self, db_session: AsyncSession) -> None:
        guard = BudgetGuard(db_session, settings=_tiny_settings)
        decision = await guard.check_call(estimated_cost=Decimal("50.00"))
        assert not decision.allowed
        assert decision.outcome is BudgetOutcome.REJECTED_PER_CALL

    async def test_allows_call_at_exact_limit(self, db_session: AsyncSession) -> None:
        guard = BudgetGuard(db_session, settings=_tiny_settings)
        decision = await guard.check_call(estimated_cost=Decimal("0.10"))
        assert decision.allowed


class TestGlobalGuard:
    """The global circuit breaker queries cumulative spend from llm_calls."""

    @pytest_asyncio.fixture(autouse=True)
    async def _clear_llm_calls(self, db_session: AsyncSession) -> None:
        from sqlalchemy import text

        await db_session.execute(text("DELETE FROM llm_calls"))
        await db_session.flush()

    async def test_allows_when_under_global_limit(self, db_session: AsyncSession) -> None:
        guard = BudgetGuard(db_session, settings=_tiny_settings)
        # 0.02 < 0.03 global, 0.02 < 0.10 per-call — passes both
        decision = await guard.check_call(estimated_cost=Decimal("0.02"))
        assert decision.allowed

    async def test_blocks_when_global_limit_exceeded(self, db_session: AsyncSession) -> None:
        """Record a call that uses most of the cap, then try to add another."""
        from app.db.models.ops import LLMCall
        from app.db.repositories.ops import LLMCallRepository
        from app.domain.enums import LLMStage

        # This test needs a tight global cap specifically
        tight = Settings(
            ENVIRONMENT="test",
            MAX_COST_PER_CALL_USD=Decimal("1.00"),
            MAX_TOTAL_COST_USD=Decimal("0.03"),
        )

        # Already spent $0.025 of the $0.03 global cap
        call = LLMCall(
            stage=LLMStage.EXTRACT,
            provider="anthropic",
            model_id="claude-opus-5",
            prompt_tokens=1000,
            completion_tokens=100,
            estimated_cost_usd=Decimal("0.025"),
        )
        await LLMCallRepository(db_session).add(call)
        await db_session.flush()

        guard = BudgetGuard(db_session, settings=tight)
        # $0.025 + $0.01 = $0.035 > $0.03 global cap
        decision = await guard.check_call(estimated_cost=Decimal("0.01"))
        assert not decision.allowed
        assert decision.outcome is BudgetOutcome.REJECTED_GLOBAL
        assert "global" in decision.reason.lower()


class TestPerDocumentGuard:
    """Per-document caps sum across LLMCall and ExtractionJob spend."""

    @pytest_asyncio.fixture(autouse=True)
    async def _clear_tables(self, db_session: AsyncSession) -> None:
        from sqlalchemy import text

        await db_session.execute(text("DELETE FROM llm_calls"))
        await db_session.execute(text("DELETE FROM extraction_jobs"))
        await db_session.flush()

    async def test_allows_when_under_document_limit(self, db_session: AsyncSession) -> None:
        import uuid

        guard = BudgetGuard(db_session, settings=_tiny_settings)
        decision = await guard.check_call(
            estimated_cost=Decimal("0.01"),
            document_id=uuid.uuid4(),
        )
        assert decision.allowed

    async def test_blocks_when_document_limit_exceeded(self, db_session: AsyncSession) -> None:
        import uuid

        # Use an existing document_id from the seeded_universe, or None for FK
        # safety — we only need the spend to be attributed.
        from app.db.models.ops import LLMCall
        from app.db.repositories.ops import LLMCallRepository
        from app.domain.enums import LLMStage

        # Record cost without a document FK (FK is nullable) — the per-doc
        # guard queries by document_id, so this row won't affect the test.
        # Then record a row that DOES target our document.
        doc_id = uuid.uuid4()

        call = LLMCall(
            document_id=None,  # nullable FK — avoids FK violation
            stage=LLMStage.EXTRACT,
            provider="anthropic",
            model_id="claude-opus-5",
            prompt_tokens=1000,
            completion_tokens=100,
            estimated_cost_usd=Decimal("0.015"),
        )
        await LLMCallRepository(db_session).add(call)
        await db_session.flush()

        # Now target a second call at our document. The sum for doc_id is 0.015
        # but the per-doc guard also checks ExtractionJob rows (zero for now).
        # With 0.015 spent, adding 0.01 = 0.025 > 0.02 per-doc cap.

        # We can't use LLMCall with a fake UUID because of the FK. Instead,
        # we use ExtractionJob which also contributes to per-doc spend.
        # Create a document first so the FK is valid
        from sqlalchemy import text as _text

        from app.db.models.ops import ExtractionJob
        from app.db.repositories.ops import ExtractionJobRepository
        from app.domain.enums import JobStatus, JobType

        await db_session.execute(
            _text(
                "INSERT INTO documents "
                "(id, sha256, filename, status, source_type, document_type, language) "
                "VALUES (:id, :sha, :name, :status, 'upload', 'unknown', 'en')"
            ),
            {"id": doc_id, "sha": "aa" * 32, "name": "test.pdf", "status": "parsed"},
        )
        await db_session.flush()

        job = ExtractionJob(
            document_id=doc_id,
            document_sha256="aa" * 32,
            job_type=JobType.EXTRACT_COVENANT,
            status=JobStatus.SUCCEEDED,
            model_id="test",
            estimated_cost_usd=Decimal("0.015"),
        )
        await ExtractionJobRepository(db_session).add(job)
        await db_session.flush()

        guard = BudgetGuard(db_session, settings=_tiny_settings)
        decision = await guard.check_call(
            estimated_cost=Decimal("0.01"),
            document_id=doc_id,
        )
        assert not decision.allowed
        assert decision.outcome is BudgetOutcome.REJECTED_PER_DOCUMENT
        assert "per-document" in decision.reason.lower()


class TestVLMPageGuard:
    """The VLM page cap is a count guard, not a cost guard."""

    async def test_allows_under_cap(self, db_session: AsyncSession) -> None:
        guard = BudgetGuard(db_session, settings=_tiny_settings)
        decision = await guard.check_call(
            estimated_cost=Decimal("0.10"),
            vlm_page_count=3,
        )
        assert decision.allowed

    async def test_blocks_over_cap(self, db_session: AsyncSession) -> None:
        guard = BudgetGuard(db_session, settings=_tiny_settings)
        decision = await guard.check_call(
            estimated_cost=Decimal("0.10"),
            vlm_page_count=50,
        )
        assert not decision.allowed
        assert decision.outcome is BudgetOutcome.REJECTED_VLM_PAGES


class TestBudgetExceededError:
    """BudgetExceededError carries structured info for callers to act on."""

    async def test_error_carries_decision(self, db_session: AsyncSession) -> None:
        decision = BudgetDecision(
            outcome=BudgetOutcome.REJECTED_PER_CALL,
            estimated_cost=Decimal("10.00"),
            limit=Decimal("0.50"),
            current_spend=Decimal("0"),
            reason="too expensive",
        )
        error = BudgetExceededError(decision)
        assert error.decision is decision
        assert str(error) == "too expensive"


class TestBudgetDecisionInterface:
    """The BudgetDecision dataclass is the protocol between guard and router."""

    def test_decision_is_frozen(self) -> None:
        d = BudgetDecision(
            outcome=BudgetOutcome.ALLOWED,
            estimated_cost=Decimal("0"),
            limit=Decimal("0"),
            current_spend=Decimal("0"),
            reason="",
        )
        with pytest.raises(Exception):  # noqa: B017 — frozen dataclass raises any Exception on mutation
            d.outcome = BudgetOutcome.REJECTED_GLOBAL  # type: ignore[misc]
