"""Budget guard — the gate every LLM call passes through before dispatch.

PLAN.md 2 specifies four guards, checked in this order:
1. Per-call ceiling (MAX_COST_PER_CALL_USD) — reject before dispatch
2. Per-document ceiling (MAX_COST_PER_DOCUMENT_USD) — abort doc, mark budget_exceeded
3. Global ceiling (MAX_TOTAL_COST_USD) — circuit-breaker opens, all calls refused
4. VLM pages per doc (MAX_VLM_PAGES_PER_DOC) — fail loudly

This is the single place where spend is rejected. Every enforcement is
accompanied by a structured log at WARNING level so the event is queryable.

Money is Decimal, never float (CLAUDE.md 6).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from app.core.config import get_settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = get_logger(__name__)


class BudgetOutcome(StrEnum):
    ALLOWED = "allowed"
    REJECTED_PER_CALL = "rejected_per_call"
    REJECTED_PER_DOCUMENT = "rejected_per_document"
    REJECTED_GLOBAL = "rejected_global"
    REJECTED_VLM_PAGES = "rejected_vlm_pages"


@dataclass(frozen=True)
class BudgetDecision:
    outcome: BudgetOutcome
    estimated_cost: Decimal
    limit: Decimal
    current_spend: Decimal
    reason: str

    @property
    def allowed(self) -> bool:
        return self.outcome is BudgetOutcome.ALLOWED


class BudgetGuard:
    """Enforces spending limits before any LLM call is dispatched.

    The guard is stateless at the instance level: all state is in the database,
    queried through the session.

    **Concurrency limit, stated rather than assumed.** There is no reservation:
    spend becomes visible to another worker only when the transaction that
    wrote its `llm_calls` row commits. Two workers starting at the same instant
    therefore both see the pre-call total and both pass, so the ceilings are
    accurate to within one in-flight call per concurrent worker. That is
    acceptable while the worker pool is small and every per-call cost is capped
    by `MAX_COST_PER_CALL_USD`; it is not a guarantee, and a reservation table
    is the fix if the pool grows. Do not read this class as making concurrent
    overspend impossible.

    Usage:
        guard = BudgetGuard(session)
        decision = await guard.check_call(
            estimated_cost=Decimal("0.15"),
            document_id=doc_id,
        )
        if not decision.allowed:
            raise BudgetExceededError(decision)
    """

    def __init__(
        self,
        session: AsyncSession,
        settings: Any | None = None,
    ) -> None:
        self._session = session
        if settings is not None:
            self._settings = settings
        else:
            self._settings = get_settings()

    async def check_call(
        self,
        *,
        estimated_cost: Decimal,
        document_id: uuid.UUID | None = None,
        vlm_page_count: int = 0,
    ) -> BudgetDecision:
        """Check all applicable guards. First failure wins.

        Args:
            estimated_cost: Pre-dispatch cost estimate from cost.estimate_cost().
            document_id: The document this call is attributed to, for per-document cap.
            vlm_page_count: Number of VLM pages this call would process.
        """
        # 1. Per-call ceiling
        decision = self._check_per_call(estimated_cost)
        if not decision.allowed:
            return decision

        # 2. Global circuit breaker
        decision = await self._check_global(estimated_cost)
        if not decision.allowed:
            return decision

        # 3. Per-document ceiling
        if document_id is not None:
            decision = await self._check_per_document(estimated_cost, document_id)
            if not decision.allowed:
                return decision

        # 4. VLM page cap
        if vlm_page_count > 0:
            decision = self._check_vlm_pages(vlm_page_count)
            if not decision.allowed:
                return decision

        return BudgetDecision(
            outcome=BudgetOutcome.ALLOWED,
            estimated_cost=estimated_cost,
            limit=Decimal("0"),
            current_spend=Decimal("0"),
            reason="all guards passed",
        )

    # -- private guards -------------------------------------------------------

    def _check_per_call(self, estimated_cost: Decimal) -> BudgetDecision:
        limit = self._settings.MAX_COST_PER_CALL_USD
        if estimated_cost <= limit:
            return BudgetDecision(
                outcome=BudgetOutcome.ALLOWED,
                estimated_cost=estimated_cost,
                limit=limit,
                current_spend=Decimal("0"),
                reason="",
            )
        reason = f"estimated cost ${estimated_cost} exceeds per-call limit ${limit}"
        logger.warning("budget.rejected_per_call", extra={"reason": reason})
        return BudgetDecision(
            outcome=BudgetOutcome.REJECTED_PER_CALL,
            estimated_cost=estimated_cost,
            limit=limit,
            current_spend=Decimal("0"),
            reason=reason,
        )

    async def _check_global(self, estimated_cost: Decimal) -> BudgetDecision:
        limit = self._settings.MAX_TOTAL_COST_USD
        from app.db.repositories.ops import LLMCallRepository

        current = await LLMCallRepository(self._session).total_cost()
        projected = current + estimated_cost
        if projected <= limit:
            return BudgetDecision(
                outcome=BudgetOutcome.ALLOWED,
                estimated_cost=estimated_cost,
                limit=limit,
                current_spend=current,
                reason="",
            )
        reason = (
            f"global spend ${current} + ${estimated_cost} = ${projected} "
            f"would exceed global limit ${limit}"
        )
        logger.warning("budget.rejected_global", extra={"reason": reason})
        return BudgetDecision(
            outcome=BudgetOutcome.REJECTED_GLOBAL,
            estimated_cost=estimated_cost,
            limit=limit,
            current_spend=current,
            reason=reason,
        )

    async def _check_per_document(
        self, estimated_cost: Decimal, document_id: uuid.UUID
    ) -> BudgetDecision:
        limit = self._settings.MAX_COST_PER_DOCUMENT_USD

        # Sum both LLMCall and ExtractionJob spend for this document.
        from app.db.repositories.ops import (
            ExtractionJobRepository,
            LLMCallRepository,
        )

        calls_cost = await LLMCallRepository(self._session).total_cost_for_document(document_id)
        jobs_cost = await ExtractionJobRepository(self._session).total_cost_for_document(
            document_id
        )
        current = calls_cost + jobs_cost
        projected = current + estimated_cost

        if projected <= limit:
            return BudgetDecision(
                outcome=BudgetOutcome.ALLOWED,
                estimated_cost=estimated_cost,
                limit=limit,
                current_spend=current,
                reason="",
            )
        reason = (
            f"document {document_id} spend ${current} + ${estimated_cost} = "
            f"${projected} would exceed per-document limit ${limit}"
        )
        logger.warning(
            "budget.rejected_per_document",
            extra={"document_id": str(document_id), "reason": reason},
        )
        return BudgetDecision(
            outcome=BudgetOutcome.REJECTED_PER_DOCUMENT,
            estimated_cost=estimated_cost,
            limit=limit,
            current_spend=current,
            reason=reason,
        )

    def _check_vlm_pages(self, page_count: int) -> BudgetDecision:
        limit = self._settings.MAX_VLM_PAGES_PER_DOC
        if page_count <= limit:
            return BudgetDecision(
                outcome=BudgetOutcome.ALLOWED,
                estimated_cost=Decimal("0"),
                limit=Decimal(limit),
                current_spend=Decimal(page_count),
                reason="",
            )
        reason = f"{page_count} VLM pages exceeds cap of {limit}"
        logger.warning("budget.rejected_vlm_pages", extra={"reason": reason})
        return BudgetDecision(
            outcome=BudgetOutcome.REJECTED_VLM_PAGES,
            estimated_cost=Decimal("0"),
            limit=Decimal(limit),
            current_spend=Decimal(page_count),
            reason=reason,
        )


class BudgetExceededError(Exception):
    """Raised when the budget guard blocks a call.

    The `decision` carries the structured reason for logging and for the
    `budget_exceeded` status flags on documents and jobs.
    """

    def __init__(self, decision: BudgetDecision) -> None:
        self.decision = decision
        super().__init__(decision.reason)
