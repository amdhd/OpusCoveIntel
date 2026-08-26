"""Spend attribution and rules-vs-LLM agreement.

Both metrics aggregate what the pipeline already recorded rather than
recomputing it:

* **Cost** comes from `llm_calls`, the ledger every provider call is written to
  by `app/llm/router.py`. There is no other source -- a cost figure derived
  anywhere else would be an estimate of the thing the ledger measures.
* **Agreement** comes from the review queue. `ExtractionPipeline._fields_disagree`
  decides, per extraction, whether the two extractors materially differ, and
  writes a `RULE_LLM_DISAGREEMENT` review when they do. Re-implementing that
  comparison here would produce a second, subtly different definition of
  disagreement, and the number would then describe neither run.

**Both report "no data" rather than a flattering default.** A corpus with no
LLM extraction has no agreement rate -- not 100 per cent -- and a system that
has never called a provider has no cost per document. docs/plan.md 2's whole posture
is that unmeasured spend is the danger; a harness that prints $0.00 for
"nobody has looked" would be part of the problem.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from decimal import Decimal

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.models.clauses import Clause, Covenant
from app.db.models.documents import Document
from app.db.models.ops import HumanReview, LLMCall
from app.db.repositories.ops import LLMCallRepository
from app.domain.enums import ExtractionMethod, ReviewTrigger
from app.evals.metrics import ratio


@dataclass(frozen=True)
class DocumentCost:
    document_id: uuid.UUID
    filename: str
    calls: int
    cache_hits: int
    cost_usd: Decimal
    cache_read_tokens: int

    def as_dict(self) -> dict[str, object]:
        return {
            "document_id": str(self.document_id),
            "filename": self.filename,
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            # Serialised as a string: JSON has no decimal type, and a threshold
            # that round-trips through a float is no longer the number the
            # budget guard compared against (CLAUDE.md 6).
            "cost_usd": str(self.cost_usd),
            "cache_read_tokens": self.cache_read_tokens,
        }


@dataclass
class CostReport:
    """Spend, attributed. The body of `make cost-report` and a section of `make eval`."""

    total_usd: Decimal = Decimal("0")
    by_stage: dict[str, Decimal] = field(default_factory=dict)
    by_document: list[DocumentCost] = field(default_factory=list)
    calls: int = 0
    cache_hits: int = 0
    cache_read_tokens: int = 0
    budget_total_usd: Decimal = Decimal("0")
    budget_per_document_usd: Decimal = Decimal("0")

    @property
    def has_spend(self) -> bool:
        return self.calls > 0

    @property
    def cost_per_document(self) -> Decimal | None:
        """Mean spend per document that actually cost something.

        Averaged over documents with spend, not over the corpus: a corpus of a
        thousand never-extracted documents would otherwise report a cost per
        document of nearly zero while a single extraction ran over budget.
        """
        charged = [item for item in self.by_document if item.cost_usd > 0]
        if not charged:
            return None
        return sum((item.cost_usd for item in charged), Decimal("0")) / len(charged)

    @property
    def cache_hit_rate(self) -> float | None:
        return ratio(self.cache_hits, self.calls)

    @property
    def over_budget_documents(self) -> list[DocumentCost]:
        return [item for item in self.by_document if item.cost_usd > self.budget_per_document_usd]

    def as_dict(self) -> dict[str, object]:
        return {
            "total_usd": str(self.total_usd),
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hit_rate,
            "cache_read_tokens": self.cache_read_tokens,
            "cost_per_document": (
                str(self.cost_per_document) if self.cost_per_document is not None else None
            ),
            "by_stage": {stage: str(amount) for stage, amount in sorted(self.by_stage.items())},
            "by_document": [item.as_dict() for item in self.by_document],
            "budget": {
                "total_usd": str(self.budget_total_usd),
                "per_document_usd": str(self.budget_per_document_usd),
                "remaining_usd": str(self.budget_total_usd - self.total_usd),
                "over_budget_documents": [item.filename for item in self.over_budget_documents],
            },
        }


@dataclass
class AgreementReport:
    """How often the two extractors said the same thing.

    `llm_covenants` is the denominator because agreement is only defined where
    the LLM produced something to agree with -- the rule extractor runs over
    every chunk, so scaling by its output would dilute the rate with spans the
    LLM was never shown.
    """

    llm_covenants: int = 0
    rule_covenants: int = 0
    disagreements: int = 0

    @property
    def agreement_rate(self) -> float | None:
        rate = ratio(self.llm_covenants - self.disagreements, self.llm_covenants)
        return rate if rate is None else max(rate, 0.0)

    @property
    def measurable(self) -> bool:
        return self.llm_covenants > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "measurable": self.measurable,
            "llm_covenants": self.llm_covenants,
            "rule_covenants": self.rule_covenants,
            "disagreements": self.disagreements,
            "agreement_rate": self.agreement_rate,
        }


class CostEvaluator:
    def __init__(self, session: AsyncSession, settings: Settings | None = None) -> None:
        self._session = session
        self._settings = settings or get_settings()
        self._calls = LLMCallRepository(session)

    async def report(self) -> CostReport:
        report = CostReport(
            budget_total_usd=self._settings.MAX_TOTAL_COST_USD,
            budget_per_document_usd=self._settings.MAX_COST_PER_DOCUMENT_USD,
        )
        report.total_usd = await self._calls.total_cost()
        report.by_stage = await self._calls.cost_by_stage()

        # Counted with SUM(CASE ...) rather than a FILTER clause so the same
        # statement works on any backend the suite might run against.
        cache_hit_count = func.coalesce(
            func.sum(case((LLMCall.cache_hit.is_(True), 1), else_=0)), 0
        )

        totals = await self._session.execute(
            select(
                func.count(LLMCall.id),
                cache_hit_count,
                func.coalesce(func.sum(LLMCall.cache_read_tokens), 0),
            )
        )
        calls, cache_hits, cache_read_tokens = totals.one()
        report.calls = int(calls)
        report.cache_hits = int(cache_hits)
        report.cache_read_tokens = int(cache_read_tokens)

        rows = await self._session.execute(
            select(
                LLMCall.document_id,
                Document.filename,
                func.count(LLMCall.id),
                cache_hit_count,
                func.coalesce(func.sum(LLMCall.estimated_cost_usd), 0),
                func.coalesce(func.sum(LLMCall.cache_read_tokens), 0),
            )
            .join(Document, Document.id == LLMCall.document_id, isouter=True)
            .group_by(LLMCall.document_id, Document.filename)
            .order_by(func.coalesce(func.sum(LLMCall.estimated_cost_usd), 0).desc())
        )
        for document_id, filename, call_count, hits, cost, cache_tokens in rows.all():
            if document_id is None:
                continue
            report.by_document.append(
                DocumentCost(
                    document_id=document_id,
                    filename=filename or "(deleted document)",
                    calls=int(call_count),
                    cache_hits=int(hits),
                    cost_usd=Decimal(cost),
                    cache_read_tokens=int(cache_tokens),
                )
            )
        return report

    async def agreement(self) -> AgreementReport:
        """Rules-vs-LLM agreement, aggregated over what the pipeline recorded."""
        llm_count = await self._count_covenants(ExtractionMethod.LLM)
        rule_count = await self._count_covenants(ExtractionMethod.RULE)
        disagreements = await self._session.execute(
            select(func.count())
            .select_from(HumanReview)
            .where(HumanReview.trigger_reason == ReviewTrigger.RULE_LLM_DISAGREEMENT)
        )
        return AgreementReport(
            llm_covenants=llm_count,
            rule_covenants=rule_count,
            disagreements=int(disagreements.scalar_one()),
        )

    async def _count_covenants(self, method: ExtractionMethod) -> int:
        result = await self._session.execute(
            select(func.count())
            .select_from(Covenant)
            .join(Clause, Covenant.clause_id == Clause.id)
            .where(Covenant.method == method)
        )
        return int(result.scalar_one())
