"""Operational repositories: jobs, LLM spend, cache, review queue, audit."""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from decimal import Decimal

from sqlalchemy import func, select

from app.db.models.ops import AuditLog, ExtractionJob, HumanReview, LLMCache, LLMCall
from app.db.repositories.base import BaseRepository
from app.domain.enums import JobStatus, JobType, ReviewStatus


class ExtractionJobRepository(BaseRepository[ExtractionJob]):
    model = ExtractionJob

    async def find_by_identity(
        self,
        *,
        document_sha256: str,
        job_type: JobType,
        prompt_version: str,
        model_id: str,
        extractor_version: str,
    ) -> ExtractionJob | None:
        """Look up the extraction identity (CLAUDE.md 1.7).

        Callers use this to skip work that has already been done. The matching
        unique constraint means two racing workers cannot both insert.
        """
        result = await self.session.execute(
            select(ExtractionJob).where(
                ExtractionJob.document_sha256 == document_sha256,
                ExtractionJob.job_type == job_type,
                ExtractionJob.prompt_version == prompt_version,
                ExtractionJob.model_id == model_id,
                ExtractionJob.extractor_version == extractor_version,
            )
        )
        return result.scalar_one_or_none()

    async def claim_next(self, job_type: JobType) -> ExtractionJob | None:
        """Claim the oldest queued job of a type, or return None if there is none.

        `FOR UPDATE ... SKIP LOCKED` is what lets several workers poll the same
        table without a broker (CLAUDE.md 9 defers Celery/Redis to Phase 8):
        each worker locks a different row instead of contending for the first.

        The claim is only durable once the caller commits -- repositories do
        not commit -- so the transaction must be closed before long work starts.
        """
        result = await self.session.execute(
            select(ExtractionJob)
            .where(
                ExtractionJob.job_type == job_type,
                ExtractionJob.status == JobStatus.QUEUED,
            )
            .order_by(ExtractionJob.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        job = result.scalar_one_or_none()
        if job is None:
            return None

        job.status = JobStatus.RUNNING
        job.started_at = dt.datetime.now(dt.UTC)
        await self.session.flush()
        return job

    async def total_cost_for_document(self, document_id: uuid.UUID) -> Decimal:
        """Spend so far on one document -- what the per-document cap tests."""
        result = await self.session.execute(
            select(func.coalesce(func.sum(ExtractionJob.estimated_cost_usd), 0)).where(
                ExtractionJob.document_id == document_id
            )
        )
        return Decimal(result.scalar_one())


class LLMCallRepository(BaseRepository[LLMCall]):
    model = LLMCall

    async def total_cost(self) -> Decimal:
        """Global spend -- what the circuit breaker tests (PLAN.md 2)."""
        result = await self.session.execute(
            select(func.coalesce(func.sum(LLMCall.estimated_cost_usd), 0))
        )
        return Decimal(result.scalar_one())

    async def total_cost_for_document(self, document_id: uuid.UUID) -> Decimal:
        result = await self.session.execute(
            select(func.coalesce(func.sum(LLMCall.estimated_cost_usd), 0)).where(
                LLMCall.document_id == document_id
            )
        )
        return Decimal(result.scalar_one())

    async def cost_by_stage(self) -> dict[str, Decimal]:
        """Spend attributed by pipeline stage -- the body of `make cost-report`."""
        result = await self.session.execute(
            select(LLMCall.stage, func.sum(LLMCall.estimated_cost_usd)).group_by(LLMCall.stage)
        )
        # `.value` rather than `str()`: both work for a StrEnum, but this states
        # that the key is the stored vocabulary term ("extract"), not a repr.
        return {stage.value: Decimal(total) for stage, total in result.all()}


class LLMCacheRepository(BaseRepository[LLMCache]):
    model = LLMCache

    async def get_by_key(self, cache_key: str) -> LLMCache | None:
        result = await self.session.execute(select(LLMCache).where(LLMCache.cache_key == cache_key))
        return result.scalar_one_or_none()

    async def record_hit(self, entry: LLMCache) -> LLMCache:
        entry.hit_count += 1
        await self.session.flush()
        return entry


class HumanReviewRepository(BaseRepository[HumanReview]):
    model = HumanReview

    async def list_pending(self, *, limit: int = 100) -> Sequence[HumanReview]:
        result = await self.session.execute(
            select(HumanReview)
            .where(HumanReview.status == ReviewStatus.PENDING)
            .order_by(HumanReview.created_at)
            .limit(limit)
        )
        return result.scalars().all()

    async def count_pending(self) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(HumanReview)
            .where(HumanReview.status == ReviewStatus.PENDING)
        )
        return int(result.scalar_one())


class AuditLogRepository(BaseRepository[AuditLog]):
    model = AuditLog

    async def list_for_entity(
        self, entity_type: str, entity_id: uuid.UUID, *, limit: int = 100
    ) -> Sequence[AuditLog]:
        result = await self.session.execute(
            select(AuditLog)
            .where(AuditLog.entity_type == entity_type, AuditLog.entity_id == entity_id)
            .order_by(AuditLog.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
