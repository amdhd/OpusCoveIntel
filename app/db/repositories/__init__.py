"""Repository layer.

Repositories own SQL. Services own transactions. Route handlers own neither
(CLAUDE.md 3, 9).
"""

from __future__ import annotations

from app.db.repositories.base import BaseRepository
from app.db.repositories.clauses import (
    CallScheduleRepository,
    ClauseRepository,
    CovenantRepository,
    RatingTriggerRepository,
)
from app.db.repositories.documents import (
    DocumentChunkRepository,
    DocumentPageRepository,
    DocumentRepository,
)
from app.db.repositories.instruments import InstrumentRepository, SukukStructureRepository
from app.db.repositories.ops import (
    AuditLogRepository,
    ExtractionJobRepository,
    HumanReviewRepository,
    LLMCacheRepository,
    LLMCallRepository,
)
from app.db.repositories.portfolio import PortfolioHoldingRepository, PortfolioRepository

__all__ = [
    "AuditLogRepository",
    "BaseRepository",
    "CallScheduleRepository",
    "ClauseRepository",
    "CovenantRepository",
    "DocumentChunkRepository",
    "DocumentPageRepository",
    "DocumentRepository",
    "ExtractionJobRepository",
    "HumanReviewRepository",
    "InstrumentRepository",
    "LLMCacheRepository",
    "LLMCallRepository",
    "PortfolioHoldingRepository",
    "PortfolioRepository",
    "RatingTriggerRepository",
    "SukukStructureRepository",
]
