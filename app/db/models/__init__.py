"""SQLAlchemy models.

Importing this package registers every table on `Base.metadata`, which is what
Alembic autogenerate reflects against. A model that is not imported here is
invisible to migrations -- add new modules to the imports below.
"""

from __future__ import annotations

from app.db.base import Base
from app.db.models.clauses import CallSchedule, Clause, Covenant, RatingTrigger
from app.db.models.documents import Document, DocumentChunk, DocumentPage
from app.db.models.instruments import Instrument, SukukStructure
from app.db.models.ops import (
    AuditLog,
    ExtractionJob,
    HumanReview,
    LLMCache,
    LLMCall,
    QueryLog,
)
from app.db.models.portfolio import Portfolio, PortfolioHolding

__all__ = [
    "AuditLog",
    "Base",
    "CallSchedule",
    "Clause",
    "Covenant",
    "Document",
    "DocumentChunk",
    "DocumentPage",
    "ExtractionJob",
    "HumanReview",
    "Instrument",
    "LLMCache",
    "LLMCall",
    "Portfolio",
    "PortfolioHolding",
    "QueryLog",
    "RatingTrigger",
    "SukukStructure",
]
