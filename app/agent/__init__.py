"""Phase 7 query agent — LangGraph graph + deterministic tools + SQL guardrail.

**Every node is deterministic. This package makes no LLM calls at all**, and
imports nothing from `app/llm/`. What the graph adds over the Phase 4 service
is structure, not language: an intent-directed plan, tool orchestration, a
verify node that strips any claim not traceable to a retrieved clause, and a
logged, audited record of every question.

CLAUDE.md's routing table reserves `claude-opus-5` for answer synthesis. That
is a designed-for position, not the current one -- see `_synthesize` in
`graph.py` for what actually runs and what changes if a model ever takes it
over.
"""

from __future__ import annotations

from app.agent.graph import build_graph
from app.agent.service import AgentQueryService
from app.agent.sql_guard import SQLGuardError, SQLGuardResult, validate_sql
from app.agent.tools import ToolResult

__all__ = [
    "AgentQueryService",
    "SQLGuardError",
    "SQLGuardResult",
    "ToolResult",
    "build_graph",
    "validate_sql",
]
