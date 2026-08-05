"""Phase 7 query agent — LangGraph graph + deterministic tools + SQL guardrail.

The graph wraps the deterministic Phase 4 service and adds LLM synthesis
for intents where prose is better than structured rows. The verify node
is the structural guard: every factual claim must trace to a citation.
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
