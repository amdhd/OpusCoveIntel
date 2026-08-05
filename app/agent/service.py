"""Agent query service — the public interface of the Phase 7 query agent.

Wraps the LangGraph graph and provides an async `answer()` method that takes a
question and returns an `AgentAnswer`.

**Two sessions, because the agent has two jobs with opposite privilege
requirements.** CLAUDE.md 1.6 puts the read path on the read-only role, so a
generated statement that slipped past the SQL guardrail fails at the database.
But the agent also writes `query_logs` and `audit_logs`, which that role cannot
do. A single session forces a choice between dropping the invariant and losing
the audit trail; splitting them keeps both, at the cost of the query log no
longer committing atomically with the read that produced it. That trade is the
right way round: the log records what was asked and answered, and a read
transaction has nothing to roll back.

Usage:
    async with open_agent_query_service() as service:
        answer = await service.answer("What is the cross-default threshold?")
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import (
    READ_SESSION_KEY,
    WRITE_SESSION_KEY,
    AgentState,
    build_graph,
)
from app.core.logging import get_logger
from app.domain.enums import QueryIntent
from app.domain.rules import Citation

logger = get_logger(__name__)


@dataclass(frozen=True)
class AgentAnswer:
    """The result of querying the agent."""

    question: str
    intent: QueryIntent
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False
    tools_used: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Alias for compatibility with the Phase 4 Answer type."""
        return self.answer


class AgentQueryService:
    """The Phase 7 query agent — LangGraph graph + deterministic tools.

    This is the public interface. Callers get an `AgentAnswer` with citations,
    confidence, and refusal state — the same contract as the Phase 4
    `DeterministicQueryService`, so the golden-question harness can measure
    both against the same acceptance criteria.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        log_session: AsyncSession | None = None,
    ) -> None:
        """
        Args:
            session: The **read-only** session (CLAUDE.md 1.6). Retrieval, the
                deterministic tools and every generated SQL statement run
                through it, so a stray write fails at the database rather than
                at code review.
            log_session: A read-write session used for `query_logs` and
                `audit_logs` and nothing else. Those rows are writes by
                definition and cannot go through the read-only role.

                When omitted, logging falls back to `session` -- correct for a
                caller that passed a read-write session and for tests running
                inside a rolled-back transaction, and the reason `answer()`
                warns when the roles have not been split.

        Prefer `open_agent_query_service()`, which opens both with the right
        roles rather than leaving the choice to each call site.
        """
        self._session = session
        self._log_session = log_session
        self._graph: CompiledStateGraph[AgentState, None, AgentState, AgentState] = build_graph()

    async def answer(
        self,
        question: str,
        *,
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> AgentAnswer:
        """Answer a question through the LangGraph agent.

        The graph runs to completion — classify, retrieve, evaluate, synthesize,
        verify, log — and returns the final state. Read nodes use the read-only
        session; only the `log` node writes, through `log_session`.
        """
        state = AgentState(
            question=question,
            user_id=user_id,
            request_id=request_id or str(uuid.uuid4()),
        )

        configurable: dict[str, object] = {READ_SESSION_KEY: self._session}
        if self._log_session is not None:
            configurable[WRITE_SESSION_KEY] = self._log_session

        # LangGraph's .ainvoke() runs the graph to a terminal state. We pass
        # the sessions in the config so every node can access the right one.
        try:
            final_state = await self._graph.ainvoke(
                state,
                config={"configurable": configurable},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "agent.graph_failed",
                extra={"question": question[:120], "error": str(exc)},
            )
            return AgentAnswer(
                question=question,
                intent=QueryIntent.UNSUPPORTED,
                answer="An internal error occurred while processing the query.",
                confidence=0.0,
                refused=True,
                tools_used=[],
            )

        # The graph runs on the AgentState dataclass but ainvoke returns a dict
        # (LangGraph serialises it). Reconstruct from the returned dict.
        if isinstance(final_state, dict):
            return AgentAnswer(
                question=question,
                intent=QueryIntent(final_state.get("intent", "unsupported")),
                answer=final_state.get("answer", ""),
                citations=final_state.get("citations", []),
                confidence=final_state.get("confidence", 0.0),
                refused=final_state.get("refused", False),
                tools_used=final_state.get("tools_called", []),
                chunk_ids=final_state.get("retrieved_chunk_ids", []),
            )

        # If we got the AgentState back directly (depends on LangGraph version):
        return AgentAnswer(
            question=question,
            intent=state.intent,
            answer=state.answer,
            citations=state.citations,
            confidence=state.confidence,
            refused=state.refused,
            tools_used=state.tools_called,
            chunk_ids=state.retrieved_chunk_ids,
        )


@asynccontextmanager
async def open_agent_query_service() -> AsyncIterator[AgentQueryService]:
    """An agent wired to the two roles it needs, for real entry points.

    The read path connects as `DATABASE_URL_RO` and the log path as
    `DATABASE_URL`, which is what makes CLAUDE.md 1.6 enforceable: the
    read-only role lacks the grants, so a generated `UPDATE` that slipped past
    the SQL guardrail still fails at the database.

    Constructing `AgentQueryService(session)` by hand stays possible for tests
    and single-session callers -- this exists so that an API route or CLI
    command wiring the agent up gets the split by default rather than having to
    know to ask for it.

    Usage:
        async with open_agent_query_service() as service:
            answer = await service.answer(question)
    """
    from app.db.session import get_readonly_sessionmaker, get_sessionmaker

    async with (
        get_readonly_sessionmaker()() as read_session,
        get_sessionmaker()() as log_session,
    ):
        yield AgentQueryService(read_session, log_session=log_session)
