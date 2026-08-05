"""Agent query service — the public interface of the Phase 7 query agent.

Wraps the LangGraph graph and provides an async `answer()` method that takes a
question and returns an `AgentAnswer`. The graph runs inside the caller's
transaction, so the query log and audit entry commit atomically with the answer.

Usage:
    service = AgentQueryService(session)
    answer = await service.answer("What is the cross-default threshold?")
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.graph import AgentState, build_graph
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

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._graph: CompiledStateGraph[AgentState, None, AgentState, AgentState] = (
            build_graph()
        )

    async def answer(
        self,
        question: str,
        *,
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> AgentAnswer:
        """Answer a question through the LangGraph agent.

        The graph runs to completion — classify, retrieve, evaluate, synthesize,
        verify, log — and returns the final state. All nodes share the session,
        so the query log row is written inside the caller's transaction.
        """
        state = AgentState(
            question=question,
            user_id=user_id,
            request_id=request_id or str(uuid.uuid4()),
        )

        # LangGraph's .ainvoke() runs the graph to a terminal state. We pass
        # the session in the config so every node can access it.
        try:
            final_state = await self._graph.ainvoke(
                state,
                config={"configurable": {"session": self._session}},
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
