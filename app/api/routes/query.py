"""Ask the query agent a question over HTTP.

Until this router existed the agent was reachable only from the CLI, which made
it impossible to put a UI in front of the one thing the system is for. The
handler translates HTTP to `AgentQueryService` and back, and decides nothing
(CLAUDE.md 3, 9) -- intent, retrieval, refusal and citation verification all
happen inside the graph.

**A refusal is a 200.** When retrieval turns up no supporting clause the agent
answers "No supporting evidence in the corpus" with `confidence: 0.0`, and that
is a correct outcome, not an error (CLAUDE.md 1.5). Returning 404 or 422 there
would train a client to treat the system's most important safety behaviour as a
bug to be retried around. Callers branch on `refused`.

Endpoint:
    POST /query
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.service import AgentQueryService
from app.api.deps import CurrentUser
from app.core.logging import get_logger
from app.db.session import get_readonly_session, get_session
from app.domain.enums import QueryIntent
from app.domain.rules import Citation

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

MAX_QUESTION_CHARS = 2000


# -- request / response schemas ----------------------------------------------


class QueryRequest(BaseModel):
    """Just the question.

    `user_id` used to be a field here. It is now taken from the session, for
    the same reason `reviewer_id` was removed from the review queue: a
    `query_logs` row that records whoever the caller claimed to be is not an
    audit trail, it is a suggestion.
    """

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)


class QueryResponse(BaseModel):
    """What the agent concluded, and everything needed to audit the conclusion.

    `citations` carries the provenance for claims drawn from document text; the
    graph's verify node has already stripped any that did not trace to a chunk
    retrieved this turn (CLAUDE.md 1.2).

    It is legitimately **empty** for intents answered from structured rows
    rather than prose -- `covenant_breach_check` and `portfolio_query` report
    what the rules engine computed over `covenants` and `portfolio_holdings`,
    and cite no span because they read none. Those rows trace to clauses of
    their own, so the provenance exists; this response just does not carry it
    yet. A UI should link through the instrument rather than assume a citation.
    """

    question: str
    intent: QueryIntent
    answer: str
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    refused: bool
    tools_used: list[str]
    chunk_ids: list[str]
    request_id: str | None = None


# -- dependency --------------------------------------------------------------


async def get_agent_service(
    read_session: Annotated[AsyncSession, Depends(get_readonly_session)],
    log_session: Annotated[AsyncSession, Depends(get_session)],
) -> AsyncIterator[AgentQueryService]:
    """The agent, wired to the two roles it needs.

    Mirrors `open_agent_query_service()` rather than calling it, because FastAPI
    owns session lifetime here: the reads run on `DATABASE_URL_RO`, whose grants
    make CLAUDE.md 1.6 enforceable at the database, and `query_logs` /
    `audit_logs` go through the read-write role because that role is the only
    one that can write them.

    Two dependencies rather than one context manager also means the override
    hook used by tests can replace either half independently.
    """
    yield AgentQueryService(read_session, log_session=log_session)


# -- endpoint ----------------------------------------------------------------


@router.post("", response_model=QueryResponse, summary="Ask the query agent")
async def ask(
    body: QueryRequest,
    request: Request,
    user: CurrentUser,
    service: Annotated[AgentQueryService, Depends(get_agent_service)],
) -> QueryResponse:
    """Answer a question with citations, or refuse.

    The agent logs every question to `query_logs` regardless of outcome, so a
    refusal is as auditable as an answer.
    """
    request_id = getattr(request.state, "request_id", None)
    answer = await service.answer(
        body.question,
        user_id=user.username,
        request_id=request_id,
    )

    logger.info(
        "query.answered",
        extra={
            "intent": answer.intent.value,
            "refused": answer.refused,
            "confidence": answer.confidence,
            "citation_count": len(answer.citations),
            "user": user.username,
        },
    )

    return QueryResponse(
        question=answer.question,
        intent=answer.intent,
        answer=answer.answer,
        citations=answer.citations,
        confidence=answer.confidence,
        refused=answer.refused,
        tools_used=answer.tools_used,
        chunk_ids=answer.chunk_ids,
        request_id=request_id,
    )
