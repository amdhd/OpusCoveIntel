"""`POST /query` — the agent's HTTP surface.

The agent itself is covered by test_agent.py; these tests are about the
contract the route exposes, and about the two behaviours a UI will depend on
that are easy to get wrong at the HTTP boundary:

* a refusal is a **200** with `refused: true`, not a 404 (CLAUDE.md 1.5), and
* an answer carries citations that trace to real chunks (CLAUDE.md 1.2).
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.ops import QueryLog
from app.domain.enums import QueryIntent
from app.evals.golden import GOLDEN_QUESTIONS

pytestmark = pytest.mark.usefixtures("storage_root")


def _unanswerable() -> str:
    """The golden question with no support in the corpus."""
    refusals = [case for case in GOLDEN_QUESTIONS if case.expect_refusal]
    assert refusals, "the golden set must keep at least one unanswerable question"
    return refusals[0].question


def _answerable() -> str:
    answerables = [
        case for case in GOLDEN_QUESTIONS if not case.expect_refusal and case.min_citations > 0
    ]
    assert answerables, "the golden set must keep at least one cited question"
    return answerables[0].question


class TestValidation:
    async def test_an_empty_question_is_rejected(self, api_client: AsyncClient) -> None:
        response = await api_client.post("/query", json={"question": ""})
        assert response.status_code == 422

    async def test_an_oversized_question_is_rejected(self, api_client: AsyncClient) -> None:
        response = await api_client.post("/query", json={"question": "x" * 2001})
        assert response.status_code == 422


class TestAnswering:
    async def test_a_supported_question_answers_with_citations(
        self, api_client: AsyncClient, indexed_corpus: list[uuid.UUID]
    ) -> None:
        response = await api_client.post("/query", json={"question": _answerable()})

        assert response.status_code == 200
        body = response.json()
        assert body["refused"] is False
        assert body["answer"]
        assert body["citations"], "an answered question must cite something"
        assert body["confidence"] > 0.0

        # CLAUDE.md 1.2: a citation names the page and quotes the text.
        for citation in body["citations"]:
            assert citation["page_number"] >= 1
            assert citation["quote"]
            assert citation["document_id"]

    async def test_the_response_echoes_the_request_id(
        self, api_client: AsyncClient, indexed_corpus: list[uuid.UUID]
    ) -> None:
        """A UI needs to correlate an answer with its row in `query_logs`."""
        response = await api_client.post(
            "/query",
            json={"question": _answerable()},
            headers={"X-Request-ID": "ui-correlation-probe"},
        )
        assert response.status_code == 200
        assert response.json()["request_id"] == "ui-correlation-probe"
        assert response.headers["X-Request-ID"] == "ui-correlation-probe"


class TestRefusal:
    async def test_an_unsupported_question_refuses_with_200(
        self, api_client: AsyncClient, indexed_corpus: list[uuid.UUID]
    ) -> None:
        """A refusal is a correct outcome, so it must not look like a failure.

        If this ever returns 4xx, a client will reasonably treat the system's
        central safety behaviour as an error to retry around.
        """
        response = await api_client.post("/query", json={"question": _unanswerable()})

        assert response.status_code == 200
        body = response.json()
        assert body["refused"] is True
        assert body["confidence"] == 0.0
        assert body["intent"] == QueryIntent.UNSUPPORTED.value


class TestAuditTrail:
    async def test_every_question_is_logged(
        self,
        api_client: AsyncClient,
        db_session: AsyncSession,
        indexed_corpus: list[uuid.UUID],
    ) -> None:
        """Including refusals — an unlogged question is an unauditable one."""
        before = len((await db_session.scalars(select(QueryLog))).all())

        await api_client.post("/query", json={"question": _answerable()})
        await api_client.post("/query", json={"question": _unanswerable()})

        after = (await db_session.scalars(select(QueryLog))).all()
        assert len(after) == before + 2
