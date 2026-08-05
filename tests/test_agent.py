"""Phase 7 agent tests — the LangGraph agent with deterministic tools.

PLAN.md, Phase 7 acceptance: "≥8/10 golden questions answered with correct
citations · agent refuses when evidence is absent."

These tests run the AgentQueryService against the same indexed_corpus fixture
used by the Phase 4 deterministic tests. The agent wraps the deterministic path,
so the golden questions that pass in Phase 4 (≥6/10) must still pass here.

The acceptance bar is **≥8/10** — the two above the Phase 4 baseline are
exercised by the agent's verify node and query logging.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.service import AgentAnswer, AgentQueryService
from app.db.models.ops import QueryLog
from app.domain.enums import QueryIntent
from app.evals.golden import GOLDEN_QUESTIONS, PHASE_7_TARGET, GoldenQuestion

pytestmark = pytest.mark.usefixtures("storage_root")


def service(session: AsyncSession) -> AgentQueryService:
    return AgentQueryService(session)


def passes(answer: AgentAnswer, case: GoldenQuestion) -> bool:

    missing = [
        needle for needle in case.must_contain if needle.lower() not in answer.answer.lower()
    ]
    return (
        answer.intent is case.expected_intent
        and answer.refused == case.expect_refusal
        and not missing
        and len(answer.citations) >= case.min_citations
    )


# -- golden set -------------------------------------------------------------


async def test_the_agent_meets_the_phase_7_golden_target(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """The aggregate bar: ≥8/10 golden questions answered correctly."""
    answered = 0
    failures: list[str] = []

    for case in GOLDEN_QUESTIONS:
        answer = await service(db_session).answer(case.question)
        if passes(answer, case):
            answered += 1
        else:
            failures.append(f"{case.id}: intent={answer.intent.value} text={answer.answer[:80]!r}")

    assert answered >= PHASE_7_TARGET, f"{answered}/{len(GOLDEN_QUESTIONS)}; failures: {failures}"


@pytest.mark.parametrize("case", GOLDEN_QUESTIONS, ids=lambda case: case.id)
async def test_each_golden_question_individually(
    db_session: AsyncSession,
    indexed_corpus: list[uuid.UUID],
    case: GoldenQuestion,
) -> None:
    """Per-question diagnostic so a regression names the one that broke."""
    answer = await service(db_session).answer(case.question)

    assert passes(answer, case), (
        f"{case.id} intent={answer.intent.value} "
        f"citations={len(answer.citations)} "
        f"refused={answer.refused} "
        f"text={answer.answer[:200]!r}"
    )


# -- refusal ----------------------------------------------------------------


async def test_the_agent_refuses_investment_advice(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    answer = await service(db_session).answer("Should we buy more Malaysian sukuk next quarter?")
    assert answer.refused
    assert answer.confidence == 0.0
    assert answer.intent is QueryIntent.UNSUPPORTED


async def test_the_agent_refuses_when_no_evidence_exists(
    db_session: AsyncSession, seeded_universe: None
) -> None:
    """An empty corpus: nothing ingested, so nothing can be cited."""
    answer = await service(db_session).answer("What does the trust deed say about insurance?")
    assert answer.refused
    assert answer.confidence == 0.0


# -- citations --------------------------------------------------------------


async def test_a_covenant_answer_has_citations(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    answer = await service(db_session).answer(
        "What is the cross-default threshold for Synthetic Green Energy Sdn Bhd?"
    )
    assert answer.citations
    citation = answer.citations[0]
    assert citation.page_number >= 1
    assert citation.quote
    assert citation.clause_id is not None


# -- breach checking --------------------------------------------------------


async def test_a_breach_check_computes_rather_than_retrieves(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    answer = await service(db_session).answer(
        "Which holdings would breach their rating trigger at the current rating?"
    )
    assert answer.intent is QueryIntent.COVENANT_BREACH_CHECK
    assert "evaluate_covenant_rule" in answer.tools_used


# -- query logging ----------------------------------------------------------


async def test_every_agent_query_is_logged(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """Phase 7 acceptance: the agent writes a QueryLog row."""
    await service(db_session).answer(
        "What is the gearing covenant for Synthetic Green Energy Sdn Bhd?"
    )

    # Query log must exist
    result = await db_session.execute(
        select(QueryLog).order_by(QueryLog.created_at.desc()).limit(1)
    )
    log_entry = result.scalar_one_or_none()
    assert log_entry is not None
    assert "gearing" in log_entry.question.lower()
    assert log_entry.intent is not None
    assert isinstance(log_entry.tools_called, list)


async def test_a_refusal_is_logged_with_refused_true(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    await service(db_session).answer("Should we short everything?")

    result = await db_session.execute(
        select(QueryLog).order_by(QueryLog.created_at.desc()).limit(1)
    )
    log_entry = result.scalar_one_or_none()
    assert log_entry is not None
    assert log_entry.refused is True


async def test_a_query_log_records_citations(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    await service(db_session).answer(
        "What is the cross-default threshold for Synthetic Green Energy Sdn Bhd?"
    )

    result = await db_session.execute(
        select(QueryLog).order_by(QueryLog.created_at.desc()).limit(1)
    )
    log_entry = result.scalar_one_or_none()
    assert log_entry is not None
    # Citations should be recorded as JSON
    assert isinstance(log_entry.citations_json, list)


# -- verify node ------------------------------------------------------------


async def test_the_verify_node_ensures_citations_are_traceable(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """Every citation from the agent must carry a traceable source."""
    answer = await service(db_session).answer(
        "What is the negative pledge covenant for Synthetic Green Energy Sdn Bhd?"
    )

    if answer.citations:
        for citation in answer.citations:
            assert citation.page_number >= 1
            assert citation.quote
            # Either a clause_id (structured) or a chunk_id (retrieval) must be set
            assert citation.clause_id is not None or citation.chunk_id is not None


class TestCovenantLookupIsNarrowedToTheQuestion:
    """The agent must not answer a question about one covenant with all of them.

    `get_covenants` has always taken a `covenant_type` filter and the graph
    never passed one, so "what is the cross-default threshold?" returned every
    covenant in the corpus and never named the threshold -- a worse answer than
    the Phase 4 service the agent wraps. Found by running the CLI, not by any
    test: the golden set has no question that names a covenant type.
    """

    async def test_a_named_covenant_type_narrows_the_answer(
        self, db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
    ) -> None:
        answer = await service(db_session).answer("What is the cross-default threshold?")

        assert answer.intent is QueryIntent.COVENANT_LOOKUP
        assert "cross_default" in answer.answer
        for unrelated in ("gearing_ratio", "shariah_non_compliance", "negative_pledge"):
            assert unrelated not in answer.answer, f"{unrelated} is not what was asked about"

    async def test_the_agent_matches_the_path_it_wraps(
        self, db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
    ) -> None:
        """Phase 7 wraps Phase 4; it should not answer the same question worse."""
        from app.query.service import DeterministicQueryService

        question = "What is the cross-default threshold?"
        agent = await service(db_session).answer(question)
        deterministic = await DeterministicQueryService(db_session).answer(question)

        assert agent.intent is deterministic.intent
        assert ("RM30" in agent.answer) == ("RM30" in deterministic.text)

    async def test_a_question_naming_no_covenant_type_still_returns_everything(
        self, db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
    ) -> None:
        """Narrowing must not become a filter that hides covenants when none is named."""
        answer = await service(db_session).answer("What covenants are there?")

        assert answer.intent is QueryIntent.COVENANT_LOOKUP
        assert answer.citations
