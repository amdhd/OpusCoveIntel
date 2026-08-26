"""The deterministic query path, and the golden set it is measured against.

docs/plan.md, Phase 4 acceptance: **≥6 of the original 10 golden questions answered
with zero LLM calls**, now ≥9 of 13 -- the three Phase 10 additions are
refusals, so the bar moved with the set. There is no LLM adapter in the
codebase yet, so "zero LLM calls" is structural rather than mocked -- nothing
here could spend money if it tried.

The other half of what is tested is refusal. A system that answers everything
is worse than one that answers less and says so (CLAUDE.md 1.5).
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import QueryIntent
from app.evals.golden import GOLDEN_QUESTIONS, PHASE_4_TARGET, GoldenQuestion
from app.query.service import NO_EVIDENCE, Answer, DeterministicQueryService

pytestmark = pytest.mark.usefixtures("storage_root")


def service(session: AsyncSession) -> DeterministicQueryService:
    return DeterministicQueryService(session)


def passes(answer: Answer, case: GoldenQuestion) -> bool:
    missing = [needle for needle in case.must_contain if needle.lower() not in answer.text.lower()]
    return (
        answer.intent is case.expected_intent
        and answer.refused == case.expect_refusal
        and not missing
        and len(answer.citations) >= case.min_citations
    )


# -- the acceptance criterion ----------------------------------------------


async def test_the_golden_set_meets_the_phase_4_target(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    answered = 0
    failures: list[str] = []

    for case in GOLDEN_QUESTIONS:
        answer = await service(db_session).answer(case.question)
        if passes(answer, case):
            answered += 1
        else:
            failures.append(f"{case.id}: intent={answer.intent.value} text={answer.text[:80]!r}")

    assert answered >= PHASE_4_TARGET, f"{answered}/{len(GOLDEN_QUESTIONS)}; {failures}"


@pytest.mark.parametrize("case", GOLDEN_QUESTIONS, ids=lambda case: case.id)
async def test_each_golden_question_individually(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID], case: GoldenQuestion
) -> None:
    """Reported per question, so a regression names the one that broke.

    The suite-level bar is the aggregate above; this is the diagnostic.
    """
    answer = await service(db_session).answer(case.question)

    assert passes(answer, case), (
        f"{case.id} intent={answer.intent.value} "
        f"citations={len(answer.citations)} text={answer.text[:200]!r}"
    )


# -- refusal ---------------------------------------------------------------


async def test_a_request_for_advice_is_refused(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    answer = await service(db_session).answer("Should we buy more Malaysian sukuk next quarter?")

    assert answer.refused
    assert answer.confidence == 0.0
    assert answer.citations == []
    assert "does not forecast" in answer.text


async def test_a_question_with_no_evidence_is_refused_rather_than_answered(
    db_session: AsyncSession, seeded_universe: None
) -> None:
    # An empty corpus: nothing ingested, so nothing can be cited.
    answer = await service(db_session).answer("What does the trust deed say about insurance?")

    assert answer.refused
    assert answer.text == NO_EVIDENCE
    assert answer.confidence == 0.0


# -- breach checking -------------------------------------------------------


async def test_a_breach_check_computes_rather_than_retrieves(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    answer = await service(db_session).answer(
        "Which holdings would breach their rating trigger at the current rating?"
    )

    assert answer.intent is QueryIntent.COVENANT_BREACH_CHECK
    assert "evaluate_covenant_rule" in answer.tools_used
    assert "BREACH" in answer.text or "No covenant breaches" in answer.text


async def test_the_rating_comparison_is_ordinal(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    answer = await service(db_session).answer("Which instruments are rated below A?")

    # A- and BBB+ are below A; AA3 (RAM) is AA-, which is not -- though it
    # sorts before "A" as a string.
    assert "A-" in answer.text
    assert "BBB+" in answer.text
    assert "AA3" not in answer.text


async def test_covenants_that_cannot_be_evaluated_are_named_not_hidden(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    answer = await service(db_session).answer("Are any covenants breached?")

    # No financial facts are held, so ratio covenants cannot be evaluated.
    # Dropping them silently from a breach report would be the most dangerous
    # output this system could produce.
    assert "Not evaluated" in answer.text
    assert "not reported" in answer.text


# -- citations -------------------------------------------------------------


async def test_a_covenant_answer_cites_a_page_and_a_quote(
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


async def test_a_document_search_answer_cites_the_chunks_it_used(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    answer = await service(db_session).answer("Who is the trustee for the programme?")

    assert answer.intent is QueryIntent.DOCUMENT_SEARCH
    assert answer.chunk_ids
    assert answer.confidence < 0.8  # retrieval is weaker evidence than extraction


# -- portfolio -------------------------------------------------------------


async def test_portfolio_aggregation_is_sql_not_prose(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    answer = await service(db_session).answer(
        "What is the total exposure of the Green Fixed Income Fund portfolio?"
    )

    assert answer.intent is QueryIntent.PORTFOLIO_QUERY
    assert "run_read_only_sql" in answer.tools_used
    assert "Green Fixed Income Fund" in answer.text
    assert "RM" in answer.text


async def test_portfolio_exposure_can_be_filtered_by_rating(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    answer = await service(db_session).answer(
        "What is our portfolio exposure to holdings rated below A?"
    )

    assert "rated below A" in answer.text
    assert not answer.refused
