"""Phase 7 agent tests — the LangGraph agent with deterministic tools.

docs/plan.md, Phase 7 acceptance: "≥8/10 golden questions answered with correct
citations · agent refuses when evidence is absent" -- ≥11 of 13 since the
Phase 10 additions.

These tests run the AgentQueryService against the same indexed_corpus fixture
used by the Phase 4 deterministic tests. The agent wraps the deterministic path,
so the golden questions that pass in Phase 4 (≥9/13) must still pass here.

The acceptance bar is **≥11/13** — the two above the Phase 4 baseline are
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
    """The aggregate bar: ≥11/13 golden questions answered correctly."""
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


async def test_the_agent_refuses_a_question_no_column_can_answer(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """The reported failure, end to end.

    "issuer" routes this to `instrument_lookup`, which answers from rows and
    never reaches the refusal path retrieval owns. It returned every instrument
    at confidence 0.95 with no citations -- fluent, uncited, and about
    something else entirely.
    """
    answer = await service(db_session).answer("What is the CEO of the issuer paid?")

    assert answer.intent is QueryIntent.INSTRUMENT_LOOKUP
    assert answer.refused
    assert answer.confidence == 0.0
    assert answer.citations == []
    # The refusal names the words it could not place, so the question can be
    # rephrased or abandoned on the evidence.
    assert "'ceo'" in answer.answer
    # And no instrument leaked into the text on the way out.
    assert "Sdn Bhd" not in answer.answer


async def test_an_unanswerable_question_runs_no_tools(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """Refused before the rules engine and the portfolio SQL, not after.

    A breach check that evaluates every covenant and *then* discards the result
    is both slower and one edit away from reporting it.
    """
    answer = await service(db_session).answer("Which holdings breach their ESG policy limits?")

    assert answer.refused
    assert "evaluate_covenant_rule" not in answer.tools_used
    assert "run_read_only_sql" not in answer.tools_used


async def test_the_verify_node_refuses_even_if_synthesis_does_not(
    db_session: AsyncSession,
) -> None:
    """The last gate, on its own.

    `_synthesize` is a per-intent match statement and is where a new intent
    gets added; verify is where an answer stops being the graph's problem. This
    drives verify directly with a state that synthesis has already turned into
    a confident answer, which is what a forgotten branch would produce.
    """
    from app.agent.graph import AgentState, _verify

    state = AgentState(
        question="What is the CEO of the issuer paid?",
        intent=QueryIntent.INSTRUMENT_LOOKUP,
        answer="3 instrument(s) matched.",
        confidence=0.95,
        unsupported_terms=["ceo", "paid"],
    )

    verified = await _verify(state)

    assert verified.refused
    assert verified.confidence == 0.0
    assert "'ceo'" in verified.answer


# -- answering about what was asked about ------------------------------------
#
# The tools fetch everything and the formatters print what they are handed, so
# a question about one instrument came back with the whole universe at
# confidence 0.95 (finding 14). Wrong in a quieter way than finding 6 -- the
# right row is in there -- but a 200-bond portfolio makes it a page of noise
# around the fact somebody asked for.


async def test_a_question_about_one_instrument_is_answered_about_one(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    answer = await service(db_session).answer("Who is the issuer of the Green Ijarah Sukuk?")

    assert "Green Ijarah Sukuk" in answer.answer
    assert "1 instrument(s)" in answer.answer
    # The other two instruments are not part of this answer.
    assert "Wakalah" not in answer.answer
    assert "Retail REIT" not in answer.answer


async def test_a_breach_check_about_one_instrument_evaluates_one(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """Narrowing reaches the rules engine, not just the printed list.

    Evaluating every covenant in the book and printing a subset would leave the
    headline count describing something the question never asked about.
    """
    answer = await service(db_session).answer(
        "Does the Green Ijarah Sukuk breach its rating trigger?"
    )

    assert "across 1 instrument(s)" in answer.answer
    assert "Retail REIT" not in answer.answer


async def test_a_question_about_one_fund_does_not_total_another(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """The worst version of this defect: a wrong number, not just noise.

    The answer carries a total, and an unnarrowed one sums holdings from every
    portfolio into the figure for the fund that was named.
    """
    answer = await service(db_session).answer(
        "What is the total exposure of the Green Fixed Income Fund portfolio?"
    )

    assert "Green Fixed Income Fund" in answer.answer
    assert "Income Growth Fund" not in answer.answer


async def test_a_question_that_names_nothing_still_gets_everything(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """The regression guard: narrowing must not narrow to nothing.

    "Which holdings breach their rating trigger?" names no instrument on
    purpose -- it is a question about the whole book, and the flagship one.
    """
    answer = await service(db_session).answer(
        "Which holdings would breach their rating trigger at the current rating?"
    )

    assert not answer.refused
    assert "across 3 instrument(s)" in answer.answer


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


class TestCovenantLookupIsNarrowedToTheDocument:
    """Finding 15: a covenant question about one document, answered from all.

    A user uploaded a 201-page base prospectus, asked for its cross-default
    threshold, and got `RM30 million` at confidence 0.85 with five citations --
    every one of them from a synthetic fixture, none from their document. The
    threshold named appears nowhere in the document the question was about.

    The real corpus makes this the *normal* case rather than an edge one: a
    document can be ingested and searchable while its covenants have never been
    extracted, because extraction costs more than the per-document cap
    (finding 4). The only rows available to answer with belong to something
    else.
    """

    @staticmethod
    async def _ingest(
        db_session: AsyncSession, object_store: object, filename: str, *, extract: bool
    ) -> uuid.UUID:
        """Ingest and index a distinctively-named document, optionally extracting it.

        Without extraction this is exactly the state every real prospectus in
        the corpus is in: searchable, and holding no covenants.
        """
        from app.extract.service import RuleExtractionService
        from app.ingest.service import IngestionService
        from app.retrieval.indexing import IndexingService
        from tests.fixtures.synthetic_pdf import build_prose_document

        body = (
            "The Issuer shall not create or permit to subsist any security interest "
            "over its assets. An event of default shall occur if any indebtedness of "
            "the Issuer exceeding RM75,000,000 becomes due and payable prior to its "
            "stated maturity."
        )
        ingestion = IngestionService(db_session, object_store)  # type: ignore[arg-type]
        outcome = await ingestion.upload(
            filename=filename, data=build_prose_document(body, heading="BASE PROSPECTUS")
        )
        await ingestion.process(outcome.document.id)
        await IndexingService(db_session).index_document(outcome.document.id)
        if extract:
            await RuleExtractionService(db_session).extract_document(outcome.document.id)
        return outcome.document.id

    async def test_a_document_with_no_covenants_is_not_answered_from_another(
        self,
        db_session: AsyncSession,
        object_store: object,
        indexed_corpus: list[uuid.UUID],
        seeded_universe: None,
    ) -> None:
        """The reported failure, end to end."""
        await self._ingest(db_session, object_store, "dubai-base-prospectus.pdf", extract=False)

        answer = await service(db_session).answer(
            "What is the cross-default threshold in the Dubai prospectus?"
        )

        assert answer.refused, answer.answer
        assert answer.confidence == 0.0
        assert answer.citations == []
        # It says which document, and why there is nothing -- "no supporting
        # evidence" would read as "your document says nothing about it".
        assert "dubai-base-prospectus.pdf" in answer.answer
        # And above all: no other document's threshold is offered as an answer.
        assert "RM30" not in answer.answer
        assert "RM50" not in answer.answer

    async def test_a_named_document_narrows_to_its_own_covenants(
        self,
        db_session: AsyncSession,
        object_store: object,
        indexed_corpus: list[uuid.UUID],
        seeded_universe: None,
    ) -> None:
        """Naming an extracted document answers from that document alone."""
        document_id = await self._ingest(
            db_session, object_store, "kuching-port-prospectus.pdf", extract=True
        )

        answer = await service(db_session).answer(
            "What does the Kuching prospectus say about cross-default?"
        )

        assert not answer.refused, answer.answer
        assert answer.citations
        documents = {citation.document_id for citation in answer.citations}
        assert documents == {str(document_id)}, "citations came from more than the named document"
        # The corpus holds RM30m and RM50m cross-default thresholds in other
        # documents; this document's is RM75m and it is the only one offered.
        assert "RM75" in answer.answer
        assert "RM30" not in answer.answer

    def test_a_generic_document_word_names_no_document(self) -> None:
        """ "The prospectus" and "the trust deed" identify a kind, not a file.

        Deliberate, and the safe direction: failing to narrow returns the
        corpus, which is noisy; narrowing on a shared word would attach one
        document's covenants to another, which is finding 15 again.
        """
        from app.query.intent import mentioned_documents

        corpus = ["trust-deed.pdf", "prospectus.pdf", "2021-trust-certificate-prospectus.pdf"]

        assert mentioned_documents("what does the trust deed say?", corpus) == []
        assert mentioned_documents("what does the prospectus say?", corpus) == []

    async def test_a_question_that_names_no_document_still_sees_the_corpus(
        self,
        db_session: AsyncSession,
        indexed_corpus: list[uuid.UUID],
        seeded_universe: None,
    ) -> None:
        """The regression guard.

        "What cross-default thresholds do we have?" is a corpus-wide question
        and must stay one; narrowing everything to nothing is the easy way to
        break this fix.
        """
        answer = await service(db_session).answer("What are the cross-default thresholds?")

        assert not answer.refused
        documents = {citation.document_id for citation in answer.citations}
        assert len(documents) > 1, "a corpus-wide question was narrowed to one document"


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


class TestTheAgentMakesNoModelCalls:
    """Pins the claim the docstrings now make.

    Three docstrings and CLAUDE.md's routing table used to say the graph "adds
    LLM synthesis". It never has: `_synthesize` formats tool results with
    Python. Documentation drifted from the code because nothing checked, so
    this checks.

    If synthesis is later handed to a model, this test should be deleted in the
    same commit that does it -- and the docs updated in that commit too.
    """

    def test_the_agent_package_imports_nothing_from_the_llm_layer(self) -> None:
        import ast
        import pathlib

        agent_dir = pathlib.Path(__file__).parent.parent / "app" / "agent"
        offenders: list[str] = []

        for path in agent_dir.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("app.llm"):
                    offenders.append(f"{path.name}: from {node.module}")
                elif isinstance(node, ast.Import):
                    offenders.extend(
                        f"{path.name}: import {alias.name}"
                        for alias in node.names
                        if alias.name.startswith("app.llm")
                    )

        assert not offenders, (
            "app/agent/ reaches the LLM layer, so the docstrings claiming it is "
            f"deterministic are now false: {offenders}"
        )

    async def test_answering_records_no_spend(
        self, db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
    ) -> None:
        """The ledger is the other half of the claim: no calls, no cost."""
        from sqlalchemy import func, select

        from app.db.models.ops import LLMCall

        before = (await db_session.execute(select(func.count()).select_from(LLMCall))).scalar_one()

        await service(db_session).answer("What is the cross-default threshold?")

        after = (await db_session.execute(select(func.count()).select_from(LLMCall))).scalar_one()
        assert after == before
