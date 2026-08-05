"""The `extract` CLI command — the one command in the tool that spends money.

The tests here are mostly about refusing to spend. `extract-rules` defaults to
every document because it is free; the same default on a billable command is
the "$10 keystroke" CLAUDE.md 9 warns about, so the guards around dispatch
matter more than the happy path.

Nothing here reaches a provider: the dispatching tests are driven through the
pipeline with `MockLLMProvider` elsewhere, and these exercise the command's
decisions before dispatch.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner

from app.cli import app
from app.extract.dry_run import estimate_document

runner = CliRunner()

pytestmark = pytest.mark.usefixtures("storage_root")


@pytest.fixture
def no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the command to see an unconfigured provider.

    Deleting the environment variable is not enough: settings also read `.env`,
    which on a developer machine holds a real key. A test whose meaning depends
    on whether the operator has paid for an API is not a test, and one that
    could dispatch because the key happened to be present is a liability.
    """
    from app.core.config import Settings

    keyless = Settings(ENVIRONMENT="test", ANTHROPIC_API_KEY=None)
    monkeypatch.setattr("app.cli.get_settings", lambda: keyless)


class TestRefusesToGuess:
    def test_no_target_is_an_error_not_a_whole_corpus_run(self, no_api_key: None) -> None:
        """The dangerous default is the absence of one."""
        result = runner.invoke(app, ["extract"])

        assert result.exit_code == 2
        assert "Refusing to guess" in result.output

    def test_all_is_how_you_mean_the_whole_corpus(self, no_api_key: None) -> None:
        result = runner.invoke(app, ["extract", "--all"])

        # Passes the target check, then stops on the missing key.
        assert "Refusing to guess" not in result.output
        assert result.exit_code == 2


class TestRefusesWithoutAKey:
    def test_a_missing_key_is_caught_before_anything_is_attempted(self, no_api_key: None) -> None:
        result = runner.invoke(app, ["extract", str(uuid.uuid4())])

        assert result.exit_code == 2
        assert "ANTHROPIC_API_KEY is not set" in result.output

    def test_dry_run_needs_no_key(self, no_api_key: None) -> None:
        """Pricing the work must not require the ability to pay for it."""
        result = runner.invoke(app, ["extract", str(uuid.uuid4()), "--dry-run"])

        assert "ANTHROPIC_API_KEY is not set" not in result.output


class TestDryRunEstimator:
    """PLAN.md 2's `--dry-run` estimator. Free: regex over rows already stored."""

    async def test_it_prices_a_real_document_without_dispatching(
        self, db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
    ) -> None:
        estimate = await estimate_document(db_session, indexed_corpus[0])

        assert estimate.candidates > 0
        assert estimate.prompt_tokens > 0
        assert estimate.ceiling_usd > 0
        assert str(estimate.document_id) in estimate.describe()

    async def test_a_document_with_no_candidates_costs_nothing(
        self, db_session: AsyncSession, seeded_universe: None
    ) -> None:
        """No candidate spans means no calls, which must read as $0 not as an error."""
        estimate = await estimate_document(db_session, uuid.uuid4())

        assert estimate.candidates == 0
        assert estimate.ceiling_usd == 0
        assert "$0" in estimate.describe()

    async def test_the_estimate_names_the_cap_it_is_measured_against(
        self, db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
    ) -> None:
        """An operator deciding whether to proceed needs both numbers."""
        estimate = await estimate_document(db_session, indexed_corpus[0])
        described = estimate.describe()

        assert str(estimate.per_document_cap) in described
        assert "worst case" in described

    async def test_the_cached_prefix_is_reported(
        self, db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
    ) -> None:
        """The gap between ceiling and reality is mostly prompt caching, so say so."""
        estimate = await estimate_document(db_session, indexed_corpus[0])

        if estimate.candidates > 1:
            assert estimate.cached_prefix_tokens > 0
            assert "0.1x" in estimate.describe()


class TestInstrumentLinking:
    async def test_llm_covenants_are_linked_to_an_instrument(
        self, db_session: AsyncSession, indexed_corpus: list[uuid.UUID], seeded_universe: None
    ) -> None:
        """An unlinked covenant is invisible to every portfolio query.

        The pipeline used to leave `instrument_id` NULL because only the rule
        extractor resolved it, so the LLM path produced covenants that the
        headline portfolio queries could not see.
        """
        from sqlalchemy import select

        from app.db.models.clauses import Covenant
        from app.domain.enums import ExtractionMethod
        from app.extract.pipeline import ExtractionPipeline
        from app.llm.mock import MockLLMProvider
        from app.llm.router import LLMRouter

        pipeline = ExtractionPipeline(
            db_session, router=LLMRouter(db_session, provider=MockLLMProvider())
        )
        await pipeline.extract(indexed_corpus[0])

        result = await db_session.execute(
            select(Covenant).where(Covenant.method == ExtractionMethod.LLM)
        )
        covenants = list(result.scalars().all())
        if not covenants:
            pytest.skip("mock output produced no LLM covenants for this fixture")

        assert all(c.instrument_id is not None for c in covenants), (
            "LLM covenants must carry an instrument or portfolio queries cannot find them"
        )


class TestAskCommand:
    """The agent's CLI entry point (`ask`).

    Phase 7 had no caller outside tests until this command existed, so the
    graph had never run against real engines -- which is where the two-session
    split (CLAUDE.md 1.6) either works or silently does not.
    """

    def test_it_uses_the_two_role_factory(self) -> None:
        """Constructing the service by hand would put the whole graph on one role.

        Checked against the parsed call graph rather than the source text: a
        comment explaining *why* not to call `AgentQueryService` directly would
        fail a substring check, which is how the first version of this test
        managed to fail on correct code.
        """
        import ast
        import inspect
        import textwrap

        from app import cli

        tree = ast.parse(textwrap.dedent(inspect.getsource(cli.ask)))
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }

        assert "open_agent_query_service" in called
        assert "AgentQueryService" not in called

    def test_it_reports_intent_confidence_and_answer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from collections.abc import AsyncIterator
        from contextlib import asynccontextmanager

        from app.agent.service import AgentAnswer
        from app.domain.enums import QueryIntent

        canned = AgentAnswer(
            question="q",
            intent=QueryIntent.COVENANT_LOOKUP,
            answer="cross_default · threshold RM30 million",
            confidence=0.85,
            tools_used=["get_covenants"],
        )

        class _Service:
            async def answer(self, question: str, **kwargs: object) -> AgentAnswer:
                return canned

        @asynccontextmanager
        async def _fake() -> AsyncIterator[_Service]:
            yield _Service()

        monkeypatch.setattr("app.agent.service.open_agent_query_service", _fake)

        result = runner.invoke(app, ["ask", "What is the cross-default threshold?", "--show-tools"])

        assert result.exit_code == 0
        assert "covenant_lookup" in result.output
        assert "0.85" in result.output
        assert "RM30 million" in result.output
        assert "get_covenants" in result.output


class TestOcrCommand:
    """`ocr` — the second billable command. Same refusals as `extract`."""

    @pytest.fixture
    def no_openai_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.core.config import Settings

        keyless = Settings(ENVIRONMENT="test", OPENAI_API_KEY=None)
        monkeypatch.setattr("app.cli.get_settings", lambda: keyless)

    def test_no_target_is_an_error(self, no_openai_key: None) -> None:
        result = runner.invoke(app, ["ocr"])

        assert result.exit_code == 2
        assert "Refusing to guess" in result.output

    def test_a_missing_key_is_caught_before_dispatch(self, no_openai_key: None) -> None:
        result = runner.invoke(app, ["ocr", str(uuid.uuid4())])

        assert result.exit_code == 2
        assert "OPENAI_API_KEY is not set" in result.output

    def test_dry_run_needs_no_key(self, no_openai_key: None) -> None:
        result = runner.invoke(app, ["ocr", str(uuid.uuid4()), "--dry-run"])

        assert "OPENAI_API_KEY is not set" not in result.output

    def test_it_runs_ocr_and_chunking_together(self) -> None:
        """Splitting these is how the transcription ended up stored and unread.

        Asserted against the parsed call graph so the coupling is structural
        rather than a matter of whoever next edits the command remembering.
        """
        import ast
        import inspect
        import textwrap

        from app import cli

        tree = ast.parse(textwrap.dedent(inspect.getsource(cli.ocr)))
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }

        assert "process_document" in called
        assert "rechunk_document" in called
