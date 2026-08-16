"""The eval harness against a real corpus.

These run over `indexed_corpus`: three synthetic documents, ingested, indexed
and rule-extracted, in a transaction that is rolled back. That is the same
shape as a `make eval` run, so what passes here is what runs there.

The assertions name exact counts rather than "greater than zero". A harness
that reports something is not a harness that reports the right thing, and
`assert x >= 0` is true of every possible bug.
"""

from __future__ import annotations

import datetime as dt
import uuid
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession
from typer.testing import CliRunner

from app.cli import app as cli_app
from app.db.models.clauses import Covenant
from app.domain.enums import CovenantType
from app.evals import harness as harness_module
from app.evals import labels as label_module
from app.evals.answers import PathScores
from app.evals.extraction import ExtractionEvaluator
from app.evals.harness import EvalReport, run_eval
from app.evals.report import render_markdown, write_report
from app.query.service import DeterministicQueryService


async def test_every_labelled_document_is_scored(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    report = await run_eval(db_session)
    assert [document.name for document in report.documents_scored] == [
        "prospectus",
        "trust-deed",
        "rating-report",
    ]
    assert report.documents_missing == []


async def test_the_rule_extractor_finds_every_labelled_covenant(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """The baseline this harness exists to defend.

    Exact counts: the corpus carries 15 labelled covenants and the deterministic
    extractor is expected to find all of them with no false positives. A
    regression in a regex shows up here as a number, which is the point.
    """
    report = await run_eval(db_session)
    totals = report.field_totals("rule")
    covenant_type = totals["covenant_type"]

    assert covenant_type.true_positives == len(label_module.COVENANT_LABELS) == 15
    assert covenant_type.false_positives == 0
    assert covenant_type.false_negatives == 0
    assert covenant_type.f1 == pytest.approx(1.0)


async def test_thresholds_and_operators_are_scored_in_decimal(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    totals = (await run_eval(db_session)).field_totals("rule")
    # Three monetary thresholds (RM30m, RM50m, RM500m) and six ratios.
    assert totals["threshold_amount"].true_positives == 3
    assert totals["threshold_ratio"].true_positives == 6
    assert totals["operator"].false_positives == 0


async def test_call_schedule_dates_are_scored_once_not_per_method(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    report = await run_eval(db_session)
    totals = report.call_schedule_totals()
    assert totals["call_date"].true_positives == 3
    assert totals["call_date"].false_positives == 0
    # `make seed` writes demo call dates as `human`; those are not predictions
    # and must not be scored as extractor output.
    assert report.call_schedule_methods() == ["rule"]


async def test_a_wrong_threshold_lowers_the_score(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """The harness must be able to report a failure, not only a pass.

    Written after a sibling test passed against a deliberately corrupted label
    and had to be rewritten: a metric is only worth its green when the red has
    been seen.
    """
    before = (await run_eval(db_session)).field_totals("rule")["threshold_amount"]
    assert before.false_positives == 0

    # RM30m, misread as RM31m: 3.3 per cent off, well outside the tolerance and
    # exactly the kind of misread the metric has to catch.
    changed = await db_session.execute(
        update(Covenant)
        .where(Covenant.threshold_amount == Decimal("30000000"))
        .values(threshold_amount=Decimal("31000000"))
        .returning(Covenant.id)
    )
    assert len(changed.all()) == 1
    await db_session.flush()

    after = (await run_eval(db_session)).field_totals("rule")["threshold_amount"]
    assert after.true_positives == before.true_positives - 1
    assert after.false_positives == 1
    assert after.false_negatives == 1


async def test_a_missing_covenant_is_a_false_negative(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    before = (await run_eval(db_session)).field_totals("rule")["covenant_type"]

    removed = await db_session.execute(
        delete(Covenant)
        .where(Covenant.covenant_type == CovenantType.CROSS_DEFAULT)
        .returning(Covenant.id)
    )
    assert len(removed.all()) == 2
    await db_session.flush()

    after = (await run_eval(db_session)).field_totals("rule")["covenant_type"]
    # Two documents state a cross-default covenant.
    assert after.false_negatives == before.false_negatives + 2
    assert after.recall is not None and after.recall < 1.0


async def test_citations_are_reverified_against_the_chunk(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    scores = await ExtractionEvaluator(db_session).score_document(label_module.PROSPECTUS_SHA)
    assert scores is not None
    citations = scores.by_method["rule"].citations
    assert citations.predicted > 0
    assert citations.quote_reverified == citations.predicted
    assert citations.reverification_rate == pytest.approx(1.0)


async def test_the_golden_questions_meet_the_phase_4_target(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    report = await run_eval(db_session, deterministic=DeterministicQueryService(db_session))
    scores = report.answers["deterministic"]
    assert scores.meets_target, [r for r in scores.results if not r.passed]
    assert scores.faithfulness == pytest.approx(1.0)
    # CLAUDE.md 1.5: the unanswerable question must be refused, and refusing
    # anything else is equally wrong.
    assert scores.refusal.false_positives == 0
    assert scores.refusal.false_negatives == 0
    assert scores.unsupported_answers == 0


async def test_agreement_is_undefined_without_an_llm_extraction(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """A $0 corpus has no agreement rate -- not a perfect one."""
    report = await run_eval(db_session)
    assert report.agreement.rule_covenants > 0
    assert report.agreement.llm_covenants == 0
    assert not report.agreement.measurable
    assert report.agreement.agreement_rate is None


async def test_cost_reports_nothing_rather_than_zero_when_nothing_was_spent(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    report = await run_eval(db_session)
    assert not report.cost.has_spend
    assert report.cost.cost_per_document is None
    assert "No provider call" in render_markdown(report)


async def test_the_ledger_is_read_through_the_session_it_was_given(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """`llm_calls` is denied to the read-only role, so it needs its own session.

    `make eval` scores answers through `opuscovintel_ro` -- the role the agent
    uses -- and that role cannot read the cost ledger: it is one of the six
    operational tables Phase 10 revoked. Reading costs through the scoring
    session made the whole command exit with `permission denied`, which no test
    caught because the suite runs everything read-write.

    Proved here without a second role: the spend row lives in an uncommitted
    transaction, so a report that reads it can only have read it through the
    session that holds it.
    """
    from app.db.models.ops import LLMCall
    from app.domain.enums import LLMStage

    db_session.add(
        LLMCall(
            stage=LLMStage.EXTRACT,
            provider="mock",
            model_id="mock-model",
            estimated_cost_usd=Decimal("1.25"),
        )
    )
    await db_session.flush()

    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool

    from tests.conftest import _test_database_url

    # A second connection to the same database. It cannot see this test's
    # uncommitted row, which is what makes the assertion below discriminating.
    engine = create_async_engine(
        _test_database_url().render_as_string(hide_password=False), poolclass=NullPool
    )
    try:
        async with AsyncSession(engine, expire_on_commit=False) as blind:
            through_cost_session = await run_eval(blind, cost_session=db_session)
            through_scoring_session = await run_eval(blind)
    finally:
        await engine.dispose()

    assert through_cost_session.cost.has_spend
    assert through_cost_session.cost.total_usd == Decimal("1.25")
    # And the control: the session that scores answers sees no spend at all, so
    # the report above can only have come from the one it was handed.
    assert not through_scoring_session.cost.has_spend


async def test_report_is_written_as_json_and_markdown(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID], tmp_path: Path
) -> None:
    report = await run_eval(db_session, deterministic=DeterministicQueryService(db_session))
    json_path, markdown_path = write_report(report, directory=tmp_path / "results")

    assert json_path.exists() and markdown_path.exists()
    assert (tmp_path / "results" / "latest.json").read_text() == json_path.read_text()

    markdown = markdown_path.read_text()
    # The caveat is not decoration: a synthetic F1 quoted as a production one is
    # the most likely way this report gets misread.
    assert "synthetic fixtures only" in markdown
    assert "## Golden questions" in markdown
    assert "## Extraction" in markdown


async def test_an_uningested_document_is_reported_missing_not_scored_zero(
    db_session: AsyncSession, seeded_universe: None
) -> None:
    """An empty corpus must not look like an extractor that found nothing."""
    report = await run_eval(db_session)
    assert report.documents_scored == []
    assert sorted(report.documents_missing) == ["prospectus", "rating-report", "trust-deed"]
    assert "never ingested" in render_markdown(report)


class TestAHarnessThatScoredNothingFails:
    """The failure this class exists for happened on 2026-08-16.

    `pymupdf` 1.28.0 -> 1.28.2 changed the bytes of the generated fixtures, the
    hashes in `evals/labels.py` stopped matching what was ingested, and the join
    on `document_sha256` returned nothing. The run logged
    `documents_scored: 0`, `documents_missing: 3`, `meets_targets: true` and the
    command exited zero -- because the golden-question targets, which are all
    `meets_targets` used to read, do not join on the labels and passed as usual.

    An eval that cannot find its corpus has measured nothing, so it has met
    nothing (CLAUDE.md 7: `assert x >= 0` is true of every possible bug).
    """

    async def test_labels_that_join_to_nothing_do_not_meet_targets(
        self,
        db_session: AsyncSession,
        indexed_corpus: list[uuid.UUID],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A hash nothing matches -- exactly what a changed fixture produces.

        The corpus is fully ingested here and the read path is scored, so the
        golden questions still hit their target. Nothing but the broken join is
        wrong, which is what makes this the reproduction: before the fix, this
        report said `meets_targets: True`.
        """
        ghost = "0" * 64
        monkeypatch.setattr(harness_module, "LABELLED_SHAS", (ghost,))

        report = await run_eval(db_session, deterministic=DeterministicQueryService(db_session))

        assert report.documents_scored == []
        assert report.documents_missing == [ghost[:12]]
        # The half that keeps passing, and hid the fault.
        assert all(scores.meets_target for scores in report.answers.values())
        # The half that must not.
        assert not report.corpus_complete
        assert not report.meets_targets

    async def test_a_partly_ingested_corpus_is_also_a_failure(
        self,
        db_session: AsyncSession,
        indexed_corpus: list[uuid.UUID],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two of three labels joining is a harness measuring two thirds of itself."""
        monkeypatch.setattr(
            harness_module,
            "LABELLED_SHAS",
            (label_module.PROSPECTUS_SHA, label_module.TRUST_DEED_SHA, "0" * 64),
        )

        report = await run_eval(db_session, deterministic=DeterministicQueryService(db_session))

        assert [document.name for document in report.documents_scored] == [
            "prospectus",
            "trust-deed",
        ]
        assert not report.corpus_complete
        assert not report.meets_targets

    def test_the_command_exits_non_zero_when_nothing_was_scored(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The other half of the fault: `opuscovintel eval` exited zero.

        The report is built rather than measured -- the command opens its own
        engines and event loop, which a session bound to this test's loop cannot
        serve. What is under test is the command's decision, and the report it
        is handed is the one the 2026-08-16 run produced: no document scored,
        three missing, and a read path comfortably over its target.
        """
        report = EvalReport(
            generated_at=dt.datetime.now(dt.UTC),
            extraction_model="claude-opus-5",
            documents_missing=["prospectus", "trust-deed", "rating-report"],
            answers={"deterministic": PathScores(path="deterministic", target=0)},
        )
        assert report.answers["deterministic"].meets_target, "the read path must look green"

        async def _stub_run_eval(*args: object, **kwargs: object) -> EvalReport:
            return report

        class _Session:
            async def __aenter__(self) -> _Session:
                return self

            async def __aexit__(self, *exc: object) -> None: ...

        async def _no_dispose() -> None: ...

        monkeypatch.setattr("app.evals.harness.run_eval", _stub_run_eval)
        monkeypatch.setattr("app.db.session.get_sessionmaker", lambda: lambda: _Session())
        monkeypatch.setattr("app.db.session.get_readonly_sessionmaker", lambda: lambda: _Session())
        monkeypatch.setattr("app.cli.dispose_engines", _no_dispose)

        result = CliRunner().invoke(
            cli_app, ["eval", "--skip-agent", "--quiet", "--output-dir", str(tmp_path)]
        )

        assert result.exit_code == 1, result.output
        output = result.output + (result.stderr or "")
        assert "Labelled but not scored" in output
        assert "prospectus" in output
        # The artefact is still written: a run that failed this way is a run
        # someone has to diagnose, and the report names what was missing.
        assert (tmp_path / "latest.md").exists()
