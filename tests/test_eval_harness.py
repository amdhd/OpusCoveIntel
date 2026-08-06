"""The eval harness against a real corpus.

These run over `indexed_corpus`: three synthetic documents, ingested, indexed
and rule-extracted, in a transaction that is rolled back. That is the same
shape as a `make eval` run, so what passes here is what runs there.

The assertions name exact counts rather than "greater than zero". A harness
that reports something is not a harness that reports the right thing, and
`assert x >= 0` is true of every possible bug.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from pathlib import Path

import pytest
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.clauses import Covenant
from app.domain.enums import CovenantType
from app.evals import labels as label_module
from app.evals.extraction import ExtractionEvaluator
from app.evals.harness import run_eval
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
