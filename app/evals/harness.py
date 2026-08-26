"""The eval harness: run every metric, return one report.

docs/plan.md Phase 8 names nine metrics. They live in four modules and are assembled
here so a run produces one artefact rather than nine command invocations:

| Metric (docs/plan.md 6)            | Where it is computed                    |
|-------------------------------|-----------------------------------------|
| Field-level F1                | `evals/extraction.py`                   |
| Enum exact match              | `evals/metrics.py` (no partial credit)  |
| Numeric tolerance             | `evals/metrics.py`, in `Decimal`        |
| Date tolerance                | `evals/metrics.py`, call schedules      |
| Citation precision / recall   | `evals/extraction.py`, via `verify_quote` |
| Answer faithfulness           | `evals/answers.py`                      |
| Refusal correctness           | `evals/answers.py`                      |
| Rules-vs-LLM agreement        | `evals/cost.py`, from the review queue  |
| Cost per document             | `evals/cost.py`, from `llm_calls`       |

**The run costs $0.** Both read paths are deterministic and no metric calls a
model, so `make eval` runs in CI under the same rule as `make test`
(CLAUDE.md 7). docs/plan.md's routing table reserves an LLM judge for faithfulness;
it is not built, and if it ever is it needs the `RUN_LIVE_LLM_TESTS=1` gate
before it can be part of this run.

**A metric with no data reports nothing, not zero.** A corpus with no LLM
extraction has no rules-vs-LLM agreement rate; a document that was never
ingested is reported missing rather than scored as a total miss. The failure
mode this avoids is the one that matters: a green number that means "nobody
looked".

**And a harness that found no corpus fails.** Reporting the gap is not enough on
its own: the labels join on `document_sha256`, so a fixture whose bytes change
stops joining rather than starting to disagree, and every extraction metric is
then computed over an empty set while the golden-question targets -- which do
not touch the labels -- carry on passing. `meets_targets` therefore reads
`corpus_complete`, and `opuscovintel eval` exits non-zero on it. This is not
hypothetical: `pymupdf` 1.28.2 changed the fixture bytes and a run scoring zero
documents logged `meets_targets: true`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.repositories.documents import DocumentRepository
from app.evals.answers import Answerer, AnswerEvaluator, PathScores
from app.evals.cost import AgreementReport, CostEvaluator, CostReport
from app.evals.extraction import DocumentScores, ExtractionEvaluator
from app.evals.golden import PHASE_4_TARGET, PHASE_7_TARGET
from app.evals.labels import LABELLED_SHAS, document_name
from app.evals.metrics import Score

logger = get_logger(__name__)

DETERMINISTIC_PATH = "deterministic"
AGENT_PATH = "agent"

# Stamped on every report. The golden set is synthetic (CLAUDE.md 7 forbids
# committing real prospectuses), so these numbers describe text written to be
# extractable. Saying so in the artefact is the only thing that stops a
# synthetic F1 being quoted later as a production one (docs/plan.md 9, question 1).
CORPUS_CAVEAT = (
    "Scored against synthetic fixtures only. No licensed prospectus is in this "
    "corpus, so these figures are a regression baseline, not a production "
    "accuracy estimate. Re-baseline when real documents arrive."
)


@dataclass
class EvalReport:
    generated_at: dt.datetime
    extraction_model: str
    documents_scored: list[DocumentScores] = field(default_factory=list)
    documents_missing: list[str] = field(default_factory=list)
    documents_unlabelled: list[str] = field(default_factory=list)
    answers: dict[str, PathScores] = field(default_factory=dict)
    cost: CostReport = field(default_factory=CostReport)
    agreement: AgreementReport = field(default_factory=AgreementReport)
    errors: list[str] = field(default_factory=list)

    @property
    def corpus_complete(self) -> bool:
        """Whether every labelled document was found and scored.

        Not a quality measure -- a precondition for there being one. The labels
        join on `document_sha256` (`evals/labels.py`), so a document whose bytes
        changed, or was never ingested, drops out of the join entirely: nothing
        is scored badly, nothing is scored at all.
        """
        return bool(self.documents_scored) and not self.documents_missing

    @property
    def meets_targets(self) -> bool:
        """Whether the run measured its corpus and met every acceptance target.

        Extraction F1 has no target here on purpose: docs/plan.md sets one for the
        golden questions and none for extraction, and inventing a threshold
        against a synthetic corpus would turn a measurement into a gate that
        says nothing about production.

        `corpus_complete` is a different thing from a target and is required
        anyway. The golden questions do not join on the labels, so they pass
        whether or not a single labelled document was scored -- which is exactly
        how a run over an empty corpus came to report success. A target no
        document was measured against has not been met.
        """
        return (
            self.corpus_complete
            and bool(self.answers)
            and all(scores.meets_target for scores in self.answers.values())
        )

    def field_totals(self, method: str) -> dict[str, Score]:
        """Per-field covenant scores summed across every document, for one method."""
        totals: dict[str, Score] = {}
        for document in self.documents_scored:
            scores = document.by_method.get(method)
            if scores is None:
                continue
            for name, score in scores.covenant_fields.items():
                totals[name] = totals.get(name, Score(name=name)) + score
        return totals

    def call_schedule_totals(self) -> dict[str, Score]:
        """Call-schedule scores across the corpus, not split by extractor."""
        totals: dict[str, Score] = {}
        for document in self.documents_scored:
            for name, score in document.call_fields.items():
                totals[name] = totals.get(name, Score(name=name)) + score
        return totals

    def call_schedule_methods(self) -> list[str]:
        seen: set[str] = set()
        for document in self.documents_scored:
            seen.update(document.call_methods)
        return sorted(seen)

    def methods(self) -> list[str]:
        seen: set[str] = set()
        for document in self.documents_scored:
            seen.update(document.by_method)
        return sorted(seen)

    def as_dict(self) -> dict[str, object]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "caveat": CORPUS_CAVEAT,
            "extraction_model": self.extraction_model,
            "meets_targets": self.meets_targets,
            "corpus": {
                "complete": self.corpus_complete,
                "scored": [document.name for document in self.documents_scored],
                "missing": self.documents_missing,
                "unlabelled_in_corpus": self.documents_unlabelled,
            },
            "extraction": {
                "documents": [document.as_dict() for document in self.documents_scored],
                "totals_by_method": {
                    method: {
                        name: score.as_dict()
                        for name, score in sorted(self.field_totals(method).items())
                    }
                    for method in self.methods()
                },
                "call_schedules": {
                    "extracted_by": self.call_schedule_methods(),
                    "totals": {
                        name: score.as_dict()
                        for name, score in sorted(self.call_schedule_totals().items())
                    },
                },
            },
            "answers": {path: scores.as_dict() for path, scores in sorted(self.answers.items())},
            "agreement": self.agreement.as_dict(),
            "cost": self.cost.as_dict(),
            "errors": self.errors,
        }


async def run_eval(
    session: AsyncSession,
    *,
    deterministic: Answerer | None = None,
    agent: Answerer | None = None,
    settings: Settings | None = None,
    cost_session: AsyncSession | None = None,
) -> EvalReport:
    """Score the corpus and the golden set through whichever paths are supplied.

    The answerers are injected rather than constructed here so the harness does
    not decide which database role the agent runs on -- that choice belongs to
    the caller, and CLAUDE.md 1.6 makes it a real one (the read path must be
    the read-only role, the query log must not be).

    `cost_session` is the same choice for the ledger, in the other direction.
    `llm_calls` is denied to the read-only role, so a caller that scores answers
    through it must hand over a session that may read what those answers cost.
    Defaults to `session`, which is right for a test or a script running
    entirely read-write.
    """
    resolved = settings or get_settings()
    report = EvalReport(
        generated_at=dt.datetime.now(dt.UTC),
        extraction_model=resolved.EXTRACTION_MODEL,
    )

    extraction = ExtractionEvaluator(session)
    for sha256 in LABELLED_SHAS:
        scores = await extraction.score_document(sha256)
        if scores is None:
            report.documents_missing.append(document_name(sha256))
            continue
        report.documents_scored.append(scores)

    # Named so a reader can tell "the extractor found nothing" from "nothing was
    # labelled for this document". Both look like a gap in the numbers and they
    # have completely different fixes.
    known = set(LABELLED_SHAS)
    for document in await DocumentRepository(session).list(limit=500):
        if document.sha256 not in known:
            report.documents_unlabelled.append(document.filename)

    evaluator = AnswerEvaluator(session)
    if deterministic is not None:
        report.answers[DETERMINISTIC_PATH] = await evaluator.score(
            deterministic, path=DETERMINISTIC_PATH, target=PHASE_4_TARGET
        )
    if agent is not None:
        report.answers[AGENT_PATH] = await evaluator.score(
            agent, path=AGENT_PATH, target=PHASE_7_TARGET
        )

    costs = CostEvaluator(cost_session or session, resolved)
    report.cost = await costs.report()
    report.agreement = await costs.agreement()

    logger.info(
        "eval.complete",
        extra={
            "documents_scored": len(report.documents_scored),
            "documents_missing": len(report.documents_missing),
            "corpus_complete": report.corpus_complete,
            "meets_targets": report.meets_targets,
            "total_cost_usd": str(report.cost.total_usd),
        },
    )
    return report
