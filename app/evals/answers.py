"""Scoring answers: the golden questions, faithfulness and refusal.

Runs `app.evals.golden.GOLDEN_QUESTIONS` down both read paths -- the Phase 4
deterministic service and the Phase 7 LangGraph agent -- and scores them the
same way. Keeping the two comparable is the point: PLAN.md sets 6/10 for the
deterministic path and 8/10 with the agent on top, and the only way to know
whether the agent earned those two questions is to ask it the same ten.

Neither path calls a model, so this is $0 and runs in CI (CLAUDE.md 7).

**Faithfulness is checked against the corpus, not against the answer.** Every
citation an answer carries is re-verified against the chunk it names, using
`verify_quote` -- the same check the extraction pipeline gates on. An answer
that quotes text which is not in the chunk it cites is unfaithful even if the
claim happens to be true, because nothing in the system can defend it.

**Refusal is scored as a metric, not as an exception.** CLAUDE.md 1.5 makes
refusal a correct outcome, so a harness that only counts answers rewards a
system for guessing. Both directions are counted: refusing what it should have
answered, and answering what it should have refused.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.repositories.documents import DocumentChunkRepository
from app.domain.enums import QueryIntent
from app.domain.rules import Citation
from app.evals.golden import GOLDEN_QUESTIONS, GoldenQuestion
from app.evals.metrics import Score, ratio
from app.extract.citations import verify_quote

logger = get_logger(__name__)

# Intents answered from structured rows and rules rather than from retrieved
# spans (CLAUDE.md 1.1: portfolio aggregation is SQL, breach checks are the
# rules engine). An answer of this kind legitimately carries no span citation,
# so scoring it as "unevidenced" would penalise the design the invariant asks
# for. Its grounding is checked differently: it must name the tool that
# computed it.
_STRUCTURED_INTENTS: frozenset[QueryIntent] = frozenset(
    {
        QueryIntent.PORTFOLIO_QUERY,
        QueryIntent.INSTRUMENT_LOOKUP,
        QueryIntent.COVENANT_BREACH_CHECK,
    }
)


class AnswerLike(Protocol):
    """The contract both read paths satisfy, so both can be scored by one function.

    Declared as read-only properties rather than attributes: both concrete
    answers are frozen dataclasses, and a protocol asking for a *settable*
    attribute is not satisfied by one that cannot be set. The scorer only ever
    reads, so read-only is also the honest declaration.
    """

    @property
    def intent(self) -> QueryIntent: ...
    @property
    def citations(self) -> list[Citation]: ...
    @property
    def confidence(self) -> float: ...
    @property
    def refused(self) -> bool: ...
    @property
    def tools_used(self) -> list[str]: ...
    @property
    def text(self) -> str: ...


class Answerer(Protocol):
    async def answer(self, question: str) -> AnswerLike: ...


@dataclass
class QuestionResult:
    """One golden question, one path."""

    id: str
    question: str
    passed: bool
    intent: str
    expected_intent: str
    refused: bool
    expected_refusal: bool
    citations: int
    citations_verified: int
    confidence: float
    missing_terms: list[str] = field(default_factory=list)
    faithful: bool = True
    notes: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "question": self.question,
            "passed": self.passed,
            "intent": self.intent,
            "expected_intent": self.expected_intent,
            "refused": self.refused,
            "expected_refusal": self.expected_refusal,
            "citations": self.citations,
            "citations_verified": self.citations_verified,
            "confidence": self.confidence,
            "missing_terms": self.missing_terms,
            "faithful": self.faithful,
            "notes": self.notes,
        }


@dataclass
class PathScores:
    """Everything measured about one read path over the golden set."""

    path: str
    target: int
    results: list[QuestionResult] = field(default_factory=list)
    refusal: Score = field(default_factory=lambda: Score(name="refusal"))
    citations_checked: int = 0
    citations_verified: int = 0
    unsupported_answers: int = 0
    ungrounded_structured_answers: int = 0

    @property
    def passed(self) -> int:
        return sum(1 for result in self.results if result.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def meets_target(self) -> bool:
        return self.passed >= self.target

    @property
    def citation_verification_rate(self) -> float | None:
        return ratio(self.citations_verified, self.citations_checked)

    @property
    def faithfulness(self) -> float | None:
        """Share of answers with nothing in them the corpus cannot support."""
        return ratio(sum(1 for result in self.results if result.faithful), self.total)

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "passed": self.passed,
            "total": self.total,
            "target": self.target,
            "meets_target": self.meets_target,
            "faithfulness": self.faithfulness,
            "citations_checked": self.citations_checked,
            "citations_verified": self.citations_verified,
            "citation_verification_rate": self.citation_verification_rate,
            "unsupported_answers": self.unsupported_answers,
            "ungrounded_structured_answers": self.ungrounded_structured_answers,
            "refusal": self.refusal.as_dict(),
            "questions": [result.as_dict() for result in self.results],
        }


class AnswerEvaluator:
    """Scores one read path over the golden question set."""

    def __init__(self, session: AsyncSession) -> None:
        # The read session, used only to fetch chunk text for citation
        # re-verification. Read-only by preference (CLAUDE.md 1.6).
        self._chunks = DocumentChunkRepository(session)

    async def score(self, answerer: Answerer, *, path: str, target: int) -> PathScores:
        scores = PathScores(path=path, target=target)
        for case in GOLDEN_QUESTIONS:
            answer = await answerer.answer(case.question)
            result = await self._score_one(case, answer, scores)
            scores.results.append(result)

        refused_correctly = sum(
            1 for result in scores.results if result.refused and result.expected_refusal
        )
        refused_wrongly = sum(
            1 for result in scores.results if result.refused and not result.expected_refusal
        )
        should_have_refused = sum(
            1 for result in scores.results if not result.refused and result.expected_refusal
        )
        scores.refusal = Score(
            name="refusal",
            true_positives=refused_correctly,
            false_positives=refused_wrongly,
            false_negatives=should_have_refused,
        )
        logger.info(
            "eval.answers",
            extra={
                "path": path,
                "passed": scores.passed,
                "total": scores.total,
                "faithfulness": scores.faithfulness,
            },
        )
        return scores

    async def _score_one(
        self, case: GoldenQuestion, answer: AnswerLike, scores: PathScores
    ) -> QuestionResult:
        text = answer.text or ""
        missing = [needle for needle in case.must_contain if needle.lower() not in text.lower()]
        verified = 0
        for citation in answer.citations:
            scores.citations_checked += 1
            if await self._citation_holds(citation):
                verified += 1
        scores.citations_verified += verified

        # `_verify` in the agent graph refuses an evidence-bearing answer with
        # no citations. The same rule, applied here, is what turns that node
        # from an assertion in a docstring into a measured property.
        unsupported = (
            not answer.refused
            and not answer.citations
            and answer.intent not in _STRUCTURED_INTENTS
            and answer.intent is not QueryIntent.UNSUPPORTED
        )
        if unsupported:
            scores.unsupported_answers += 1

        ungrounded_structured = (
            not answer.refused and answer.intent in _STRUCTURED_INTENTS and not answer.tools_used
        )
        if ungrounded_structured:
            scores.ungrounded_structured_answers += 1

        faithful = (
            not unsupported and not ungrounded_structured and verified == len(answer.citations)
        )

        passed = (
            answer.intent is case.expected_intent
            and answer.refused == case.expect_refusal
            and not missing
            and len(answer.citations) >= case.min_citations
        )
        return QuestionResult(
            id=case.id,
            question=case.question,
            passed=passed,
            intent=answer.intent.value,
            expected_intent=case.expected_intent.value,
            refused=answer.refused,
            expected_refusal=case.expect_refusal,
            citations=len(answer.citations),
            citations_verified=verified,
            confidence=answer.confidence,
            missing_terms=missing,
            faithful=faithful,
            notes=_note(unsupported, ungrounded_structured, verified, len(answer.citations)),
        )

    async def _citation_holds(self, citation: Citation) -> bool:
        """The cited quote must still occur in the chunk the citation names.

        A citation with no chunk id cannot be checked, and an unverifiable
        citation is not a passing one -- so it counts as failed rather than
        being skipped. Silence about a citation nobody can open is how an
        unfalsifiable answer gets through.
        """
        if not citation.chunk_id:
            return False
        try:
            chunk_uuid = uuid.UUID(citation.chunk_id)
        except ValueError:
            return False
        chunk = await self._chunks.get(chunk_uuid)
        if chunk is None:
            return False
        return verify_quote(citation.quote, chunk.chunk_text).verified


def _note(unsupported: bool, ungrounded: bool, verified: int, total: int) -> str:
    if unsupported:
        return "answered with no citation and no structured tool behind it"
    if ungrounded:
        return "structured answer that names no tool"
    if verified < total:
        return f"{total - verified} of {total} citations did not re-verify"
    return ""
