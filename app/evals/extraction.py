"""Scoring extraction output against the golden labels.

Reads what the pipeline persisted -- through the repositories, like every other
reader (CLAUDE.md 3) -- matches it to `app.evals.labels`, and reports per-field
precision, recall and F1.

**Scored per extraction method, not in aggregate.** PLAN.md 3 runs the rule and
LLM extractors in parallel specifically so "did the LLM actually help?" is
measurable. Pooling their output into one number destroys the only measurement
that question has, so `score_document` returns one `MethodScores` per method
present, plus the union.

**Citations are checked with the pipeline's own code.** The citation metric
calls `app.extract.citations.verify_quote`, which is what the pipeline gates on
(CLAUDE.md 1.3). A harness with its own verifier measures something the
pipeline does not enforce, and the two drift apart precisely when it matters.
It re-runs verification against the stored chunk rather than trusting
`clauses.citation_verified`: the flag records what a past run decided, and the
point of an eval is to check that claim rather than repeat it.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.clauses import Clause, Covenant
from app.db.repositories.clauses import CallScheduleRepository, CovenantRepository
from app.db.repositories.documents import DocumentChunkRepository, DocumentRepository
from app.domain.enums import CallType, ClauseType, CovenantType, ExtractionMethod, RatingAgency
from app.domain.rules import ComparisonOperator
from app.evals.labels import (
    CallScheduleLabel,
    CovenantLabel,
    call_schedule_labels_for,
    covenant_labels_for,
    document_name,
)
from app.evals.metrics import Score, ScoreBoard, ratio, to_decimal, values_match
from app.extract.citations import verify_quote
from app.extract.linking import resolve_instrument

# The fields scored on every matched covenant. `covenant_type` is the matching
# key, so its score is the covenant-detection score -- named here anyway,
# because "which covenants were found at all" is the headline number and
# leaving it implicit hides it.
COVENANT_FIELDS: tuple[str, ...] = (
    "covenant_type",
    "clause_type",
    "operator",
    "threshold_amount",
    "threshold_currency",
    "threshold_ratio",
    "trigger_rating",
    "rating_agency",
)

CALL_SCHEDULE_FIELDS: tuple[str, ...] = ("call_date", "call_price", "call_type")


@dataclass(frozen=True)
class PredictedCovenant:
    """One persisted covenant, flattened into the shape the labels are in."""

    covenant_type: CovenantType
    clause_type: ClauseType
    page_number: int
    quote: str
    chunk_id: uuid.UUID | None
    method: ExtractionMethod
    operator: ComparisonOperator | None = None
    threshold_amount: Decimal | None = None
    threshold_currency: str | None = None
    threshold_ratio: Decimal | None = None
    trigger_rating: str | None = None
    rating_agency: RatingAgency | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class PredictedCall:
    call_date: object
    call_price: Decimal | None
    call_type: CallType
    method: ExtractionMethod


@dataclass
class CitationTally:
    """Citation outcomes, counted rather than averaged.

    `quote_reverified` is the count whose stored quote still occurs in the chunk
    it names. `evidence_matched` is the count that quote the right clause -- a
    citation can be perfectly verifiable and still point at the wrong sentence,
    and only the label knows which sentence was the right one.
    """

    predicted: int = 0
    quote_reverified: int = 0
    evidence_matched: int = 0
    labels: int = 0

    @property
    def precision(self) -> float | None:
        return ratio(self.evidence_matched, self.predicted)

    @property
    def recall(self) -> float | None:
        return ratio(self.evidence_matched, self.labels)

    @property
    def reverification_rate(self) -> float | None:
        return ratio(self.quote_reverified, self.predicted)

    def as_dict(self) -> dict[str, object]:
        return {
            "predicted": self.predicted,
            "labels": self.labels,
            "quote_reverified": self.quote_reverified,
            "evidence_matched": self.evidence_matched,
            "precision": self.precision,
            "recall": self.recall,
            "reverification_rate": self.reverification_rate,
        }


@dataclass
class MethodScores:
    """Everything measured about one extractor over one document."""

    method: str
    covenant_fields: dict[str, Score] = field(default_factory=dict)
    citations: CitationTally = field(default_factory=CitationTally)
    matched: int = 0
    unmatched_labels: int = 0
    unmatched_predictions: int = 0
    # Recall broken out by document language: the corpus states the same gearing
    # covenant in English and in Bahasa Malaysia, and one number hides which of
    # them the extractor can read (PLAN.md 9, open question 7).
    recall_by_language: dict[str, tuple[int, int]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "method": self.method,
            "matched": self.matched,
            "unmatched_labels": self.unmatched_labels,
            "unmatched_predictions": self.unmatched_predictions,
            "covenant_fields": {
                name: score.as_dict() for name, score in sorted(self.covenant_fields.items())
            },
            "citations": self.citations.as_dict(),
            "recall_by_language": {
                language: {"found": found, "labelled": labelled, "recall": ratio(found, labelled)}
                for language, (found, labelled) in sorted(self.recall_by_language.items())
            },
        }


@dataclass
class DocumentScores:
    """One labelled document: covenants per extractor, call schedules once.

    Call schedules are not split by method, because only one extractor produces
    them -- `ExtractionPipeline` has no call-schedule step, so the LLM path
    would score a flat zero on a capability it was never given, and a reader
    would take that for a failure rather than an absence. `call_methods` names
    which extractors actually contributed rows.
    """

    sha256: str
    name: str
    document_id: uuid.UUID
    by_method: dict[str, MethodScores] = field(default_factory=dict)
    call_fields: dict[str, Score] = field(default_factory=dict)
    call_methods: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        return {
            "document": self.name,
            "sha256": self.sha256,
            "document_id": str(self.document_id),
            "methods": {name: scores.as_dict() for name, scores in sorted(self.by_method.items())},
            "call_schedules": {
                "extracted_by": list(self.call_methods),
                "fields": {
                    name: score.as_dict() for name, score in sorted(self.call_fields.items())
                },
            },
        }


class ExtractionEvaluator:
    """Scores persisted extraction output for one labelled document."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._documents = DocumentRepository(session)
        self._covenants = CovenantRepository(session)
        self._calls = CallScheduleRepository(session)
        self._chunks = DocumentChunkRepository(session)

    async def score_document(self, sha256: str) -> DocumentScores | None:
        """Score one labelled document, or None if it is not in the corpus."""
        document = await self._documents.get_by_sha256(sha256)
        if document is None:
            return None

        scores = DocumentScores(sha256=sha256, name=document_name(sha256), document_id=document.id)
        rows = [
            (covenant, clause)
            for covenant, clause in await self._covenants.list_with_clause_for_document(document.id)
            if covenant.method is not ExtractionMethod.HUMAN
        ]
        predictions = [self._to_prediction(covenant, clause) for covenant, clause in rows]
        chunk_text = await self._chunk_text_for(rows)

        calls = await self._predicted_calls(document.id)
        covenant_labels = covenant_labels_for(sha256)

        # One table per extractor and no union row. Pooling them looks like the
        # question an operator asks ("does the system get this right?") and is
        # not: the two extractors run over the same spans by design (PLAN.md 3),
        # so every covenant both of them find appears twice, and one of the two
        # is then counted as a false positive against a label already taken. The
        # pooled precision that produces measures the duplication, not the
        # system.
        for method in sorted({prediction.method.value for prediction in predictions}):
            scores.by_method[method] = self._score_method(
                method=method,
                predictions=[p for p in predictions if p.method.value == method],
                labels=covenant_labels,
                chunk_text=chunk_text,
            )

        scores.call_fields = self._score_calls(calls, call_schedule_labels_for(sha256))
        scores.call_methods = tuple(sorted({call.method.value for call in calls}))
        return scores

    # -- scoring -----------------------------------------------------------

    def _score_method(
        self,
        *,
        method: str,
        predictions: list[PredictedCovenant],
        labels: tuple[CovenantLabel, ...],
        chunk_text: dict[uuid.UUID, str],
    ) -> MethodScores:
        result = MethodScores(method=method)
        board = ScoreBoard()
        citations = CitationTally(predicted=len(predictions), labels=len(labels))

        pairs, unmatched_labels, unmatched_predictions = match_covenants(labels, predictions)
        result.matched = len(pairs)
        result.unmatched_labels = len(unmatched_labels)
        result.unmatched_predictions = len(unmatched_predictions)

        for label, prediction in pairs:
            for name in COVENANT_FIELDS:
                board.compare(name, getattr(label, name), getattr(prediction, name))
            self._score_citation(label, prediction, chunk_text, citations)
            self._count_language(result, label, found=True)

        for label in unmatched_labels:
            for name in COVENANT_FIELDS:
                board.missing(name, getattr(label, name))
            self._count_language(result, label, found=False)

        for prediction in unmatched_predictions:
            for name in COVENANT_FIELDS:
                board.spurious(name, getattr(prediction, name))
            # An unmatched prediction still made a citation claim; re-verifying
            # it keeps `quote_reverified` a statement about every stored quote
            # rather than only the ones that happened to land on a label.
            if self._reverifies(prediction, chunk_text):
                citations.quote_reverified += 1

        result.covenant_fields = {name: board.score(name) for name in COVENANT_FIELDS}
        result.citations = citations
        return result

    def _score_calls(
        self, calls: list[PredictedCall], labels: tuple[CallScheduleLabel, ...]
    ) -> dict[str, Score]:
        """Call schedules, matched on the date within tolerance.

        Matching on date rather than on position: a schedule read out of order,
        or one with a row missing, must not shift every later row into a
        mismatch and report three errors where there is one.
        """
        board = ScoreBoard()
        remaining = list(calls)
        for label in labels:
            match = next(
                (call for call in remaining if values_match(label.call_date, call.call_date)),
                None,
            )
            if match is None:
                for name in CALL_SCHEDULE_FIELDS:
                    board.missing(name, getattr(label, name))
                continue
            remaining.remove(match)
            for name in CALL_SCHEDULE_FIELDS:
                board.compare(name, getattr(label, name), getattr(match, name))

        for spare in remaining:
            for name in CALL_SCHEDULE_FIELDS:
                board.spurious(name, getattr(spare, name))

        return {name: board.score(name) for name in CALL_SCHEDULE_FIELDS}

    def _score_citation(
        self,
        label: CovenantLabel,
        prediction: PredictedCovenant,
        chunk_text: dict[uuid.UUID, str],
        tally: CitationTally,
    ) -> None:
        if self._reverifies(prediction, chunk_text):
            tally.quote_reverified += 1
        if quotes_labelled_clause(label.evidence, prediction.quote):
            tally.evidence_matched += 1

    def _reverifies(self, prediction: PredictedCovenant, chunk_text: dict[uuid.UUID, str]) -> bool:
        if prediction.chunk_id is None:
            return False
        text = chunk_text.get(prediction.chunk_id)
        if text is None:
            return False
        return verify_quote(prediction.quote, text).verified

    def _count_language(self, result: MethodScores, label: CovenantLabel, *, found: bool) -> None:
        key = label.language.value
        seen, total = result.recall_by_language.get(key, (0, 0))
        result.recall_by_language[key] = (seen + (1 if found else 0), total + 1)

    # -- loading -----------------------------------------------------------

    async def _chunk_text_for(
        self, rows: Sequence[tuple[Covenant, Clause]]
    ) -> dict[uuid.UUID, str]:
        """Chunk text for every chunk a clause cites, fetched once each."""
        text: dict[uuid.UUID, str] = {}
        for _covenant, clause in rows:
            chunk_id = clause.source_chunk_id
            if chunk_id is None or chunk_id in text:
                continue
            chunk = await self._chunks.get(chunk_id)
            if chunk is not None:
                text[chunk_id] = chunk.chunk_text
        return text

    async def _predicted_calls(self, document_id: uuid.UUID) -> list[PredictedCall]:
        """Call schedules for the instrument this document describes.

        `call_schedules` hangs off the instrument, not the document, so the link
        has to be resolved the same way the extractor resolved it -- through
        `resolve_instrument`. An unlinked document has no call schedules to
        score, which is itself the finding.
        """
        instrument_id = await resolve_instrument(self._session, document_id)
        if instrument_id is None:
            return []
        return [
            PredictedCall(
                call_date=row.call_date,
                call_price=row.call_price,
                call_type=row.call_type,
                method=row.method,
            )
            for row in await self._calls.list_for_instrument(instrument_id)
            # A row a person entered is not a prediction. `make seed` writes
            # demo call dates as `human`, and counting those as extractor output
            # scored the seed data -- which produced a false positive on a
            # document whose schedule the extractor had read perfectly.
            if row.method is not ExtractionMethod.HUMAN
        ]

    def _to_prediction(self, covenant: Covenant, clause: Clause) -> PredictedCovenant:
        thresholds = covenant.thresholds_json or {}
        conditions = covenant.conditions_json or {}
        return PredictedCovenant(
            covenant_type=covenant.covenant_type,
            clause_type=clause.clause_type,
            page_number=clause.page_number,
            quote=clause.source_quote,
            chunk_id=clause.source_chunk_id,
            method=covenant.method,
            operator=_as_enum(ComparisonOperator, conditions.get("operator")),
            threshold_amount=covenant.threshold_amount,
            threshold_currency=covenant.threshold_currency,
            threshold_ratio=to_decimal(thresholds.get("threshold_ratio")),
            trigger_rating=_as_str(thresholds.get("trigger_rating")),
            rating_agency=_as_enum(RatingAgency, thresholds.get("rating_agency")),
            confidence=covenant.confidence,
        )


def quotes_labelled_clause(evidence: str, quote: str) -> bool:
    """Whether a quote actually contains the labelled clause.

    Uses `verify_quote` -- the pipeline's checker -- but accepts only its exact
    and normalised legs, not the fuzzy one.

    The fuzzy leg exists so a model that drops a footnote marker still passes
    the pipeline's gate, and at a 0.92 partial ratio it will happily match
    "gearing ratio of not more than **2.25** times" against a quote saying
    **1.75**: one digit in a fifty-character phrase is well inside the
    threshold. Three documents in this corpus state a gearing covenant at three
    different levels precisely so an extractor has to tell them apart, so a
    citation metric that cannot is measuring nothing. A label is copied verbatim
    from the fixture, so exact-or-normalised is the right bar for it -- line
    wrapping is absorbed, a changed number is not.

    Re-verifying the extractor's own quote against its chunk stays on the full
    three-leg check, because there the point is to re-run the gate the pipeline
    actually applied.
    """
    return verify_quote(evidence, quote).method in ("exact", "normalised")


def match_covenants(
    labels: tuple[CovenantLabel, ...],
    predictions: list[PredictedCovenant],
) -> tuple[
    list[tuple[CovenantLabel, PredictedCovenant]],
    list[CovenantLabel],
    list[PredictedCovenant],
]:
    """Pair labels with predictions of the same covenant type.

    Greedy by agreement, best pair first. The corpus states the *same* gearing
    covenant twice -- once in English on page 2 and once in Bahasa Malaysia on
    page 4 -- so covenant type alone does not identify a label, and pairing by
    order of appearance would score the English reading against the Malay label.
    Agreement on the page and on the stated values breaks the tie, which is the
    same evidence a human would use.
    """
    candidates: list[tuple[int, int, int]] = []
    for label_index, label in enumerate(labels):
        for prediction_index, prediction in enumerate(predictions):
            if label.covenant_type is not prediction.covenant_type:
                continue
            candidates.append((_agreement(label, prediction), label_index, prediction_index))

    # Sort by agreement descending, then by index so the pairing is stable and a
    # re-run of the same corpus produces the same report.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    taken_labels: set[int] = set()
    taken_predictions: set[int] = set()
    pairs: list[tuple[CovenantLabel, PredictedCovenant]] = []
    for _agreed, label_index, prediction_index in candidates:
        if label_index in taken_labels or prediction_index in taken_predictions:
            continue
        taken_labels.add(label_index)
        taken_predictions.add(prediction_index)
        pairs.append((labels[label_index], predictions[prediction_index]))

    unmatched_labels = [label for index, label in enumerate(labels) if index not in taken_labels]
    unmatched_predictions = [
        prediction for index, prediction in enumerate(predictions) if index not in taken_predictions
    ]
    return pairs, unmatched_labels, unmatched_predictions


def _agreement(label: CovenantLabel, prediction: PredictedCovenant) -> int:
    """How well a candidate pair agrees, used only to order the matching."""
    score = 4 if label.page_number == prediction.page_number else 0
    for name in COVENANT_FIELDS:
        expected = getattr(label, name)
        actual = getattr(prediction, name)
        if expected is not None and actual is not None and values_match(expected, actual):
            score += 1
    return score


def _as_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _as_enum[EnumT](enum_type: type[EnumT], value: object) -> EnumT | None:
    if value is None:
        return None
    try:
        return enum_type(value)  # type: ignore[call-arg]
    except ValueError:
        return None
