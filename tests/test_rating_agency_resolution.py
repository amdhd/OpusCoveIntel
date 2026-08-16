"""`rating_agency` extraction — docs/review.md finding 9.

The one weak field in `make eval`: P 0.50 / R 0.50 on the LLM path and R 0.50
on rules, against >=0.94 for every other field. Two labelled instances, one
found by both extractors and one missed by both.

**The review guessed the wrong module.** It proposed starting with the `(m)` /
`id` national-scale suffixes in `app/rules/ratings.py`. That module normalises
the *notch* and already strips those suffixes; `BBB-` parses there without
complaint. The missing value is the *agency*, which that module never touches.

Running it showed two defects instead:

1. **The agency is looked up inside one chunk.** In `rating-report.pdf` the
   trigger sentence is its own 224-character chunk and names no agency; MARC is
   in the chunk before it and the chunk after. `_agency_near` searches only the
   text it is handed, so no context window reaches the label. Both extractors
   miss it, which is why this looked like a normalisation bug rather than a
   scoping one.
2. **The LLM path persists `"unknown"` as a value.** `rule_extractor` is
   explicit that "an absent agency is a fact, 'unknown' as a value is noise"
   and omits it; the pipeline wrote `output.rating_agency.value` behind a
   truthiness check, and `RatingAgency.UNKNOWN` is truthy. That is the false
   positive that separates the two paths' precision.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.clauses import Clause, Covenant
from app.db.models.documents import Document
from app.domain.enums import ClauseType, CovenantType, ExtractionMethod, RatingAgency
from app.extract.pipeline import ExtractionPipeline
from app.extract.rule_extractor import extract, resolve_document_agency
from app.extract.schemas import LLMCovenantExtraction
from app.llm.mock import MockLLMProvider
from app.llm.router import LLMRouter

pytestmark = pytest.mark.usefixtures("storage_root")

# The three blocks of the rating report fixture, as they chunk: the agency is
# named in the first and third, and the trigger stands alone in the second.
_RATIONALE = (
    "RATING RATIONALE\n\nMARC has affirmed its BBB+ rating on the RM250,000,000 "
    "Musharakah sukuk issued by Synthetic Retail REIT Berhad."
)
_TRIGGER = (
    "RATING TRIGGER\n\nThe transaction documents provide that where the rating is "
    "downgraded below BBB-, the Issuer shall procure additional security or, failing "
    "which, the trustee may declare the sukuk immediately due and payable."
)
_HEADROOM = (
    "COVENANT HEADROOM\n\nThe issuer is required to maintain a consolidated gearing "
    "ratio of not more than 2.50 times. MARC notes that headroom against this covenant "
    "narrowed following the acquisition completed in the period under review."
)


# -- the resolver itself ------------------------------------------------------


class TestResolveDocumentAgency:
    """One agency named anywhere in the document, or nothing."""

    def test_it_finds_the_agency_named_in_a_neighbouring_chunk(self) -> None:
        assert resolve_document_agency([_RATIONALE, _TRIGGER, _HEADROOM]) is RatingAgency.MARC

    def test_two_agencies_resolve_to_nothing(self) -> None:
        """Ambiguity narrows to nothing rather than guessing.

        The same choice finding 14 made: a wrong attribution is worse than an
        absent one, and MARC's `A-` and RAM's `AA3` are different scales.
        """
        both = [
            _RATIONALE,  # names MARC
            _TRIGGER,
            "The programme is also rated AA3 by RAM Rating Services Berhad.",
        ]

        assert resolve_document_agency(both) is None

    def test_no_agency_resolves_to_nothing(self) -> None:
        assert resolve_document_agency([_TRIGGER]) is None

    def test_a_repeated_single_agency_is_still_unambiguous(self) -> None:
        """MARC twice is one agency, not a conflict."""
        assert resolve_document_agency([_RATIONALE, _HEADROOM]) is RatingAgency.MARC


# -- the chunk-local extractor keeps its own behaviour ------------------------


def test_the_chunk_extractor_still_finds_an_agency_in_the_sentence() -> None:
    """The prospectus case, which already worked and must keep working."""
    text = (
        "RATING TRIGGER\n\nIn the event the rating assigned to the Sukuk Ijarah by MARC "
        "is downgraded below BBB+, the Issuer shall notify the Trustee within five "
        "business days of such downgrade."
    )

    triggers = [e for e in extract(text) if e.covenant_type is CovenantType.RATING_TRIGGER]

    assert triggers, "the rating trigger should still be detected"
    assert triggers[0].terms is not None
    assert triggers[0].terms.rating_agency is RatingAgency.MARC


def test_a_document_agency_fills_in_a_span_that_names_none() -> None:
    """The failing case, at the level the fix operates on."""
    without = [e for e in extract(_TRIGGER) if e.covenant_type is CovenantType.RATING_TRIGGER]
    assert without and without[0].terms is not None
    assert without[0].terms.rating_agency is RatingAgency.UNKNOWN, (
        "nothing in this span names an agency; that is the input to the fix"
    )

    with_doc = [
        e
        for e in extract(_TRIGGER, document_agency=RatingAgency.MARC)
        if e.covenant_type is CovenantType.RATING_TRIGGER
    ]

    assert with_doc and with_doc[0].terms is not None
    assert with_doc[0].terms.rating_agency is RatingAgency.MARC


def test_a_span_that_names_an_agency_outranks_the_document() -> None:
    """A document-level default must never override what the sentence says.

    A cross-border prospectus can carry a MARC national-scale trigger and a
    Fitch international one; the sentence is the better evidence.
    """
    text = (
        "RATING TRIGGER\n\nIn the event the rating assigned to the Sukuk Ijarah by MARC "
        "is downgraded below BBB+, the Issuer shall notify the Trustee within five "
        "business days of such downgrade."
    )

    found = [
        e
        for e in extract(text, document_agency=RatingAgency.RAM)
        if e.covenant_type is CovenantType.RATING_TRIGGER
    ]

    assert found and found[0].terms is not None
    assert found[0].terms.rating_agency is RatingAgency.MARC


# -- end to end, both persistence paths --------------------------------------


@pytest_asyncio.fixture
async def mock_router(db_session: AsyncSession) -> LLMRouter:
    return LLMRouter(db_session, provider=MockLLMProvider())


async def _rating_trigger_agencies(
    session: AsyncSession, document_id: uuid.UUID, method: ExtractionMethod
) -> list[str | None]:
    rows = (
        (
            await session.execute(
                select(Covenant)
                .join(Clause, Covenant.clause_id == Clause.id)
                .where(
                    Clause.document_id == document_id,
                    Clause.method == method,
                    Covenant.covenant_type == CovenantType.RATING_TRIGGER,
                )
            )
        )
        .scalars()
        .all()
    )
    values: list[str | None] = []
    for covenant in rows:
        raw = (covenant.thresholds_json or {}).get("rating_agency")
        values.append(str(raw) if raw is not None else None)
    return values


async def _rating_report(session: AsyncSession) -> Document:
    doc = (
        (await session.execute(select(Document).where(Document.filename.like("%rating-report%"))))
        .scalars()
        .first()
    )
    if doc is None:
        pytest.skip("rating-report fixture not ingested in this database")
    return doc


class TestUnknownIsNeverPersistedAsAValue:
    """`RatingAgency.UNKNOWN` is truthy, which is how it reached the column.

    An absent agency must be an absent *key*. The string makes a covenant look,
    to every reader that checks the field, like it carries an agency -- and it
    scored as the false positive that held LLM precision at 0.50.

    Pinned on `_build_thresholds` rather than end to end: whether the mock
    provider returns a rating trigger for a given fixture is incidental, and a
    test that asserts on an empty list passes for the wrong reason.
    """

    def _extraction(self, agency: RatingAgency | None) -> LLMCovenantExtraction:
        return LLMCovenantExtraction(
            clause_type=ClauseType.RATING_TRIGGER,
            covenant_type=CovenantType.RATING_TRIGGER,
            source_quote="the rating is downgraded below BBB-",
            confidence=0.95,
            trigger_rating="BBB-",
            rating_agency=agency,
        )

    def test_unknown_is_omitted(self, db_session: AsyncSession, mock_router: LLMRouter) -> None:
        pipeline = ExtractionPipeline(db_session, router=mock_router)

        thresholds = pipeline._build_thresholds(self._extraction(RatingAgency.UNKNOWN))

        assert thresholds["trigger_rating"] == "BBB-"
        assert "rating_agency" not in thresholds

    def test_a_real_agency_is_kept(self, db_session: AsyncSession, mock_router: LLMRouter) -> None:
        pipeline = ExtractionPipeline(db_session, router=mock_router)

        thresholds = pipeline._build_thresholds(self._extraction(RatingAgency.MARC))

        assert thresholds["rating_agency"] == RatingAgency.MARC.value

    def test_the_document_agency_fills_an_unknown(
        self, db_session: AsyncSession, mock_router: LLMRouter
    ) -> None:
        """The LLM path gets the same fallback the rule path does."""
        pipeline = ExtractionPipeline(db_session, router=mock_router)
        pipeline._document_agency = RatingAgency.MARC

        thresholds = pipeline._build_thresholds(self._extraction(RatingAgency.UNKNOWN))

        assert thresholds["rating_agency"] == RatingAgency.MARC.value

    def test_what_the_model_named_outranks_the_document(
        self, db_session: AsyncSession, mock_router: LLMRouter
    ) -> None:
        pipeline = ExtractionPipeline(db_session, router=mock_router)
        pipeline._document_agency = RatingAgency.RAM

        thresholds = pipeline._build_thresholds(self._extraction(RatingAgency.MARC))

        assert thresholds["rating_agency"] == RatingAgency.MARC.value


async def test_the_document_agency_does_not_leak_between_documents(
    db_session: AsyncSession,
    indexed_corpus: list[uuid.UUID],
    mock_router: LLMRouter,
    seeded_universe: None,
) -> None:
    """`extract --all` reuses one pipeline for the whole corpus.

    The resolved agency lives on the instance, so it has to be cleared per
    document. Otherwise a document that names no agency inherits whichever one
    the previous document named and stamps its covenants with it -- a wrong
    attribution, which is worse than the absent value this fix set out to fill.
    """
    pipeline = ExtractionPipeline(db_session, router=mock_router)

    # A document that names MARC, then every other document in the corpus.
    report = await _rating_report(db_session)
    await pipeline.extract(report.id, force=True)
    assert pipeline._document_agency is RatingAgency.MARC

    for document_id in indexed_corpus:
        if document_id != report.id:
            await pipeline.extract(document_id, force=True)

    # The trust deed names no agency at all; nothing it produced may claim one.
    trust_deed = (
        (await db_session.execute(select(Document).where(Document.filename.like("%trust-deed%"))))
        .scalars()
        .first()
    )
    if trust_deed is None:
        pytest.skip("trust-deed fixture not ingested")

    for method in (ExtractionMethod.RULE, ExtractionMethod.LLM):
        assert not any(await _rating_trigger_agencies(db_session, trust_deed.id, method)), (
            f"{method.value} stamped an agency on a document that names none"
        )
