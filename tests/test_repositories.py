"""Repository CRUD tests against a real Postgres.

These run inside a transaction that is rolled back, so they leave no trace.
They test against real Postgres rather than a stand-in because the behaviour
worth verifying -- CHECK constraints, unique constraints, cascade deletes,
Decimal round-tripping -- only exists in the real database.
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.clauses import CallSchedule, Clause, Covenant, RatingTrigger
from app.db.models.documents import Document, DocumentChunk, DocumentPage
from app.db.models.instruments import Instrument
from app.db.models.ops import ExtractionJob, HumanReview, LLMCall
from app.db.models.portfolio import Portfolio, PortfolioHolding
from app.db.repositories.clauses import (
    CallScheduleRepository,
    ClauseRepository,
    CovenantRepository,
    RatingTriggerRepository,
)
from app.db.repositories.documents import (
    DocumentChunkRepository,
    DocumentPageRepository,
    DocumentRepository,
)
from app.db.repositories.instruments import InstrumentRepository
from app.db.repositories.ops import (
    ExtractionJobRepository,
    HumanReviewRepository,
    LLMCallRepository,
)
from app.db.repositories.portfolio import PortfolioHoldingRepository, PortfolioRepository
from app.domain.enums import (
    ClauseType,
    CovenantType,
    DocumentStatus,
    ExtractionStatus,
    InstrumentType,
    JobType,
    LLMStage,
    RatingAgency,
    ReviewStatus,
    ReviewTrigger,
    TriggerDirection,
)
from app.rules.ratings import rank

# ---------------------------------------------------------------- helpers


def make_document(sha: str = "a" * 64, **kw: object) -> Document:
    defaults: dict[str, object] = {
        "sha256": sha,
        "filename": "synthetic_prospectus.pdf",
        "status": DocumentStatus.UPLOADED,
    }
    return Document(**{**defaults, **kw})


def make_instrument(name: str = "RM300m Test Sukuk", **kw: object) -> Instrument:
    defaults: dict[str, object] = {
        "issuer_name": "Synthetic Issuer Bhd",
        "instrument_name": name,
        "instrument_type": InstrumentType.SUKUK,
        "currency": "MYR",
    }
    return Instrument(**{**defaults, **kw})


# ---------------------------------------------------------------- documents


class TestDocumentRepository:
    async def test_add_assigns_uuid7_and_timestamps(self, db_session: AsyncSession) -> None:
        repo = DocumentRepository(db_session)
        doc = await repo.add(make_document())

        assert doc.id is not None
        assert doc.id.version == 7
        assert doc.created_at is not None
        assert doc.created_at.tzinfo is not None, "timestamps must be timezone-aware"

    async def test_get_round_trip(self, db_session: AsyncSession) -> None:
        repo = DocumentRepository(db_session)
        doc = await repo.add(make_document())
        assert (await repo.get(doc.id)) is doc

    async def test_get_by_sha256_is_the_dedup_path(self, db_session: AsyncSession) -> None:
        repo = DocumentRepository(db_session)
        await repo.add(make_document(sha="b" * 64))

        assert await repo.exists_by_sha256("b" * 64)
        assert not await repo.exists_by_sha256("c" * 64)
        found = await repo.get_by_sha256("b" * 64)
        assert found is not None and found.sha256 == "b" * 64

    async def test_duplicate_sha256_is_rejected_by_the_database(
        self, db_session: AsyncSession
    ) -> None:
        """Dedup is a constraint, not a convention -- two workers can race."""
        repo = DocumentRepository(db_session)
        await repo.add(make_document(sha="d" * 64))
        with pytest.raises(IntegrityError):
            await repo.add(make_document(sha="d" * 64, filename="other.pdf"))

    async def test_get_or_raise_reports_a_missing_row(self, db_session: AsyncSession) -> None:
        from app.domain.ids import uuid7

        repo = DocumentRepository(db_session)
        with pytest.raises(LookupError, match="not found"):
            await repo.get_or_raise(uuid7())

    async def test_update_and_delete(self, db_session: AsyncSession) -> None:
        repo = DocumentRepository(db_session)
        doc = await repo.add(make_document())

        await repo.update(doc, status=DocumentStatus.PARSED, page_count=42)
        assert doc.status == DocumentStatus.PARSED
        assert doc.page_count == 42

        await repo.delete(doc)
        assert await repo.get(doc.id) is None

    async def test_list_by_status_and_count(self, db_session: AsyncSession) -> None:
        repo = DocumentRepository(db_session)
        await repo.add(make_document(sha="1" * 64, status=DocumentStatus.PARSED))
        await repo.add(make_document(sha="2" * 64, status=DocumentStatus.PARSED))
        await repo.add(make_document(sha="3" * 64, status=DocumentStatus.UPLOADED))

        assert len(await repo.list_by_status(DocumentStatus.PARSED)) == 2
        assert await repo.count(status=DocumentStatus.UPLOADED) == 1

    async def test_invalid_confidence_is_rejected(self, db_session: AsyncSession) -> None:
        repo = DocumentRepository(db_session)
        with pytest.raises(IntegrityError):
            await repo.add(make_document(parse_confidence=1.5))


class TestDocumentPageRepository:
    async def test_vlm_pages_must_record_a_reason(self, db_session: AsyncSession) -> None:
        """CLAUDE.md 4: 'why did this document cost $8' must be answerable."""
        docs = DocumentRepository(db_session)
        pages = DocumentPageRepository(db_session)
        doc = await docs.add(make_document())

        with pytest.raises(IntegrityError):
            await pages.add(
                DocumentPage(document_id=doc.id, page_number=1, vlm_used=True, vlm_reason=None)
            )

    async def test_list_needing_vlm_excludes_already_processed(
        self, db_session: AsyncSession
    ) -> None:
        docs = DocumentRepository(db_session)
        pages = DocumentPageRepository(db_session)
        doc = await docs.add(make_document())

        await pages.add(DocumentPage(document_id=doc.id, page_number=1, confidence=0.2))
        await pages.add(DocumentPage(document_id=doc.id, page_number=2, confidence=0.95))
        await pages.add(
            DocumentPage(
                document_id=doc.id,
                page_number=3,
                confidence=0.1,
                vlm_used=True,
                vlm_reason="no_text_layer",
                # A VLM-processed page must carry what the VLM read; the
                # `vlm_use_requires_ocr_text` CHECK refuses the flag without it.
                ocr_text="transcribed page text",
            )
        )

        needing = await pages.list_needing_vlm(doc.id, confidence_threshold=0.5)
        assert [p.page_number for p in needing] == [1]
        assert await pages.count_vlm_used(doc.id) == 1

    async def test_page_numbers_are_unique_per_document(self, db_session: AsyncSession) -> None:
        docs = DocumentRepository(db_session)
        pages = DocumentPageRepository(db_session)
        doc = await docs.add(make_document())

        await pages.add(DocumentPage(document_id=doc.id, page_number=1))
        with pytest.raises(IntegrityError):
            await pages.add(DocumentPage(document_id=doc.id, page_number=1))


class TestDocumentChunkRepository:
    async def test_chunk_stores_span_offsets(self, db_session: AsyncSession) -> None:
        """CLAUDE.md 1.2: the citation chain starts with exact offsets."""
        docs = DocumentRepository(db_session)
        chunks = DocumentChunkRepository(db_session)
        doc = await docs.add(make_document())

        chunk = await chunks.add(
            DocumentChunk(
                document_id=doc.id,
                page_number=12,
                chunk_text="The Issuer shall not create any Security Interest...",
                char_start=100,
                char_end=155,
                hash="h1",
            )
        )
        assert chunk.char_end > chunk.char_start
        assert chunk.fts_config == "english"

    async def test_reversed_span_is_rejected(self, db_session: AsyncSession) -> None:
        docs = DocumentRepository(db_session)
        chunks = DocumentChunkRepository(db_session)
        doc = await docs.add(make_document())

        with pytest.raises(IntegrityError):
            await chunks.add(
                DocumentChunk(
                    document_id=doc.id,
                    page_number=1,
                    chunk_text="x",
                    char_start=500,
                    char_end=100,
                    hash="h2",
                )
            )

    async def test_only_supported_fts_configs_are_allowed(self, db_session: AsyncSession) -> None:
        """Postgres has no Malay stemmer; BM chunks must fall back to 'simple'."""
        docs = DocumentRepository(db_session)
        chunks = DocumentChunkRepository(db_session)
        doc = await docs.add(make_document())

        await chunks.add(
            DocumentChunk(
                document_id=doc.id,
                page_number=1,
                chunk_text="Penerbit tidak boleh mewujudkan sebarang gadaian",
                char_start=0,
                char_end=47,
                hash="ms1",
                fts_config="simple",
            )
        )
        with pytest.raises(IntegrityError):
            await chunks.add(
                DocumentChunk(
                    document_id=doc.id,
                    page_number=1,
                    chunk_text="x",
                    char_start=0,
                    char_end=1,
                    hash="ms2",
                    fts_config="malay",
                )
            )

    async def test_embedding_round_trips_at_configured_dimension(
        self, db_session: AsyncSession
    ) -> None:
        from app.core.config import get_settings

        dim = get_settings().VECTOR_DIMENSION
        docs = DocumentRepository(db_session)
        chunks = DocumentChunkRepository(db_session)
        doc = await docs.add(make_document())

        chunk = await chunks.add(
            DocumentChunk(
                document_id=doc.id,
                page_number=1,
                chunk_text="x",
                char_start=0,
                char_end=1,
                hash="emb",
                embedding=[0.1] * dim,
                embedding_model="text-embedding-v4",
            )
        )
        assert chunk.embedding is not None
        assert len(chunk.embedding) == dim
        assert await chunks.count_embedded(doc.id) == 1

    async def test_wrong_embedding_dimension_is_rejected(self, db_session: AsyncSession) -> None:
        docs = DocumentRepository(db_session)
        chunks = DocumentChunkRepository(db_session)
        doc = await docs.add(make_document())

        with pytest.raises(Exception):  # noqa: B017 -- pgvector raises a DataError subclass
            await chunks.add(
                DocumentChunk(
                    document_id=doc.id,
                    page_number=1,
                    chunk_text="x",
                    char_start=0,
                    char_end=1,
                    hash="bad",
                    embedding=[0.1] * 3,
                )
            )

    async def test_deleting_a_document_cascades_to_chunks(self, db_session: AsyncSession) -> None:
        docs = DocumentRepository(db_session)
        chunks = DocumentChunkRepository(db_session)
        doc = await docs.add(make_document())
        await chunks.add(
            DocumentChunk(
                document_id=doc.id,
                page_number=1,
                chunk_text="x",
                char_start=0,
                char_end=1,
                hash="c1",
            )
        )

        await docs.delete(doc)
        assert await chunks.count(document_id=doc.id) == 0


# ---------------------------------------------------------------- instruments


class TestInstrumentRepository:
    async def test_set_rating_keeps_the_ordinal_rank_in_sync(
        self, db_session: AsyncSession
    ) -> None:
        repo = InstrumentRepository(db_session)
        inst = await repo.add(make_instrument(rating_agency=RatingAgency.MARC))

        await repo.set_rating(inst, "A-")
        assert inst.current_rating == "A-"
        assert inst.current_rating_rank == rank("A-")

    async def test_ram_notation_ranks_onto_the_shared_scale(self, db_session: AsyncSession) -> None:
        """A RAM 'AA3' holding must rank identically to a MARC 'AA-' one."""
        repo = InstrumentRepository(db_session)
        ram = await repo.add(make_instrument("RAM Sukuk", rating_agency=RatingAgency.RAM))
        marc = await repo.add(make_instrument("MARC Sukuk", rating_agency=RatingAgency.MARC))

        await repo.set_rating(ram, "AA3")
        await repo.set_rating(marc, "AA-")
        assert ram.current_rating_rank == marc.current_rating_rank

    async def test_unparseable_rating_leaves_rank_null_rather_than_guessing(
        self, db_session: AsyncSession
    ) -> None:
        repo = InstrumentRepository(db_session)
        inst = await repo.add(make_instrument())

        await repo.set_rating(inst, "NR")
        assert inst.current_rating == "NR"
        assert inst.current_rating_rank is None

    async def test_list_rated_at_or_below_uses_ordinal_not_lexical_order(
        self, db_session: AsyncSession
    ) -> None:
        """The flagship query. Lexically 'AA-' < 'A+', which would be backwards."""
        repo = InstrumentRepository(db_session)
        aa = await repo.add(make_instrument("AA sukuk"))
        a_minus = await repo.add(make_instrument("A- sukuk"))
        bbb = await repo.add(make_instrument("BBB sukuk"))
        await repo.set_rating(aa, "AA-")
        await repo.set_rating(a_minus, "A-")
        await repo.set_rating(bbb, "BBB")

        below_a = await repo.list_rated_at_or_below(rank("A"))
        names = {i.instrument_name for i in below_a}
        assert names == {"A- sukuk", "BBB sukuk"}
        assert "AA sukuk" not in names

    async def test_issue_size_round_trips_as_exact_decimal(self, db_session: AsyncSession) -> None:
        """CLAUDE.md 6: money is Decimal. Float would lose exactness."""
        repo = InstrumentRepository(db_session)
        inst = await repo.add(make_instrument(issue_size=Decimal("300000000.5000")))
        await db_session.refresh(inst)

        assert isinstance(inst.issue_size, Decimal)
        assert inst.issue_size == Decimal("300000000.5000")

    async def test_isin_is_unique(self, db_session: AsyncSession) -> None:
        repo = InstrumentRepository(db_session)
        await repo.add(make_instrument("One", isin="MYSYN0000009"))
        with pytest.raises(IntegrityError):
            await repo.add(make_instrument("Two", isin="MYSYN0000009"))

    async def test_negative_issue_size_is_rejected(self, db_session: AsyncSession) -> None:
        repo = InstrumentRepository(db_session)
        with pytest.raises(IntegrityError):
            await repo.add(make_instrument(issue_size=Decimal("-1")))


# ---------------------------------------------------------------- clauses


class TestClauseRepository:
    async def _clause(self, db_session: AsyncSession, **kw: object) -> Clause:
        doc = await DocumentRepository(db_session).add(make_document())
        defaults: dict[str, object] = {
            "document_id": doc.id,
            "clause_type": ClauseType.CROSS_DEFAULT,
            "clause_text": "Cross default threshold of RM30,000,000",
            "page_number": 87,
            "source_quote": "RM30,000,000",
            "citation_verified": True,
            "citation_match_score": 1.0,
        }
        return Clause(**{**defaults, **kw})

    async def test_add_verified_persists_a_verified_clause(self, db_session: AsyncSession) -> None:
        repo = ClauseRepository(db_session)
        clause = await repo.add_verified(await self._clause(db_session))
        assert clause.id is not None

    async def test_add_verified_refuses_an_unverified_citation(
        self, db_session: AsyncSession
    ) -> None:
        """CLAUDE.md 1.3 -- the repository is the last line of defence."""
        repo = ClauseRepository(db_session)
        clause = await self._clause(
            db_session,
            citation_verified=False,
            citation_match_score=None,
            extraction_status=ExtractionStatus.CITATION_FAILED,
        )
        with pytest.raises(ValueError, match="unverified citation"):
            await repo.add_verified(clause)

    async def test_verified_flag_requires_a_score(self, db_session: AsyncSession) -> None:
        """Stops a code path flipping the flag without running the check."""
        repo = ClauseRepository(db_session)
        clause = await self._clause(db_session, citation_verified=True, citation_match_score=None)
        with pytest.raises(IntegrityError):
            await repo.add(clause)

    async def test_empty_source_quote_is_rejected(self, db_session: AsyncSession) -> None:
        repo = ClauseRepository(db_session)
        with pytest.raises(IntegrityError):
            await repo.add(await self._clause(db_session, source_quote=""))

    async def test_list_unverified_feeds_the_review_queue(self, db_session: AsyncSession) -> None:
        repo = ClauseRepository(db_session)
        await repo.add(
            await self._clause(
                db_session,
                citation_verified=False,
                citation_match_score=None,
                extraction_status=ExtractionStatus.CITATION_FAILED,
            )
        )
        assert len(await repo.list_unverified()) == 1


class TestCovenantRepository:
    async def test_threshold_query_uses_the_denormalised_column(
        self, db_session: AsyncSession
    ) -> None:
        """'Cross-default below RM50m' -- an indexed numeric predicate."""
        docs = DocumentRepository(db_session)
        clauses = ClauseRepository(db_session)
        covenants = CovenantRepository(db_session)
        doc = await docs.add(make_document())

        for amount in (Decimal("30000000"), Decimal("80000000"), Decimal("20000000")):
            clause = await clauses.add(
                Clause(
                    document_id=doc.id,
                    clause_type=ClauseType.CROSS_DEFAULT,
                    clause_text="...",
                    page_number=1,
                    source_quote=f"RM{amount}",
                    citation_verified=True,
                    citation_match_score=1.0,
                )
            )
            await covenants.add(
                Covenant(
                    clause_id=clause.id,
                    covenant_type=CovenantType.CROSS_DEFAULT,
                    threshold_amount=amount,
                    threshold_currency="MYR",
                )
            )

        below = await covenants.list_below_threshold(
            CovenantType.CROSS_DEFAULT, Decimal("50000000")
        )
        # The query filters out NULL thresholds; narrow explicitly so the
        # comparison is well-typed rather than relying on that invariant.
        amounts = sorted(c.threshold_amount for c in below if c.threshold_amount is not None)
        assert amounts == [Decimal("20000000"), Decimal("30000000")]
        assert len(amounts) == len(below)

    async def test_threshold_amount_requires_a_currency(self, db_session: AsyncSession) -> None:
        """An amount without a currency is not a comparable quantity."""
        docs = DocumentRepository(db_session)
        clauses = ClauseRepository(db_session)
        covenants = CovenantRepository(db_session)
        doc = await docs.add(make_document())
        clause = await clauses.add(
            Clause(
                document_id=doc.id,
                clause_type=ClauseType.CROSS_DEFAULT,
                clause_text="...",
                page_number=1,
                source_quote="x",
                citation_verified=True,
                citation_match_score=1.0,
            )
        )
        with pytest.raises(IntegrityError):
            await covenants.add(
                Covenant(
                    clause_id=clause.id,
                    covenant_type=CovenantType.CROSS_DEFAULT,
                    threshold_amount=Decimal("1000"),
                    threshold_currency=None,
                )
            )


class TestRatingTriggerRepository:
    async def test_triggers_are_selected_by_ordinal_rank(self, db_session: AsyncSession) -> None:
        instruments = InstrumentRepository(db_session)
        triggers = RatingTriggerRepository(db_session)

        for name, trigger_rating in [("A trigger", "A"), ("BBB trigger", "BBB")]:
            inst = await instruments.add(make_instrument(name))
            await triggers.add(
                RatingTrigger(
                    instrument_id=inst.id,
                    trigger_rating=trigger_rating,
                    trigger_rank=rank(trigger_rating),
                    trigger_direction=TriggerDirection.DOWNGRADE_BELOW,
                    consequence="Investor put",
                )
            )

        fires_at_a = await triggers.list_triggered_at_rank(rank("A"))
        assert {t.trigger_rating for t in fires_at_a} == {"A"}

        fires_at_bbb = await triggers.list_triggered_at_rank(rank("BBB"))
        assert {t.trigger_rating for t in fires_at_bbb} == {"A", "BBB"}


class TestCallScheduleRepository:
    async def test_list_between_filters_by_date(self, db_session: AsyncSession) -> None:
        instruments = InstrumentRepository(db_session)
        calls = CallScheduleRepository(db_session)
        inst = await instruments.add(make_instrument())

        for call_date in (dt.date(2026, 12, 1), dt.date(2027, 6, 15), dt.date(2030, 1, 1)):
            await calls.add(CallSchedule(instrument_id=inst.id, call_date=call_date))

        window = await calls.list_between(dt.date(2026, 1, 1), dt.date(2028, 1, 1))
        assert [c.call_date for c in window] == [dt.date(2026, 12, 1), dt.date(2027, 6, 15)]


# ---------------------------------------------------------------- portfolio


class TestPortfolioRepository:
    async def test_holdings_are_a_time_series_not_a_snapshot(
        self, db_session: AsyncSession
    ) -> None:
        portfolios = PortfolioRepository(db_session)
        instruments = InstrumentRepository(db_session)
        holdings = PortfolioHoldingRepository(db_session)

        pf = await portfolios.add(Portfolio(name="Test Fund"))
        inst = await instruments.add(make_instrument())

        for as_of, value in [
            (dt.date(2025, 12, 1), Decimal("10")),
            (dt.date(2026, 1, 1), Decimal("25")),
        ]:
            await holdings.add(
                PortfolioHolding(
                    portfolio_id=pf.id,
                    instrument_id=inst.id,
                    market_value=value,
                    as_of_date=as_of,
                )
            )

        assert await holdings.latest_as_of_date(pf.id) == dt.date(2026, 1, 1)
        latest = await holdings.list_holdings(pf.id)
        assert len(latest) == 1
        assert latest[0].market_value == Decimal("25.0000")

    async def test_one_holding_per_instrument_per_date(self, db_session: AsyncSession) -> None:
        portfolios = PortfolioRepository(db_session)
        instruments = InstrumentRepository(db_session)
        holdings = PortfolioHoldingRepository(db_session)
        pf = await portfolios.add(Portfolio(name="Dup Fund"))
        inst = await instruments.add(make_instrument())

        await holdings.add(
            PortfolioHolding(
                portfolio_id=pf.id, instrument_id=inst.id, as_of_date=dt.date(2026, 1, 1)
            )
        )
        with pytest.raises(IntegrityError):
            await holdings.add(
                PortfolioHolding(
                    portfolio_id=pf.id, instrument_id=inst.id, as_of_date=dt.date(2026, 1, 1)
                )
            )

    async def test_nav_weight_must_be_a_fraction(self, db_session: AsyncSession) -> None:
        """Weights are fractions in [0,1], not percentages -- 12 would be a bug."""
        portfolios = PortfolioRepository(db_session)
        instruments = InstrumentRepository(db_session)
        holdings = PortfolioHoldingRepository(db_session)
        pf = await portfolios.add(Portfolio(name="Weight Fund"))
        inst = await instruments.add(make_instrument())

        with pytest.raises(IntegrityError):
            await holdings.add(
                PortfolioHolding(
                    portfolio_id=pf.id,
                    instrument_id=inst.id,
                    as_of_date=dt.date(2026, 1, 1),
                    nav_weight=Decimal("12"),
                )
            )

    async def test_exposure_join_by_rating_rank(self, db_session: AsyncSession) -> None:
        portfolios = PortfolioRepository(db_session)
        instruments = InstrumentRepository(db_session)
        holdings = PortfolioHoldingRepository(db_session)

        pf = await portfolios.add(Portfolio(name="Exposure Fund"))
        good = await instruments.add(make_instrument("AAA sukuk"))
        bad = await instruments.add(make_instrument("BBB sukuk"))
        await instruments.set_rating(good, "AAA")
        await instruments.set_rating(bad, "BBB")

        for inst in (good, bad):
            await holdings.add(
                PortfolioHolding(
                    portfolio_id=pf.id,
                    instrument_id=inst.id,
                    market_value=Decimal("100"),
                    as_of_date=dt.date(2026, 1, 1),
                )
            )

        rows = await holdings.list_holdings_rated_at_or_below(rank("A"), portfolio_id=pf.id)
        assert len(rows) == 1
        assert rows[0][1].instrument_name == "BBB sukuk"


# ---------------------------------------------------------------- ops


class TestExtractionJobRepository:
    async def test_extraction_identity_is_unique(self, db_session: AsyncSession) -> None:
        """CLAUDE.md 1.7: idempotency is a database guarantee, not a convention."""
        docs = DocumentRepository(db_session)
        jobs = ExtractionJobRepository(db_session)
        doc = await docs.add(make_document(sha="e" * 64))

        identity: dict[str, object] = {
            "document_sha256": "e" * 64,
            "job_type": JobType.EXTRACT_COVENANT,
            "prompt_version": "v1",
            "model_id": "claude-opus-5",
            "extractor_version": "v1",
        }
        await jobs.add(ExtractionJob(document_id=doc.id, **identity))
        with pytest.raises(IntegrityError):
            await jobs.add(ExtractionJob(document_id=doc.id, **identity))

    async def test_find_by_identity_enables_skipping_completed_work(
        self, db_session: AsyncSession
    ) -> None:
        docs = DocumentRepository(db_session)
        jobs = ExtractionJobRepository(db_session)
        doc = await docs.add(make_document(sha="f" * 64))
        await jobs.add(
            ExtractionJob(
                document_id=doc.id,
                document_sha256="f" * 64,
                job_type=JobType.EXTRACT_COVENANT,
                prompt_version="v1",
                model_id="claude-opus-5",
                extractor_version="v1",
            )
        )

        found = await jobs.find_by_identity(
            document_sha256="f" * 64,
            job_type=JobType.EXTRACT_COVENANT,
            prompt_version="v1",
            model_id="claude-opus-5",
            extractor_version="v1",
        )
        assert found is not None

        # A new prompt version is different work and must not be skipped.
        assert (
            await jobs.find_by_identity(
                document_sha256="f" * 64,
                job_type=JobType.EXTRACT_COVENANT,
                prompt_version="v2",
                model_id="claude-opus-5",
                extractor_version="v1",
            )
            is None
        )


class TestLLMCallRepository:
    async def test_cost_aggregation_by_document_and_stage(self, db_session: AsyncSession) -> None:
        docs = DocumentRepository(db_session)
        calls = LLMCallRepository(db_session)
        doc = await docs.add(make_document())

        await calls.add(
            LLMCall(
                document_id=doc.id,
                stage=LLMStage.EXTRACT,
                provider="anthropic",
                model_id="claude-opus-5",
                estimated_cost_usd=Decimal("0.350000"),
            )
        )
        await calls.add(
            LLMCall(
                document_id=doc.id,
                stage=LLMStage.EMBED,
                provider="qwen",
                model_id="text-embedding-v4",
                estimated_cost_usd=Decimal("0.001500"),
            )
        )

        assert await calls.total_cost_for_document(doc.id) == Decimal("0.351500")
        by_stage = await calls.cost_by_stage()
        assert by_stage[LLMStage.EXTRACT.value] == Decimal("0.350000")
        assert by_stage[LLMStage.EMBED.value] == Decimal("0.001500")

    async def test_a_cache_hit_must_be_free(self, db_session: AsyncSession) -> None:
        """PLAN.md 2: a cached response that still bills is a broken cache."""
        calls = LLMCallRepository(db_session)
        with pytest.raises(IntegrityError):
            await calls.add(
                LLMCall(
                    stage=LLMStage.EXTRACT,
                    provider="anthropic",
                    model_id="claude-opus-5",
                    cache_hit=True,
                    estimated_cost_usd=Decimal("0.10"),
                )
            )


class TestHumanReviewRepository:
    async def test_resolved_review_must_name_a_reviewer(self, db_session: AsyncSession) -> None:
        from app.domain.ids import uuid7

        repo = HumanReviewRepository(db_session)
        with pytest.raises(IntegrityError):
            await repo.add(
                HumanReview(
                    entity_type="covenant",
                    entity_id=uuid7(),
                    field_name="threshold_amount",
                    trigger_reason=ReviewTrigger.LOW_CONFIDENCE,
                    status=ReviewStatus.APPROVED,
                    reviewer_id=None,
                )
            )

    async def test_correction_preserves_the_original_value(self, db_session: AsyncSession) -> None:
        """Phase 7 acceptance: an audit must see what the machine first said."""
        from app.domain.ids import uuid7

        repo = HumanReviewRepository(db_session)
        review = await repo.add(
            HumanReview(
                entity_type="covenant",
                entity_id=uuid7(),
                field_name="threshold_amount",
                old_value="30000000",
                trigger_reason=ReviewTrigger.RULE_LLM_DISAGREEMENT,
                confidence=0.62,
            )
        )
        assert await repo.count_pending() == 1

        await repo.update(
            review,
            new_value="35000000",
            status=ReviewStatus.CORRECTED,
            reviewer_id="analyst@example.com",
            reviewed_at=dt.datetime.now(dt.UTC),
        )
        assert review.old_value == "30000000"
        assert review.new_value == "35000000"
        assert await repo.count_pending() == 0
