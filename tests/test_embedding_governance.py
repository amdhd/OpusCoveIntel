"""Exercise configured embeddings through real routing with a free provider.

Repository/session doubles keep these regressions independent of any configured
database. The provider cannot make HTTP requests, including on the old bypass.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.db.models.documents import Document, DocumentChunk
from app.db.models.ops import ExtractionJob, LLMCall
from app.db.repositories.documents import DocumentChunkRepository
from app.db.repositories.ops import ExtractionJobRepository, LLMCallRepository
from app.domain.enums import JobStatus, LLMStage
from app.extract.candidates import CandidateDetectionService
from app.llm.budget import BudgetExceededError, BudgetOutcome
from app.llm.embeddings import get_embedder
from app.llm.mock import MockLLMProvider
from app.retrieval.hybrid import HybridSearcher
from app.retrieval.indexing import IndexingService


@dataclass
class Accounting:
    rows: list[LLMCall] = field(default_factory=list)
    sessions: list[AsyncMock] = field(default_factory=list)
    provider: MockLLMProvider = field(default_factory=MockLLMProvider)
    dispatch: AsyncMock = field(default_factory=AsyncMock)
    close: AsyncMock = field(default_factory=AsyncMock)


@pytest.fixture
def accounting(monkeypatch: pytest.MonkeyPatch) -> Accounting:
    # Exercise production selection with a dummy key, never ambient credentials.
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("QWEN_API_KEY", "dummy-offline-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v4")
    monkeypatch.setenv("MAX_COST_PER_CALL_USD", "1")
    monkeypatch.setenv("MAX_COST_PER_DOCUMENT_USD", "1")
    monkeypatch.setenv("MAX_TOTAL_COST_USD", "1")
    state = Accounting()
    state.dispatch.side_effect = state.provider.embed
    monkeypatch.setattr(state.provider, "embed", state.dispatch)
    monkeypatch.setattr(state.provider, "close", state.close, raising=False)
    monkeypatch.setattr("app.llm.adapters.qwen.QwenAdapter", lambda: state.provider)

    pending: dict[int, list[LLMCall]] = {}

    def new_session() -> AsyncMock:
        session = AsyncMock(spec=AsyncSession)
        session.__aenter__.return_value = session
        pending[id(session)] = []

        async def commit() -> None:
            state.rows.extend(pending[id(session)])
            pending[id(session)].clear()

        session.commit.side_effect = commit
        state.sessions.append(session)
        return session

    monkeypatch.setattr("app.db.session.get_sessionmaker", lambda: new_session)

    async def total_cost(repo: LLMCallRepository) -> Decimal:
        return sum((row.estimated_cost_usd for row in state.rows), Decimal("0"))

    async def document_cost(repo: LLMCallRepository, document_id: uuid.UUID) -> Decimal:
        return sum(
            (row.estimated_cost_usd for row in state.rows if row.document_id == document_id),
            Decimal("0"),
        )

    async def add(repo: LLMCallRepository, row: LLMCall) -> LLMCall:
        row.id = uuid.uuid4()
        pending[id(repo.session)].append(row)
        return row

    monkeypatch.setattr(LLMCallRepository, "total_cost", total_cost)
    monkeypatch.setattr(LLMCallRepository, "total_cost_for_document", document_cost)
    monkeypatch.setattr(LLMCallRepository, "add", add)
    monkeypatch.setattr(
        ExtractionJobRepository, "total_cost_for_document", AsyncMock(return_value=Decimal("0"))
    )
    return state


def existing_spend(amount: str, document_id: uuid.UUID | None = None) -> LLMCall:
    return LLMCall(
        stage=LLMStage.EMBED,
        provider="qwen",
        model_id="text-embedding-v4",
        document_id=document_id,
        estimated_cost_usd=Decimal(amount),
    )


@pytest.mark.parametrize(
    ("limit", "outcome"),
    [
        ("MAX_COST_PER_CALL_USD", BudgetOutcome.REJECTED_PER_CALL),
        ("MAX_TOTAL_COST_USD", BudgetOutcome.REJECTED_GLOBAL),
    ],
)
async def test_configured_embedder_denies_before_provider_dispatch(
    accounting: Accounting, monkeypatch: pytest.MonkeyPatch, limit: str, outcome: BudgetOutcome
) -> None:
    monkeypatch.setenv(limit, "0.00000001")

    with pytest.raises(BudgetExceededError) as error:
        await get_embedder().embed(["synthetic covenant question"])

    assert error.value.decision.outcome == outcome
    accounting.dispatch.assert_not_awaited()
    accounting.close.assert_not_awaited()  # Rejection precedes adapter construction.
    accounting.sessions[0].commit.assert_not_awaited()
    assert accounting.rows == []


async def test_search_commits_spend_without_writing_through_readonly_session(
    accounting: Accounting, monkeypatch: pytest.MonkeyPatch
) -> None:
    readonly = AsyncMock(spec=AsyncSession)
    chunk = DocumentChunk(chunk_text="synthetic covenant clause")
    monkeypatch.setattr(
        DocumentChunkRepository, "search_by_vector", AsyncMock(return_value=[(chunk, 0.9)])
    )

    hits = await HybridSearcher(readonly).search_vector("synthetic covenant question")
    await readonly.rollback()

    assert hits == [(chunk, 0.9)]
    accounting.dispatch.assert_awaited_once()
    assert len(accounting.rows) == 1
    row = accounting.rows[0]
    assert row.stage is LLMStage.EMBED
    assert row.provider == "qwen"
    assert row.model_id == "text-embedding-v4"
    assert row.document_id is None
    assert row.prompt_tokens == 6
    # The six-decimal ledger must not round a small query's spend to zero.
    assert row.estimated_cost_usd == Decimal("0.000001")
    accounting.sessions[0].commit.assert_awaited_once()
    accounting.close.assert_awaited_once()
    readonly.add.assert_not_called()
    readonly.execute.assert_not_awaited()
    readonly.commit.assert_not_awaited()


def indexing_service(document_id: uuid.UUID) -> IndexingService:
    service = IndexingService(AsyncMock(spec=AsyncSession))
    service._documents = MagicMock()
    service._documents.get = AsyncMock(return_value=Document(id=document_id, sha256="a" * 64))
    service._jobs = MagicMock()
    service._jobs.find_by_identity = AsyncMock(return_value=ExtractionJob(status=JobStatus.QUEUED))
    service._chunks = MagicMock()
    service._chunks.refresh_fts = AsyncMock(return_value=1)
    service._chunks.list_for_document = AsyncMock(
        return_value=[DocumentChunk(chunk_text="synthetic covenant clause")]
    )
    return service


async def test_indexing_enforces_document_budget(
    accounting: Accounting, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id = uuid.uuid4()
    monkeypatch.setenv("MAX_TOTAL_COST_USD", "2")
    accounting.rows.append(existing_spend("1", document_id))

    with pytest.raises(BudgetExceededError) as error:
        await indexing_service(document_id).index_document(document_id)

    assert error.value.decision.outcome == BudgetOutcome.REJECTED_PER_DOCUMENT
    accounting.dispatch.assert_not_awaited()
    assert len(accounting.rows) == 1


async def test_candidate_embeddings_use_the_document_budget(
    accounting: Accounting, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id = uuid.uuid4()
    monkeypatch.setenv("MAX_TOTAL_COST_USD", "2")
    accounting.rows.append(existing_spend("1", document_id))
    monkeypatch.setattr(DocumentChunkRepository, "search_by_fts", AsyncMock(return_value=[]))
    vector_search = AsyncMock(return_value=[])
    monkeypatch.setattr(DocumentChunkRepository, "search_by_vector", vector_search)

    results = await CandidateDetectionService(AsyncMock(spec=AsyncSession))._search_both_legs(
        document_id, "synthetic covenant exemplar"
    )

    # Candidate detection degrades to the other legs, without dispatching spend.
    assert results == []
    accounting.dispatch.assert_not_awaited()
    vector_search.assert_not_awaited()
    assert len(accounting.rows) == 1


async def test_provider_failure_closes_adapter_without_committing_spend(
    accounting: Accounting,
) -> None:
    accounting.dispatch.side_effect = RuntimeError("synthetic provider failure")

    with pytest.raises(RuntimeError, match="synthetic provider failure"):
        await get_embedder().embed(["synthetic covenant question"])

    accounting.close.assert_awaited_once()
    accounting.sessions[0].commit.assert_not_awaited()
    accounting.sessions[0].__aexit__.assert_awaited_once()
    assert accounting.rows == []


async def test_indexing_spend_survives_later_index_rollback(accounting: Accounting) -> None:
    document_id = uuid.uuid4()
    service = indexing_service(document_id)
    session = cast(AsyncMock, service._session)
    session.flush.side_effect = [None, RuntimeError("synthetic indexing failure")]

    with pytest.raises(RuntimeError, match="synthetic indexing failure"):
        await service.index_document(document_id)

    session.rollback.assert_awaited_once()
    assert len(accounting.rows) == 1
    assert accounting.rows[0].document_id == document_id
    assert accounting.rows[0].estimated_cost_usd == Decimal("0.000001")
    accounting.sessions[0].commit.assert_awaited_once()


async def test_committed_embedding_spend_blocks_the_next_call(
    accounting: Accounting, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_TOTAL_COST_USD", "0.0000015")
    embedder = get_embedder()
    await embedder.embed(["synthetic covenant question"])

    with pytest.raises(BudgetExceededError) as error:
        await embedder.embed(["synthetic covenant question"])

    assert error.value.decision.outcome == BudgetOutcome.REJECTED_GLOBAL
    accounting.dispatch.assert_awaited_once()
    assert len(accounting.rows) == 1


async def test_embedding_ledger_is_visible_from_another_database_session(
    db_engine: AsyncEngine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI's dedicated Postgres verifies the commit and stored numeric precision."""
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("QWEN_API_KEY", "dummy-offline-key")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v4")
    monkeypatch.setenv("MAX_COST_PER_CALL_USD", "1")
    monkeypatch.setenv("MAX_COST_PER_DOCUMENT_USD", "1")
    monkeypatch.setenv("MAX_TOTAL_COST_USD", "1000")
    get_settings.cache_clear()  # db_engine resolved test settings before this body.
    provider = MockLLMProvider()
    close = AsyncMock()
    monkeypatch.setattr(provider, "close", close, raising=False)
    monkeypatch.setattr("app.llm.adapters.qwen.QwenAdapter", lambda: provider)
    factory = async_sessionmaker(db_engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.session.get_sessionmaker", lambda: factory)
    document_id = uuid.uuid4()

    try:
        async with factory() as setup:
            setup.add(
                Document(id=document_id, sha256=uuid.uuid4().hex * 2, filename="synthetic.pdf")
            )
            await setup.commit()

        await get_embedder(document_id=document_id).embed(["synthetic covenant question"])

        async with factory() as observer:
            row = (
                await observer.execute(select(LLMCall).where(LLMCall.document_id == document_id))
            ).scalar_one()
            assert row.stage is LLMStage.EMBED
            assert row.estimated_cost_usd == Decimal("0.000001")
            assert row.prompt_tokens == 6
        close.assert_awaited_once()
    finally:
        async with factory() as cleanup:
            await cleanup.execute(delete(LLMCall).where(LLMCall.document_id == document_id))
            await cleanup.execute(delete(Document).where(Document.id == document_id))
            await cleanup.commit()
