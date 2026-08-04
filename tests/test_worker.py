"""The worker's claim-and-ingest pass.

There is no broker: `extraction_jobs.status = 'queued'` is the queue (CLAUDE.md
9 defers Celery/Redis to Phase 8). What is worth testing is that a queued row
actually turns into a chunked document, and that one bad document does not stop
the pass.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.enums import DocumentStatus
from app.ingest.service import IngestionService, storage_key
from app.ingest.storage import LocalFileStore
from app.worker import main as worker
from tests.fixtures.synthetic_pdf import build_prospectus

pytestmark = pytest.mark.usefixtures("storage_root")


class _FakeSessionFactory:
    """Hands the worker the test's session instead of opening its own.

    The worker closes the session it opens; the test's must outlive the pass so
    its transaction can still be rolled back.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def __call__(self) -> _FakeSessionFactory:
        return self

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


@pytest.fixture(autouse=True)
def _wire_worker(
    monkeypatch: pytest.MonkeyPatch, db_session: AsyncSession, object_store: LocalFileStore
) -> None:
    monkeypatch.setattr(worker, "get_sessionmaker", lambda: _FakeSessionFactory(db_session))
    monkeypatch.setattr(worker, "get_object_store", lambda: object_store)


async def test_a_queued_document_is_claimed_and_ingested(
    ingestion_service: IngestionService,
) -> None:
    outcome = await ingestion_service.upload(filename="im.pdf", data=build_prospectus())

    processed = await worker.process_queued_documents()

    assert processed == 1
    document = await ingestion_service.get_document(outcome.document.id)
    assert document is not None
    assert document.status is DocumentStatus.CHUNKED
    assert len(await ingestion_service.list_chunks(outcome.document.id)) > 0


async def test_an_empty_queue_does_no_work(ingestion_service: IngestionService) -> None:
    assert await worker.process_queued_documents() == 0


async def test_a_claimed_job_is_not_claimed_again(
    ingestion_service: IngestionService,
) -> None:
    await ingestion_service.upload(filename="im.pdf", data=build_prospectus())
    await worker.process_queued_documents()

    # The job is no longer queued, so a second pass finds nothing.
    assert await worker.process_queued_documents() == 0


async def test_a_failing_document_is_recorded_and_the_pass_continues(
    ingestion_service: IngestionService, object_store: LocalFileStore
) -> None:
    outcome = await ingestion_service.upload(filename="im.pdf", data=build_prospectus())
    await object_store.put(storage_key(outcome.document.sha256), b"%PDF-1.7 shredded")

    processed = await worker.process_queued_documents()

    assert processed == 0
    document = await ingestion_service.get_document(outcome.document.id)
    assert document is not None
    assert document.status is DocumentStatus.FAILED


async def test_a_pass_is_bounded_so_a_backlog_cannot_starve_shutdown(
    ingestion_service: IngestionService,
) -> None:
    await ingestion_service.upload(filename="im.pdf", data=build_prospectus())

    assert await worker.process_queued_documents(limit=0) == 0
