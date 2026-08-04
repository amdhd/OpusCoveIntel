"""The ingestion service: dedup, job identity, spans, and loud failure.

These run against a real Postgres (see `tests/conftest.py`) because the things
being asserted -- unique constraints, the `vlm_used`/`vlm_reason` CHECK, cascade
behaviour -- do not exist in a fake.
"""

from __future__ import annotations

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.models.documents import Document
from app.db.models.ops import ExtractionJob
from app.db.repositories.ops import ExtractionJobRepository
from app.domain.enums import ChunkType, DocumentStatus, JobStatus, JobType, Language
from app.ingest.service import (
    IngestionError,
    IngestionService,
    PayloadTooLargeError,
    UnsupportedMediaTypeError,
    UploadOutcome,
    VlmBudgetExceededError,
    storage_key,
)
from app.ingest.storage import LocalFileStore
from tests.fixtures.synthetic_pdf import (
    build_mixed_document,
    build_prospectus,
    build_scanned_document,
)

pytestmark = pytest.mark.usefixtures("storage_root")


async def upload(
    service: IngestionService, data: bytes | None = None, name: str = "im.pdf"
) -> UploadOutcome:
    payload = data if data is not None else build_prospectus()
    return await service.upload(filename=name, data=payload)


async def jobs_for(session: AsyncSession, sha256: str) -> dict[JobType, ExtractionJob]:
    result = await session.execute(
        select(ExtractionJob).where(ExtractionJob.document_sha256 == sha256)
    )
    return {job.job_type: job for job in result.scalars().all()}


# -- upload ----------------------------------------------------------------


async def test_upload_stores_the_bytes_and_records_the_document(
    ingestion_service: IngestionService, object_store: LocalFileStore
) -> None:
    data = build_prospectus()

    outcome = await upload(ingestion_service, data)

    assert outcome.duplicate is False
    assert outcome.document.status is DocumentStatus.UPLOADED
    assert outcome.document.byte_size == len(data)
    assert await object_store.get(storage_key(outcome.document.sha256)) == data


async def test_upload_queues_a_parse_job(
    ingestion_service: IngestionService, db_session: AsyncSession
) -> None:
    outcome = await upload(ingestion_service)

    jobs = await jobs_for(db_session, outcome.document.sha256)
    assert jobs[JobType.PARSE].status is JobStatus.QUEUED
    assert jobs[JobType.PARSE].model_id == "none"  # Phase 3 spends nothing


async def test_the_same_bytes_under_another_filename_are_one_document(
    ingestion_service: IngestionService, db_session: AsyncSession
) -> None:
    data = build_prospectus()

    first = await upload(ingestion_service, data, name="original.pdf")
    second = await upload(ingestion_service, data, name="forwarded-copy.pdf")

    assert second.duplicate is True
    assert second.document.id == first.document.id
    # Dedup is by content: the second filename never made it to a second row.
    assert second.document.filename == "original.pdf"
    rows = await db_session.execute(
        select(Document.id).where(Document.sha256 == first.document.sha256)
    )
    assert len(rows.scalars().all()) == 1


async def test_a_client_supplied_path_is_reduced_to_its_leaf(
    ingestion_service: IngestionService,
) -> None:
    outcome = await upload(ingestion_service, name="../../etc/passwd.pdf")

    assert outcome.document.filename == "passwd.pdf"


async def test_non_pdf_bytes_are_rejected(ingestion_service: IngestionService) -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        await upload(ingestion_service, b"PK\x03\x04 this is a zip")


async def test_an_oversized_upload_is_rejected(
    db_session: AsyncSession, object_store: LocalFileStore
) -> None:
    service = IngestionService(db_session, object_store, Settings(MAX_UPLOAD_SIZE_MB=0))

    with pytest.raises(PayloadTooLargeError):
        await service.upload(filename="im.pdf", data=build_prospectus())


# -- processing ------------------------------------------------------------


async def test_processing_parses_scores_and_chunks(
    ingestion_service: IngestionService,
) -> None:
    outcome = await upload(ingestion_service)

    result = await ingestion_service.process(outcome.document.id)

    assert result.skipped is False
    assert result.page_count == 4
    assert result.chunk_count > 0
    assert result.pages_flagged_for_vlm == 0
    assert result.status is DocumentStatus.CHUNKED


async def test_the_document_row_records_the_parse_outcome(
    ingestion_service: IngestionService,
) -> None:
    outcome = await upload(ingestion_service)
    await ingestion_service.process(outcome.document.id)

    document = await ingestion_service.get_document(outcome.document.id)
    assert document is not None
    assert document.status is DocumentStatus.CHUNKED
    assert document.page_count == 4
    assert document.parse_confidence == 1.0
    assert document.language is Language.MIXED


async def test_every_page_gets_a_telemetry_row(ingestion_service: IngestionService) -> None:
    outcome = await upload(ingestion_service)
    await ingestion_service.process(outcome.document.id)

    pages = await ingestion_service.list_pages(outcome.document.id)

    assert [page.page_number for page in pages] == [1, 2, 3, 4]
    assert all(page.has_text_layer for page in pages)
    assert all(page.confidence == 1.0 for page in pages)
    assert all(page.vlm_reason is None for page in pages)


async def test_every_chunk_carries_a_span_that_reproduces_its_text(
    ingestion_service: IngestionService,
) -> None:
    from app.ingest.pdf import parse_pdf

    data = build_prospectus()
    outcome = await upload(ingestion_service, data)
    await ingestion_service.process(outcome.document.id)

    parsed = parse_pdf(data, max_pages=get_settings().MAX_PDF_PAGES)
    pages = {page.page_number: page.text for page in parsed.pages}

    chunks = await ingestion_service.list_chunks(outcome.document.id)
    assert chunks
    for chunk in chunks:
        # CLAUDE.md 1.2: the span is the citation. It has to be real.
        assert pages[chunk.page_number][chunk.char_start : chunk.char_end] == chunk.chunk_text


async def test_the_table_page_produces_a_table_chunk(
    ingestion_service: IngestionService,
) -> None:
    outcome = await upload(ingestion_service)
    await ingestion_service.process(outcome.document.id)

    chunks = await ingestion_service.list_chunks(outcome.document.id)
    tables = [chunk for chunk in chunks if chunk.chunk_type is ChunkType.TABLE]

    assert len(tables) == 1
    assert "2028-06-15" in tables[0].chunk_text


async def test_malay_chunks_are_indexed_under_the_simple_configuration(
    ingestion_service: IngestionService,
) -> None:
    outcome = await upload(ingestion_service)
    await ingestion_service.process(outcome.document.id)

    chunks = await ingestion_service.list_chunks(outcome.document.id)
    malay = [chunk for chunk in chunks if chunk.language is Language.MS]

    assert malay
    assert all(chunk.fts_config == "simple" for chunk in malay)


# -- idempotency -----------------------------------------------------------


async def test_reprocessing_an_unchanged_document_is_a_no_op(
    ingestion_service: IngestionService,
) -> None:
    outcome = await upload(ingestion_service)
    first = await ingestion_service.process(outcome.document.id)

    second = await ingestion_service.process(outcome.document.id)

    # CLAUDE.md 1.7: the extraction identity already succeeded, so nothing runs.
    assert second.skipped is True
    assert second.chunk_count == first.chunk_count
    assert second.page_count == first.page_count


async def test_reprocessing_does_not_duplicate_pages_or_chunks(
    ingestion_service: IngestionService, db_session: AsyncSession
) -> None:
    outcome = await upload(ingestion_service)
    await ingestion_service.process(outcome.document.id)
    before = len(await ingestion_service.list_chunks(outcome.document.id))

    # Force the work to run again by clearing the job status the guard reads.
    jobs = await jobs_for(db_session, outcome.document.sha256)
    for job in jobs.values():
        job.status = JobStatus.QUEUED
    await db_session.commit()

    await ingestion_service.process(outcome.document.id)

    assert len(await ingestion_service.list_chunks(outcome.document.id)) == before
    assert len(await ingestion_service.list_pages(outcome.document.id)) == 4


async def test_both_stage_jobs_end_up_successful(
    ingestion_service: IngestionService, db_session: AsyncSession
) -> None:
    outcome = await upload(ingestion_service)
    await ingestion_service.process(outcome.document.id)

    jobs = await jobs_for(db_session, outcome.document.sha256)

    assert {JobType.PARSE, JobType.CHUNK} <= set(jobs)
    assert all(job.status is JobStatus.SUCCEEDED for job in jobs.values())
    assert all(job.finished_at is not None for job in jobs.values())
    assert all(job.estimated_cost_usd == 0 for job in jobs.values())  # Phase 3 is $0


# -- low-confidence pages --------------------------------------------------


async def test_a_scanned_page_is_persisted_with_its_reason_but_not_sent_anywhere(
    ingestion_service: IngestionService,
) -> None:
    outcome = await upload(ingestion_service, build_mixed_document())

    result = await ingestion_service.process(outcome.document.id)

    assert result.pages_flagged_for_vlm == 1
    pages = await ingestion_service.list_pages(outcome.document.id)
    flagged = pages[1]
    assert flagged.confidence == 0.0
    assert "no_text_layer" in (flagged.vlm_reason or "")
    # Phase 3 detects only. Nothing has been sent to a paid model.
    assert flagged.vlm_used is False


async def test_a_document_over_the_vlm_page_cap_fails_loudly(
    db_session: AsyncSession, object_store: LocalFileStore
) -> None:
    service = IngestionService(db_session, object_store, Settings(MAX_VLM_PAGES_PER_DOC=0))
    outcome = await service.upload(filename="scan.pdf", data=build_scanned_document())

    with pytest.raises(VlmBudgetExceededError):
        await service.process(outcome.document.id)

    document = await service.get_document(outcome.document.id)
    assert document is not None
    assert document.status is DocumentStatus.FAILED
    # The telemetry survives the failure, so an operator can see which pages
    # caused it rather than re-running the parser to find out.
    pages = await service.list_pages(outcome.document.id)
    assert len(pages) == 1
    assert pages[0].vlm_reason


async def test_a_failure_marks_the_document_and_its_jobs(
    ingestion_service: IngestionService,
    db_session: AsyncSession,
    object_store: LocalFileStore,
) -> None:
    outcome = await upload(ingestion_service)
    # Corrupt the stored object behind the row's back.
    await object_store.put(storage_key(outcome.document.sha256), b"%PDF-1.7 shredded")

    with pytest.raises(IngestionError):
        await ingestion_service.process(outcome.document.id)

    document = await ingestion_service.get_document(outcome.document.id)
    assert document is not None
    assert document.status is DocumentStatus.FAILED
    jobs = await jobs_for(db_session, outcome.document.sha256)
    assert jobs[JobType.PARSE].status is JobStatus.FAILED
    assert jobs[JobType.PARSE].error_message


async def test_processing_an_unknown_document_raises(
    ingestion_service: IngestionService,
) -> None:
    import uuid

    with pytest.raises(LookupError):
        await ingestion_service.process(uuid.uuid4())


# -- worker claim ----------------------------------------------------------


async def test_a_queued_job_can_be_claimed_exactly_once(
    ingestion_service: IngestionService, db_session: AsyncSession
) -> None:
    await upload(ingestion_service)
    repository = ExtractionJobRepository(db_session)

    claimed = await repository.claim_next(JobType.PARSE)
    assert claimed is not None
    assert claimed.status is JobStatus.RUNNING
    assert claimed.started_at is not None

    # Already running, so the next poll finds nothing to do.
    assert await repository.claim_next(JobType.PARSE) is None
