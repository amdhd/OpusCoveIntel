"""Background worker: claims queued parse jobs and ingests documents.

No broker. `extraction_jobs.status = 'queued'` *is* the queue, claimed with
`FOR UPDATE ... SKIP LOCKED` so several workers can poll the same table without
colliding. Celery and Redis stay deferred to Phase 8 (CLAUDE.md 9) -- until job
volume justifies the operational surface, a polled column is simpler and loses
nothing that matters here.

Failure handling is deliberately shallow: the ingestion service records the
failure on the job and the document, and the worker moves on. A failed job is
not retried automatically, because the common causes -- a corrupt PDF, a scan
over the VLM page cap -- are not transient, and a retry loop would hide them.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.repositories.ops import ExtractionJobRepository
from app.db.session import check_database, dispose_engines, get_sessionmaker
from app.domain.enums import JobType
from app.ingest.service import ingest_and_index
from app.ingest.storage import get_object_store

logger = get_logger(__name__)

POLL_INTERVAL_SECONDS = 5.0
# Bound one pass so a large backlog cannot starve shutdown.
MAX_JOBS_PER_PASS = 20


async def process_queued_documents(*, limit: int = MAX_JOBS_PER_PASS) -> int:
    """Claim and ingest up to `limit` queued documents. Returns how many ran."""
    settings = get_settings()
    store = get_object_store()
    processed = 0

    async with get_sessionmaker()() as session:
        for _ in range(limit):
            job = await ExtractionJobRepository(session).claim_next(JobType.PARSE)
            if job is None:
                break
            document_id = job.document_id
            # Commit the claim before the long work, so a second worker sees
            # the job as taken rather than blocking on our row lock.
            await session.commit()

            try:
                # Indexed here too: a parsed document that nothing indexed is
                # invisible to every question asked about it, and the step that
                # used to do it was a CLI command nobody ran.
                await ingest_and_index(session, store, document_id, settings)
            except Exception:
                logger.exception(
                    "document ingestion failed", extra={"document_id": str(document_id)}
                )
                continue
            processed += 1

    return processed


async def run() -> None:
    settings = get_settings()
    configure_logging(settings)
    logger.info("worker starting", extra={"environment": settings.ENVIRONMENT})

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # not available on Windows
            loop.add_signal_handler(sig, stop.set)

    if not await check_database():
        logger.warning("database unavailable at startup; will retry while polling")

    try:
        while not stop.is_set():
            try:
                processed = await process_queued_documents()
            except Exception:
                logger.exception("worker pass failed")
                processed = 0

            # Only idle when there was nothing to do; a backlog drains at speed.
            if processed == 0:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=POLL_INTERVAL_SECONDS)
    finally:
        await dispose_engines()
        logger.info("worker stopped")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
