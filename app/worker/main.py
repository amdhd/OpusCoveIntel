"""Background worker entrypoint.

PLACEHOLDER (Phase 1). The worker service exists in compose so Phase 3 has a
process to grow into; it currently starts, verifies the database, and idles.

Phase 3 replaces the idle loop with document ingestion job processing. Celery /
Redis stay deferred to Phase 8 (CLAUDE.md 9) -- until job volume justifies a
broker, polling a `status='queued'` column is simpler and fully sufficient.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.db.session import check_database, dispose_engines

logger = get_logger(__name__)

POLL_INTERVAL_SECONDS = 5.0


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
            # Phase 3: claim and process queued extraction jobs here.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=POLL_INTERVAL_SECONDS)
    finally:
        await dispose_engines()
        logger.info("worker stopped")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
