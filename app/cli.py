"""Operator CLI.

Configuration inspection, seeding and ingestion. Extraction, query and eval
commands arrive in Phases 5-8.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import check_database, dispose_engines

app = typer.Typer(help="OpusCovIntel operator CLI", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the application version."""
    from app import __version__

    typer.echo(__version__)


@app.command()
def config() -> None:
    """Print effective settings. Secrets are redacted by SecretStr."""
    settings = get_settings()
    for key, value in settings.model_dump().items():
        typer.echo(f"{key}={value}")


@app.command()
def seed() -> None:
    """Load synthetic demo instruments, portfolios and holdings. Idempotent."""
    from app.db.seed import _main

    settings = get_settings()
    configure_logging(settings)
    counts = asyncio.run(_main())
    for key, value in counts.items():
        typer.echo(f"{key}: {value}")


@app.command()
def ingest(
    path: Path,
    process: bool = typer.Option(True, help="Parse and chunk now instead of only queueing."),
    uploaded_by: str | None = typer.Option(None, help="Recorded on the document row."),
) -> None:
    """Upload a PDF, then parse, score and chunk it. Idempotent by content hash."""
    from app.ingest.service import IngestionService
    from app.ingest.storage import get_object_store

    settings = get_settings()
    configure_logging(settings)

    if not path.is_file():
        typer.echo(f"no such file: {path}")
        raise typer.Exit(code=1)

    async def _run() -> tuple[str, bool, str]:
        from app.db.session import get_sessionmaker

        try:
            async with get_sessionmaker()() as session:
                service = IngestionService(session, get_object_store(), settings)
                outcome = await service.upload(
                    filename=path.name,
                    # Off the event loop: blocking file I/O in an async path.
                    data=await asyncio.to_thread(path.read_bytes),
                    uploaded_by=uploaded_by,
                )
                document_id = outcome.document.id
                if not process:
                    return str(document_id), outcome.duplicate, "queued"
                result = await service.process(document_id)
                summary = (
                    f"{result.page_count} pages, {result.chunk_count} chunks, "
                    f"{result.pages_flagged_for_vlm} flagged for VLM"
                    + (" (skipped; already ingested)" if result.skipped else "")
                )
                return str(document_id), outcome.duplicate, summary
        finally:
            await dispose_engines()

    document_id, duplicate, summary = asyncio.run(_run())
    typer.echo(f"document: {document_id}")
    typer.echo(f"duplicate: {duplicate}")
    typer.echo(summary)


@app.command()
def check() -> None:
    """Verify external dependencies are reachable. Exit 1 if any check fails."""
    settings = get_settings()
    configure_logging(settings)

    async def _run() -> bool:
        try:
            return await check_database()
        finally:
            await dispose_engines()

    db_ok = asyncio.run(_run())
    typer.echo(f"database: {'ok' if db_ok else 'FAIL'}")
    if not db_ok:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
