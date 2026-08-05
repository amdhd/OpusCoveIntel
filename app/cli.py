"""Operator CLI.

Configuration, seeding, ingestion, indexing, rule extraction and the
deterministic query path. Everything here runs at **$0** -- no command in this
file can reach a paid provider, because none exists until Phase 5.
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
def index(document_id: str | None = typer.Argument(None, help="Default: every document.")) -> None:
    """Embed and full-text index chunks so they are retrievable. Idempotent, $0."""
    import uuid as _uuid

    from app.db.repositories.documents import DocumentRepository
    from app.db.session import get_sessionmaker
    from app.retrieval.indexing import IndexingService

    settings = get_settings()
    configure_logging(settings)

    async def _run() -> list[str]:
        lines: list[str] = []
        try:
            async with get_sessionmaker()() as session:
                service = IndexingService(session)
                targets = (
                    [_uuid.UUID(document_id)]
                    if document_id
                    else [row.id for row in await DocumentRepository(session).list(limit=500)]
                )
                for target in targets:
                    outcome = await service.index_document(target)
                    lines.append(
                        f"{target}: {outcome.chunks_embedded} embedded, "
                        f"{outcome.chunks_indexed_for_fts} indexed"
                        + (" (skipped)" if outcome.skipped else "")
                    )
        finally:
            await dispose_engines()
        return lines

    for line in asyncio.run(_run()):
        typer.echo(line)


@app.command("extract-rules")
def extract_rules(
    document_id: str | None = typer.Argument(None, help="Default: every document."),
) -> None:
    """Run the deterministic extractor and persist cited clauses and covenants."""
    import uuid as _uuid

    from app.db.repositories.documents import DocumentRepository
    from app.db.session import get_sessionmaker
    from app.extract.service import RuleExtractionService

    settings = get_settings()
    configure_logging(settings)

    async def _run() -> list[str]:
        lines: list[str] = []
        try:
            async with get_sessionmaker()() as session:
                service = RuleExtractionService(session, settings)
                targets = (
                    [_uuid.UUID(document_id)]
                    if document_id
                    else [row.id for row in await DocumentRepository(session).list(limit=500)]
                )
                for target in targets:
                    outcome = await service.extract_document(target)
                    lines.append(
                        f"{target}: {outcome.clauses} clauses, {outcome.covenants} covenants, "
                        f"{outcome.call_schedules} call dates, "
                        f"{outcome.rating_triggers} rating triggers, "
                        f"{outcome.queued_for_review} queued for review"
                        + (" (skipped)" if outcome.skipped else "")
                    )
        finally:
            await dispose_engines()
        return lines

    for line in asyncio.run(_run()):
        typer.echo(line)


@app.command()
def query(question: str) -> None:
    """Answer a question over the deterministic path. No LLM, no spend."""
    from app.db.session import get_sessionmaker
    from app.query.service import Answer, DeterministicQueryService

    settings = get_settings()
    configure_logging(settings)

    async def _run() -> Answer:
        try:
            async with get_sessionmaker()() as session:
                return await DeterministicQueryService(session).answer(question)
        finally:
            await dispose_engines()

    answer = asyncio.run(_run())
    typer.echo(f"intent:     {answer.intent.value}")
    typer.echo(f"confidence: {answer.confidence:.2f}{'  (refused)' if answer.refused else ''}")
    typer.echo("")
    typer.echo(answer.text)
    if answer.citations:
        typer.echo("\nsources:")
        for citation in answer.citations:
            quote = " ".join(citation.quote.split())[:110]
            typer.echo(f"  - page {citation.page_number}: {quote}...")


@app.command("golden")
def golden() -> None:
    """Run the golden question set over the deterministic path.

    PLAN.md, Phase 4 acceptance: at least 6 of 10 answered with zero LLM calls.
    """
    from app.db.session import get_sessionmaker
    from app.evals.golden import GOLDEN_QUESTIONS, PHASE_4_TARGET
    from app.query.service import DeterministicQueryService

    settings = get_settings()
    configure_logging(settings)

    async def _run() -> list[tuple[str, bool, str]]:
        results: list[tuple[str, bool, str]] = []
        try:
            async with get_sessionmaker()() as session:
                service = DeterministicQueryService(session)
                for case in GOLDEN_QUESTIONS:
                    answer = await service.answer(case.question)
                    missing = [
                        needle
                        for needle in case.must_contain
                        if needle.lower() not in answer.text.lower()
                    ]
                    ok = (
                        answer.intent is case.expected_intent
                        and answer.refused == case.expect_refusal
                        and not missing
                        and len(answer.citations) >= case.min_citations
                    )
                    detail = f"intent={answer.intent.value} citations={len(answer.citations)}" + (
                        f" missing={missing}" if missing else ""
                    )
                    results.append((case.id, ok, detail))
        finally:
            await dispose_engines()
        return results

    results = asyncio.run(_run())
    for case_id, ok, detail in results:
        typer.echo(f"{case_id} {'PASS' if ok else 'FAIL'}  {detail}")

    passed = sum(1 for _, ok, _ in results if ok)
    typer.echo(f"\n{passed}/{len(results)} passed (Phase 4 target: {PHASE_4_TARGET})")
    if passed < PHASE_4_TARGET:
        raise typer.Exit(code=1)


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
