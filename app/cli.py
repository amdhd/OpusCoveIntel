"""Operator CLI.

Configuration, seeding, ingestion, indexing, extraction and the deterministic
query path.

**Two commands in this file spend money: `extract` and `ocr`.** Everything else
is $0 and stays that way. Because those two are the exception, they behave
differently from their neighbours on purpose:

* they refuse to run without an explicit target -- `extract-rules` defaults to
  every document, and the same default on a billable command is the "$10
  keystroke" CLAUDE.md warns about;
* `--dry-run` prices the work without dispatching anything;
* they print the ceilings and ask before spending, unless `--yes`.

`ocr` also chunks what the model read, in the same command. Separating the two
is how a transcription came to be stored and never looked at.
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
def extract(
    document_id: str | None = typer.Argument(None, help="The document to extract."),
    all_documents: bool = typer.Option(
        False, "--all", help="Every document. Prices the whole corpus; use with --dry-run first."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Price the work from candidate spans without calling the model."
    ),
    force: bool = typer.Option(
        False, "--force", help="Re-run even if this extraction identity already succeeded."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the spend confirmation."),
    instrument_id: str | None = typer.Option(
        None, "--instrument-id", help="Link covenants to this instrument instead of resolving it."
    ),
) -> None:
    """Run rule + LLM extraction over a document. **This spends money.**

    The LLM half is billable and budget-guarded. `--dry-run` prices it first
    from the candidate spans, which is free: candidate detection is regex over
    text already in the database.
    """
    import uuid as _uuid

    from app.db.repositories.documents import DocumentRepository
    from app.db.session import get_sessionmaker
    from app.extract.pipeline import ExtractionPipeline

    settings = get_settings()
    configure_logging(settings)

    if not document_id and not all_documents:
        # `extract-rules` defaults to every document because it is free. Doing
        # the same here would put the whole corpus through Opus on a bare
        # command, which is exactly the keystroke CLAUDE.md 9 warns about.
        typer.echo("Refusing to guess: pass a document id, or --all to mean it.", err=True)
        raise typer.Exit(code=2)

    if settings.ANTHROPIC_API_KEY is None and not dry_run:
        typer.echo(
            "ANTHROPIC_API_KEY is not set. Set it, or use --dry-run to price the work.", err=True
        )
        raise typer.Exit(code=2)

    # One event loop for the whole command. asyncpg binds a connection to the
    # loop that opened it and the engine pool is process-cached, so a second
    # `asyncio.run` hands the new loop a connection belonging to the old one
    # and fails with "attached to a different loop". The confirmation prompt
    # therefore happens inside the coroutine; blocking the loop on stdin is
    # harmless in a single-user CLI.
    async def _run() -> None:
        from app.extract.dry_run import estimate_document
        from app.llm.router import LLMRouter

        try:
            async with get_sessionmaker()() as session:
                targets = (
                    [_uuid.UUID(document_id)]
                    if document_id
                    else [row.id for row in await DocumentRepository(session).list(limit=500)]
                )
                if not targets:
                    typer.echo("No documents found.")
                    return

                for target in targets:
                    estimate = await estimate_document(session, target, settings=settings)
                    typer.echo(estimate.describe())

                if dry_run:
                    typer.echo("\nDry run: nothing was dispatched and nothing was charged.")
                    return

                typer.echo(
                    f"\nCeilings: ${settings.MAX_COST_PER_CALL_USD}/call · "
                    f"${settings.MAX_COST_PER_DOCUMENT_USD}/document · "
                    f"${settings.MAX_TOTAL_COST_USD} total"
                )
                if not yes and not typer.confirm(
                    f"Send {len(targets)} document(s) to {settings.EXTRACTION_MODEL}?"
                ):
                    typer.echo("Aborted; nothing was charged.")
                    return

                router = LLMRouter(session)
                try:
                    pipeline = ExtractionPipeline(session, router=router)
                    for target in targets:
                        outcome = await pipeline.extract(
                            target,
                            instrument_id=_uuid.UUID(instrument_id) if instrument_id else None,
                            force=force,
                        )
                        if outcome.skipped:
                            typer.echo(
                                f"{target}: skipped; this extraction identity already ran ($0)"
                            )
                            continue
                        typer.echo(
                            f"{target}: {outcome.llm_clauses} LLM clauses, "
                            f"{outcome.llm_covenants} covenants, "
                            f"{outcome.llm_failed} failed, "
                            f"{outcome.disagreements} rule/LLM disagreements, "
                            f"{outcome.queued_for_review} queued for review, "
                            f"${outcome.total_cost_usd:.4f}"
                            + (" [BUDGET EXCEEDED]" if outcome.budget_exceeded else "")
                        )
                        for error in outcome.errors:
                            typer.echo(f"  ! {error}")
                finally:
                    # Adapters hold an httpx client each; the router closes them.
                    await router.aclose()
        finally:
            await dispose_engines()

    asyncio.run(_run())


@app.command()
def ocr(
    document_id: str | None = typer.Argument(None, help="The document to OCR."),
    all_documents: bool = typer.Option(False, "--all", help="Every document with flagged pages."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report which pages would be sent, and what they would cost."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the spend confirmation."),
) -> None:
    """OCR scanned pages with the vision model, then chunk what it read.

    **This spends money.** Both halves run together on purpose: a transcription
    nobody chunked is spend that bought nothing retrievable, which is what the
    VLM did until this command existed.

    A document needing more than `MAX_VLM_PAGES_PER_DOC` pages fails loudly
    rather than processing the first N (CLAUDE.md 4).
    """
    import uuid as _uuid

    from app.db.repositories.documents import DocumentPageRepository, DocumentRepository
    from app.db.session import get_sessionmaker
    from app.ingest.ocr_chunks import OcrChunkingService
    from app.llm.adapters._http import ProviderQuotaExhaustedError
    from app.llm.cost import estimate_vlm_page_cost
    from app.llm.vlm import VlmPageCapExceededError, VlmService

    settings = get_settings()
    configure_logging(settings)

    if not document_id and not all_documents:
        typer.echo("Refusing to guess: pass a document id, or --all to mean it.", err=True)
        raise typer.Exit(code=2)

    if settings.OPENAI_API_KEY is None and not dry_run:
        typer.echo(
            "OPENAI_API_KEY is not set. Set it, or use --dry-run to see what would be sent.",
            err=True,
        )
        raise typer.Exit(code=2)

    async def _run() -> None:
        from app.llm.router import LLMRouter

        try:
            async with get_sessionmaker()() as session:
                targets = (
                    [_uuid.UUID(document_id)]
                    if document_id
                    else [row.id for row in await DocumentRepository(session).list(limit=500)]
                )

                pages_repo = DocumentPageRepository(session)
                flagged: dict[_uuid.UUID, int] = {}
                for target in targets:
                    pages = await pages_repo.list_needing_vlm(
                        target, confidence_threshold=settings.DEFAULT_CONFIDENCE_THRESHOLD
                    )
                    if pages:
                        flagged[target] = len(pages)

                total_pages = sum(flagged.values())
                if not total_pages:
                    typer.echo("No pages are flagged for the vision model. Nothing to do, $0.")
                    return

                per_page = estimate_vlm_page_cost()
                for target, count in flagged.items():
                    over = count > settings.MAX_VLM_PAGES_PER_DOC
                    note = (
                        f"  ** exceeds MAX_VLM_PAGES_PER_DOC={settings.MAX_VLM_PAGES_PER_DOC}; "
                        f"this document will be refused **"
                        if over
                        else ""
                    )
                    typer.echo(f"{target}: {count} page(s), ~${per_page * count:.4f}{note}")

                if dry_run:
                    typer.echo("\nDry run: nothing was dispatched and nothing was charged.")
                    return

                typer.echo(
                    f"\nEstimated ~${per_page * total_pages:.4f} for {total_pages} page(s) "
                    f"at ~${per_page}/page · ceilings ${settings.MAX_COST_PER_CALL_USD}/call · "
                    f"${settings.MAX_COST_PER_DOCUMENT_USD}/document"
                )
                if not yes and not typer.confirm(
                    f"Send {total_pages} page image(s) to {settings.VLM_MODEL}?"
                ):
                    typer.echo("Aborted; nothing was charged.")
                    return

                router = LLMRouter(session)
                try:
                    vlm = VlmService(session, router=router)
                    rechunker = OcrChunkingService(session)
                    for target in flagged:
                        try:
                            outcome = await vlm.process_document(target)
                        except VlmPageCapExceededError as exc:
                            typer.echo(f"{target}: refused — {exc}")
                            continue
                        except ProviderQuotaExhaustedError as exc:
                            # Every remaining page would fail identically, and
                            # backoff cannot fix a billing state. Say what to do.
                            typer.echo(f"\n{exc.provider} is out of credit: {exc.detail}", err=True)
                            raise typer.Exit(code=3) from exc
                        # Chunk in the same breath. Splitting these into two
                        # commands is how the transcription ended up stored and
                        # unread in the first place.
                        rechunked = await rechunker.rechunk_document(target)
                        await session.commit()
                        typer.echo(
                            f"{target}: {outcome.pages_processed} page(s) OCR'd, "
                            f"{outcome.pages_failed} failed, "
                            f"{rechunked.chunks_created} chunk(s) created, "
                            f"${outcome.total_cost_usd:.4f}"
                        )
                        typer.echo("    now run `index` to make the new chunks retrievable")
                finally:
                    await router.aclose()
        finally:
            await dispose_engines()

    asyncio.run(_run())


@app.command()
def query(question: str) -> None:
    """Answer a question over the deterministic path. No LLM, no spend."""
    # CLAUDE.md 1.6: the query path uses the read-only role and nothing else.
    # The read-write sessionmaker would have worked, which is exactly the
    # problem -- the invariant is enforced by the database only if the query
    # path actually connects as the role that lacks the grants.
    from app.db.session import get_readonly_sessionmaker
    from app.query.service import Answer, DeterministicQueryService

    settings = get_settings()
    configure_logging(settings)

    async def _run() -> Answer:
        try:
            async with get_readonly_sessionmaker()() as session:
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


@app.command()
def ask(
    question: str,
    user_id: str | None = typer.Option(None, "--user-id", help="Recorded on the query log row."),
    show_tools: bool = typer.Option(False, "--show-tools", help="List the tools the graph called."),
) -> None:
    """Answer a question through the LangGraph agent. Logged and audited.

    The same contract as `query` -- an answer, citations, a confidence and a
    refusal flag -- but routed through the Phase 7 graph, so the question and
    its answer land in `query_logs` and `audit_logs`.

    Free today: every node is deterministic, so nothing here reaches a paid
    provider. That changes if synthesis is ever handed to a model.
    """
    from app.agent.service import AgentAnswer, open_agent_query_service

    settings = get_settings()
    configure_logging(settings)

    async def _run() -> AgentAnswer:
        try:
            # Two sessions, opened with the roles CLAUDE.md 1.6 requires: the
            # read path as DATABASE_URL_RO, the query log as DATABASE_URL.
            # Constructing AgentQueryService(session) by hand here would work
            # and would quietly put the whole graph on one role.
            async with open_agent_query_service() as service:
                return await service.answer(question, user_id=user_id)
        finally:
            await dispose_engines()

    answer = asyncio.run(_run())
    typer.echo(f"intent:     {answer.intent.value}")
    typer.echo(f"confidence: {answer.confidence:.2f}{'  (refused)' if answer.refused else ''}")
    if show_tools and answer.tools_used:
        typer.echo(f"tools:      {', '.join(answer.tools_used)}")
    typer.echo("")
    typer.echo(answer.answer)
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
    # Read-only, for the same reason as `query` above (CLAUDE.md 1.6).
    from app.db.session import get_readonly_sessionmaker
    from app.evals.golden import GOLDEN_QUESTIONS, PHASE_4_TARGET
    from app.query.service import DeterministicQueryService

    settings = get_settings()
    configure_logging(settings)

    async def _run() -> list[tuple[str, bool, str]]:
        results: list[tuple[str, bool, str]] = []
        try:
            async with get_readonly_sessionmaker()() as session:
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


@app.command("check-schema")
def check_schema() -> None:
    """Compare the database's enum CHECK constraints against the models.

    `alembic check` cannot see these -- Alembic excludes type-bound
    constraints from autogenerate -- so adding a value to a StrEnum without a
    migration is invisible to it. This asks the database directly.
    """
    from app.db.schema_check import enum_constraint_drift
    from app.db.session import get_sessionmaker

    settings = get_settings()
    configure_logging(settings)

    async def _run() -> list[str]:
        try:
            async with get_sessionmaker()() as session:
                return [item.describe() for item in await enum_constraint_drift(session)]
        finally:
            await dispose_engines()

    drift = asyncio.run(_run())
    if not drift:
        typer.echo("enum constraints match the models")
        return
    for line in drift:
        typer.echo(f"DRIFT: {line}")
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
