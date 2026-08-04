"""Operator CLI.

Phase 1 ships configuration inspection only. Ingestion, extraction, query, and
eval commands arrive in Phases 3-8.
"""

from __future__ import annotations

import asyncio

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
