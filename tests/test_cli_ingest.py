"""The `ingest` CLI command.

One thing worth pinning: what the command tells the service about the file.

`IngestionService.upload` has taken a `document_type` since it was written, and
this command did not pass one -- so every document ingested from a terminal was
recorded as `unknown`, and the classification stage that would fill it in is not
built (CLAUDE.md 2). That is invisible from the service's own tests, which call
`upload` directly and can pass whatever they like.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from app.cli import app
from app.domain.enums import DocumentType

runner = CliRunner()


class _Outcome:
    def __init__(self) -> None:
        self.document = type("_Doc", (), {"id": uuid.uuid4()})()
        self.duplicate = False


class _RecordingService:
    """Stands in for `IngestionService`, recording what the command asked for."""

    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, *args: object, **kwargs: object) -> None: ...

    async def upload(self, **kwargs: Any) -> _Outcome:
        type(self).calls.append(kwargs)
        return _Outcome()


@pytest.fixture
def recording(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingService]:
    """Intercept the service the command imports, and the database it opens."""
    _RecordingService.calls = []
    monkeypatch.setattr("app.ingest.service.IngestionService", _RecordingService)
    monkeypatch.setattr("app.ingest.storage.get_object_store", lambda *a, **k: object())

    class _Session:
        async def __aenter__(self) -> _Session:
            return self

        async def __aexit__(self, *exc: object) -> None: ...

    monkeypatch.setattr("app.db.session.get_sessionmaker", lambda: lambda: _Session())

    async def _no_dispose() -> None: ...

    monkeypatch.setattr("app.cli.dispose_engines", _no_dispose)
    return _RecordingService


@pytest.fixture
def a_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "trust-deed.pdf"
    path.write_bytes(b"%PDF-1.7\n")
    return path


class TestTheDeclaredType:
    def test_the_type_reaches_the_service(
        self, recording: type[_RecordingService], a_pdf: Path
    ) -> None:
        result = runner.invoke(app, ["ingest", str(a_pdf), "--no-process", "--type", "trust_deed"])

        assert result.exit_code == 0, result.output
        assert recording.calls, "the command should have called upload"
        assert recording.calls[0]["document_type"] is DocumentType.TRUST_DEED

    def test_saying_nothing_still_means_unknown(
        self, recording: type[_RecordingService], a_pdf: Path
    ) -> None:
        """The default is the honest one: nobody said what this file was."""
        result = runner.invoke(app, ["ingest", str(a_pdf), "--no-process"])

        assert result.exit_code == 0, result.output
        assert recording.calls[0]["document_type"] is DocumentType.UNKNOWN

    def test_a_type_the_enum_does_not_have_is_refused(
        self, recording: type[_RecordingService], a_pdf: Path
    ) -> None:
        result = runner.invoke(app, ["ingest", str(a_pdf), "--type", "financial_statement"])

        assert result.exit_code == 2
        assert not recording.calls, "nothing should be stored on a bad type"
