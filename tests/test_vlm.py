"""VLM fallback tests.

The VLM is the only stage that can spend real money on a *page*, so the
properties worth pinning are all about not wasting it:

- the transcription is persisted (it is the entire product of the spend);
- an over-cap document fails loudly rather than reporting a quiet zero;
- the no-router path works, because that is the path CI takes.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.db.models.documents import DocumentPage
from app.db.repositories.documents import DocumentPageRepository
from app.domain.enums import ParseMethod
from app.llm.mock import MockLLMProvider
from app.llm.router import LLMRouter
from app.llm.vlm import VlmOutcome, VlmPageCapExceededError, VlmService

pytestmark = pytest.mark.usefixtures("storage_root")


@pytest_asyncio.fixture
async def scanned_document(
    db_session: AsyncSession, object_store: object, storage_root: Path
) -> uuid.UUID:
    """A two-page document whose pages both failed the text-layer checks."""
    from app.db.repositories.documents import DocumentRepository
    from app.ingest.service import IngestionService
    from tests.fixtures.synthetic_pdf import build_prospectus

    ingestion = IngestionService(db_session, object_store)  # type: ignore[arg-type]
    outcome = await ingestion.upload(filename="scan.pdf", data=build_prospectus())
    await ingestion.process(outcome.document.id)
    document_id = outcome.document.id

    # Force the pages into the "needs VLM" state the detector would have set.
    pages = await DocumentPageRepository(db_session).list_for_document(document_id)
    if not pages:
        await DocumentPageRepository(db_session).add(
            DocumentPage(
                document_id=document_id,
                page_number=1,
                confidence=0.1,
                vlm_reason="no_text_layer",
            )
        )
    for page in pages[:2]:
        page.confidence = 0.1
        page.vlm_reason = "no_text_layer"
        page.vlm_used = False
    await db_session.flush()

    assert await DocumentRepository(db_session).get(document_id) is not None
    return document_id


async def test_transcription_is_persisted(
    db_session: AsyncSession, scanned_document: uuid.UUID
) -> None:
    """The OCR text lands on the page row.

    It used to be read off the response into a local and dropped, while the
    page was still flagged `vlm_used` -- so the spend bought nothing and the
    flag made the page ineligible for a second attempt.
    """
    service = VlmService(db_session, router=LLMRouter(db_session, provider=MockLLMProvider()))

    outcome = await service.process_document(scanned_document)

    assert outcome.pages_processed > 0
    assert outcome.pages_failed == 0

    pages = await DocumentPageRepository(db_session).list_for_document(scanned_document)
    processed = [page for page in pages if page.vlm_used]
    assert processed
    for page in processed:
        assert page.ocr_text
        assert page.parse_method is ParseMethod.VLM


async def test_works_without_a_router(
    db_session: AsyncSession, scanned_document: uuid.UUID
) -> None:
    """The no-router path is what CI takes, and it used to fail every page.

    It reached `result.cache_hit` on a bare provider response, raised
    AttributeError, and had it swallowed by the per-page `except` -- so every
    page silently "failed" and the outcome still looked like a clean run.
    """
    service = VlmService(db_session, router=None)

    outcome = await service.process_document(scanned_document)

    assert isinstance(outcome, VlmOutcome)
    assert outcome.pages_processed > 0
    assert outcome.pages_failed == 0
    assert outcome.total_cost_usd == Decimal("0")


async def test_over_cap_document_raises(
    db_session: AsyncSession, scanned_document: uuid.UUID
) -> None:
    """CLAUDE.md 4: a document over the page cap must fail loudly.

    Returning a zero-page outcome is indistinguishable from "nothing needed
    OCR", which is the quiet failure the cap exists to prevent.
    """
    service = VlmService(db_session, router=LLMRouter(db_session, provider=MockLLMProvider()))
    service._settings = Settings(ENVIRONMENT="test", MAX_VLM_PAGES_PER_DOC=0)

    with pytest.raises(VlmPageCapExceededError) as excinfo:
        await service.process_document(scanned_document)

    assert excinfo.value.cap == 0
    assert excinfo.value.needed > 0

    # Nothing was marked processed, so a re-run after raising the cap still works.
    pages = await DocumentPageRepository(db_session).list_for_document(scanned_document)
    assert not any(page.vlm_used for page in pages)
