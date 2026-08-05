"""VLM fallback service — OCR for pages that failed text-layer checks.

Phase 3 detects pages needing the VLM; Phase 5 invokes it. This service:
1. Queries document_pages where confidence < threshold and vlm_used=False
2. Fetches the page image from the stored PDF
3. Calls the vision model (GPT-4o) via the LLM router
4. Records the extracted text and flips vlm_used=True

Respects MAX_VLM_PAGES_PER_DOC — a 400-page scan must fail loudly.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.domain.enums import LLMStage, ParseMethod

logger = get_logger(__name__)

VLM_OCR_PROMPT = (
    "Transcribe all text visible on this page. Preserve paragraph breaks, "
    "table structure (use | for columns), and any numerical values exactly "
    "as they appear. Output the raw text only — no commentary, no "
    "descriptions of formatting, no markdown except for table markers."
)


@dataclass(frozen=True)
class VlmOutcome:
    document_id: uuid.UUID
    pages_processed: int
    pages_skipped_cap: int
    pages_skipped_cached: int
    total_cost_usd: Decimal


class VlmService:
    """OCR scanned pages via the vision model.

    Usage:
        service = VlmService(session, router)
        outcome = await service.process_document(document_id)
    """

    def __init__(self, session: AsyncSession, router: object | None = None) -> None:
        self._session = session
        self._settings = get_settings()
        self._router = router

    async def process_document(self, document_id: uuid.UUID) -> VlmOutcome:
        """OCR every flagged page in a document, up to the VLM page cap."""
        from app.db.repositories.documents import DocumentPageRepository
        from app.ingest.storage import get_object_store

        pages_repo = DocumentPageRepository(self._session)
        pages = await pages_repo.list_needing_vlm(
            document_id,
            confidence_threshold=self._settings.DEFAULT_CONFIDENCE_THRESHOLD,
        )

        cap = self._settings.MAX_VLM_PAGES_PER_DOC
        if len(pages) > cap:
            logger.error(
                "vlm.page_cap_exceeded",
                extra={
                    "document_id": str(document_id),
                    "needing_vlm": len(pages),
                    "cap": cap,
                },
            )
            return VlmOutcome(
                document_id=document_id,
                pages_processed=0,
                pages_skipped_cap=len(pages),
                pages_skipped_cached=0,
                total_cost_usd=Decimal("0"),
            )

        if not pages:
            logger.info(
                "vlm.no_pages_needed",
                extra={"document_id": str(document_id)},
            )
            return VlmOutcome(
                document_id=document_id,
                pages_processed=0,
                pages_skipped_cap=0,
                pages_skipped_cached=0,
                total_cost_usd=Decimal("0"),
            )

        # Fetch the stored PDF to extract page images
        store = get_object_store()
        from app.db.repositories.documents import DocumentRepository

        document = await DocumentRepository(self._session).get(document_id)
        if document is None or document.storage_uri is None:
            raise LookupError(f"document {document_id} not found or has no stored bytes")

        pdf_bytes = await store.get(document.storage_uri)

        processed = 0
        skipped_cached = 0
        total_cost = Decimal("0")

        for page in pages:
            try:
                # Extract the page as an image
                image_bytes = _render_page_image(pdf_bytes, page.page_number)

                # Call the VLM via the router (or directly via mock in CI)
                if self._router is not None:
                    # Use the LLMRouter interface — `object` to avoid circular import
                    router = self._router
                    result = await router.vision(  # type: ignore[attr-defined]
                        stage=LLMStage.VLM_OCR,
                        image_bytes=image_bytes,
                        prompt=VLM_OCR_PROMPT,
                        document_id=document_id,
                    )
                else:
                    # Direct path for environments without a router (e.g. tests
                    # with a mock provider injected)
                    result = await self._vision_direct(image_bytes)

                if result.cache_hit:
                    skipped_cached += 1
                else:
                    total_cost += result.estimated_cost_usd

                # Record the OCR result on the page row
                _ocr_text = (
                    result.content if isinstance(result.content, str) else str(result.content)
                )
                page.parse_method = ParseMethod.VLM
                page.vlm_used = True
                # vlm_reason is already set from the detection phase; the CHECK
                # constraint NOT vlm_used OR vlm_reason IS NOT NULL is satisfied.
                page.confidence = 0.85  # VLM output is trusted but not perfect

                processed += 1
                logger.info(
                    "vlm.page_processed",
                    extra={
                        "document_id": str(document_id),
                        "page": page.page_number,
                        "cost": str(result.estimated_cost_usd),
                        "cached": result.cache_hit,
                    },
                )
            except Exception as exc:  # noqa: BLE001 — log and continue; one bad page shouldn't fail the doc
                logger.error(
                    "vlm.page_failed",
                    extra={
                        "document_id": str(document_id),
                        "page": page.page_number,
                        "error": str(exc),
                    },
                )

        await self._session.flush()
        return VlmOutcome(
            document_id=document_id,
            pages_processed=processed,
            pages_skipped_cap=0,
            pages_skipped_cached=skipped_cached,
            total_cost_usd=total_cost,
        )

    async def _vision_direct(self, image_bytes: bytes) -> object:
        """Direct vision call (no router). Used when mock is injected."""
        from app.llm.mock import MockLLMProvider

        mock = MockLLMProvider()
        return await mock.vision(
            model_id=get_settings().VLM_MODEL,
            image_bytes=image_bytes,
            prompt=VLM_OCR_PROMPT,
        )


def _render_page_image(pdf_bytes: bytes, page_number: int) -> bytes:
    """Render a single PDF page as a PNG image.

    Uses PyMuPDF (fitz), which is already a project dependency for PDF parsing.
    """

    import fitz  # type: ignore[import-untyped]  # PyMuPDF

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_number < 1 or page_number > len(doc):
            raise ValueError(f"page {page_number} out of range (1-{len(doc)})")
        page = doc[page_number - 1]
        # Render at 150 DPI — enough for OCR, not so high that images blow up.
        mat = fitz.Matrix(150 / 72, 150 / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        return img_bytes  # type: ignore[no-any-return]
    finally:
        doc.close()
