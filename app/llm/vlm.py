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
from app.llm.adapters._http import ProviderQuotaExhaustedError

logger = get_logger(__name__)

VLM_OCR_PROMPT = (
    "Transcribe all text visible on this page. Preserve paragraph breaks, "
    "table structure (use | for columns), and any numerical values exactly "
    "as they appear. Output the raw text only — no commentary, no "
    "descriptions of formatting, no markdown except for table markers."
)

# What a VLM-transcribed page is worth as evidence. Below 1.0 on purpose: OCR
# of a scan is good, not authoritative, and CLAUDE.md 5 routes anything sourced
# from a VLM page to human review regardless.
VLM_PAGE_CONFIDENCE = 0.85


@dataclass(frozen=True)
class VlmOutcome:
    document_id: uuid.UUID
    pages_processed: int
    pages_skipped_cap: int
    pages_skipped_cached: int
    total_cost_usd: Decimal
    pages_failed: int = 0


class VlmPageCapExceededError(RuntimeError):
    """A document needs more VLM pages than `MAX_VLM_PAGES_PER_DOC` allows.

    CLAUDE.md 4: a 400-page scan that trips every confidence check must fail
    loudly rather than quietly spend $80 -- or, just as bad, quietly do nothing
    and report success with zero pages processed.
    """

    def __init__(self, document_id: uuid.UUID, *, needed: int, cap: int) -> None:
        self.document_id = document_id
        self.needed = needed
        self.cap = cap
        super().__init__(
            f"document {document_id} needs VLM on {needed} pages, "
            f"which exceeds MAX_VLM_PAGES_PER_DOC={cap}"
        )


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
            # CLAUDE.md 4: "must fail loudly, not quietly spend $80". Returning
            # a zero-page outcome was quiet -- a caller checking only
            # `pages_processed` could not tell "nothing needed OCR" from "this
            # document would have cost more than the cap allows".
            logger.error(
                "vlm.page_cap_exceeded",
                extra={
                    "document_id": str(document_id),
                    "needing_vlm": len(pages),
                    "cap": cap,
                },
            )
            raise VlmPageCapExceededError(document_id, needed=len(pages), cap=cap)

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

        # Fetch the stored PDF to extract page images.
        store = get_object_store()
        from app.db.repositories.documents import DocumentRepository
        from app.ingest.service import storage_key

        document = await DocumentRepository(self._session).get(document_id)
        if document is None or document.storage_uri is None:
            raise LookupError(f"document {document_id} not found or has no stored bytes")

        # `storage_uri` is a URI ("file:///..."); the store is keyed by
        # content hash. Passing the URI made every call raise ObjectStoreError,
        # so this service could not load a single document -- it had no test
        # that reached this line. `IngestionService` derives the key the same way.
        pdf_bytes = await store.get(storage_key(document.sha256))

        processed = 0
        skipped_cached = 0
        failed = 0
        total_cost = Decimal("0")

        for page in pages:
            try:
                image_bytes = _render_page_image(pdf_bytes, page.page_number)
            except Exception as exc:  # noqa: BLE001 — one unrenderable page is not the document
                failed += 1
                logger.error(
                    "vlm.render_failed",
                    extra={
                        "document_id": str(document_id),
                        "page": page.page_number,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                continue

            try:
                ocr_text, cache_hit, cost = await self._ocr_page(image_bytes, document_id)

                if not ocr_text.strip():
                    # An empty transcription is a failure, not a result. Marking
                    # the page `vlm_used` would exclude it from every future
                    # attempt while leaving the document no better off.
                    raise ValueError("VLM returned no text for the page")

                if cache_hit:
                    skipped_cached += 1
                else:
                    total_cost += cost

                page.parse_method = ParseMethod.VLM
                page.vlm_used = True
                # This is the product of the spend. Dropping it -- which is what
                # this loop used to do -- paid for an OCR pass and then binned it.
                page.ocr_text = ocr_text
                # vlm_reason is already set from the detection phase; the CHECK
                # constraint NOT vlm_used OR vlm_reason IS NOT NULL is satisfied.
                page.confidence = VLM_PAGE_CONFIDENCE

                processed += 1
                logger.info(
                    "vlm.page_processed",
                    extra={
                        "document_id": str(document_id),
                        "page": page.page_number,
                        "cost": str(cost),
                        "cached": cache_hit,
                        "ocr_chars": len(ocr_text),
                    },
                )
            except ProviderQuotaExhaustedError:
                # Not a bad page — the account is out of money, so every
                # remaining page fails the same way. Continuing would turn one
                # billing problem into N identical log lines and hide the cause.
                logger.error(
                    "vlm.aborted_quota_exhausted",
                    extra={"document_id": str(document_id), "page": page.page_number},
                )
                raise
            except Exception as exc:  # noqa: BLE001 — log and continue; one bad page shouldn't fail the doc
                failed += 1
                logger.error(
                    "vlm.page_failed",
                    extra={
                        "document_id": str(document_id),
                        "page": page.page_number,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )

        await self._session.flush()
        return VlmOutcome(
            document_id=document_id,
            pages_processed=processed,
            pages_skipped_cap=0,
            pages_skipped_cached=skipped_cached,
            total_cost_usd=total_cost,
            pages_failed=failed,
        )

    async def _ocr_page(
        self, image_bytes: bytes, document_id: uuid.UUID
    ) -> tuple[str, bool, Decimal]:
        """One page through the router, or through the mock when there is none.

        Returns `(text, cache_hit, cost)`. The two paths return different
        objects -- an `LLMCallResult` from the router, a bare provider response
        from the mock -- so the difference is flattened here rather than left
        for the caller to trip over. It used to reach `result.cache_hit` on the
        mock's response, raise AttributeError, and be swallowed by the
        page-level `except`: every page silently "failed" whenever no router
        was passed.
        """
        if self._router is not None:
            result = await self._router.vision(  # type: ignore[attr-defined]
                stage=LLMStage.VLM_OCR,
                image_bytes=image_bytes,
                prompt=VLM_OCR_PROMPT,
                document_id=document_id,
            )
            return _as_text(result.content), bool(result.cache_hit), result.estimated_cost_usd

        from app.llm.mock import MockLLMProvider

        response = await MockLLMProvider().vision(
            model_id=self._settings.VLM_MODEL,
            image_bytes=image_bytes,
            prompt=VLM_OCR_PROMPT,
        )
        # The mock never bills and never caches, so both are trivially known.
        return _as_text(response.content), False, Decimal("0")


def _as_text(content: object) -> str:
    """Flatten a provider response body to the transcribed text."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        # A structured response would carry the text under a single field.
        for key in ("text", "content", "ocr_text"):
            value = content.get(key)
            if isinstance(value, str):
                return value
    return str(content)


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
