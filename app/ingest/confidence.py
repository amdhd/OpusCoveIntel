"""Page-confidence scoring -- the gate that decides VLM spend.

CLAUDE.md 4 fixes the routing rule; this module is its executable form:

    needs_vlm = (
        page.text_char_count < 120
        or page.image_area_ratio > 0.70
        or page.text_layer is None
        or (page.has_table_hint and pdfplumber_extraction_failed)
        or page.garbled_unicode_ratio > 0.25
    )

Two properties matter more than the exact numbers. It is a **pure function** of
metrics, so a routing decision can be replayed from the stored row without the
PDF. And it records *which* check tripped, because "this document cost $8" must
be answerable by query rather than by re-running the parser.

Phase 3 only detects. Nothing here calls a model.
"""

from __future__ import annotations

from app.domain.enums import VlmReason
from app.domain.ingest import PageAssessment, PageMetrics

MIN_CHAR_COUNT = 120
MAX_IMAGE_AREA_RATIO = 0.70
MAX_GARBLED_RATIO = 0.25

# Confidence penalties. A page with no text layer at all is worth nothing to
# the text pipeline, so it costs the entire budget; the softer signals are
# additive because they co-occur (a scan is usually image-heavy *and* short).
_PENALTIES: dict[VlmReason, float] = {
    VlmReason.NO_TEXT_LAYER: 1.00,
    VlmReason.LOW_CHAR_COUNT: 0.40,
    VlmReason.HIGH_IMAGE_AREA: 0.30,
    VlmReason.GARBLED_UNICODE: 0.30,
    VlmReason.TABLE_EXTRACTION_FAILED: 0.15,
}


def assess_page(metrics: PageMetrics) -> PageAssessment:
    """Score one page and name every check it failed."""
    reasons: list[VlmReason] = []

    if not metrics.has_text_layer:
        reasons.append(VlmReason.NO_TEXT_LAYER)
    if metrics.char_count < MIN_CHAR_COUNT:
        reasons.append(VlmReason.LOW_CHAR_COUNT)
    if metrics.image_area_ratio > MAX_IMAGE_AREA_RATIO:
        reasons.append(VlmReason.HIGH_IMAGE_AREA)
    # A table hint with no extractable table is the classic ruled-scan case:
    # the lines are drawn, the cells are pixels.
    if metrics.has_table_hint and metrics.table_extraction_failed:
        reasons.append(VlmReason.TABLE_EXTRACTION_FAILED)
    if metrics.garbled_unicode_ratio > MAX_GARBLED_RATIO:
        reasons.append(VlmReason.GARBLED_UNICODE)

    penalty = sum(_PENALTIES[reason] for reason in reasons)
    confidence = max(0.0, min(1.0, 1.0 - penalty))
    return PageAssessment(
        needs_vlm=bool(reasons),
        confidence=confidence,
        reasons=tuple(reasons),
    )


def document_confidence(assessments: list[PageAssessment]) -> float:
    """Mean page confidence -- `documents.parse_confidence`.

    A mean, not a minimum: one scanned appendix in a 300-page prospectus should
    not describe the whole document as unparsed.
    """
    if not assessments:
        return 0.0
    return sum(item.confidence for item in assessments) / len(assessments)
