"""The VLM routing heuristic (CLAUDE.md 4).

This is a spend gate, so each check is tested at its boundary rather than with
a comfortably wrong value: the difference between 119 and 120 characters is the
difference between a free page and a billed one.
"""

from __future__ import annotations

import pytest

from app.domain.enums import VlmReason
from app.domain.ingest import PageMetrics
from app.ingest.confidence import assess_page, document_confidence


def metrics(**overrides: object) -> PageMetrics:
    """A clean, fully parsed page, unless a field is overridden."""
    defaults: dict[str, object] = {
        "page_number": 1,
        "char_count": 2_000,
        "image_area_ratio": 0.05,
        "has_text_layer": True,
        "garbled_unicode_ratio": 0.0,
        "has_table_hint": False,
        "table_extraction_failed": False,
    }
    return PageMetrics(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_clean_page_needs_no_vlm() -> None:
    assessment = assess_page(metrics())

    assert assessment.needs_vlm is False
    assert assessment.confidence == 1.0
    assert assessment.reasons == ()
    assert assessment.reason_text is None


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"has_text_layer": False}, VlmReason.NO_TEXT_LAYER),
        ({"char_count": 119}, VlmReason.LOW_CHAR_COUNT),
        ({"image_area_ratio": 0.71}, VlmReason.HIGH_IMAGE_AREA),
        ({"garbled_unicode_ratio": 0.26}, VlmReason.GARBLED_UNICODE),
        (
            {"has_table_hint": True, "table_extraction_failed": True},
            VlmReason.TABLE_EXTRACTION_FAILED,
        ),
    ],
)
def test_each_check_routes_to_the_vlm(overrides: dict[str, object], reason: VlmReason) -> None:
    assessment = assess_page(metrics(**overrides))

    assert assessment.needs_vlm is True
    assert reason in assessment.reasons
    assert assessment.confidence < 1.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"char_count": 120},
        {"image_area_ratio": 0.70},
        {"garbled_unicode_ratio": 0.25},
        # A table hint alone is not a failure -- most tables extract fine.
        {"has_table_hint": True},
        # ...and a failure without a hint means there was no table to miss.
        {"table_extraction_failed": True},
    ],
)
def test_boundary_values_stay_on_the_free_path(overrides: dict[str, object]) -> None:
    assert assess_page(metrics(**overrides)).needs_vlm is False


def test_reasons_accumulate_for_a_scanned_page() -> None:
    assessment = assess_page(metrics(has_text_layer=False, char_count=0, image_area_ratio=0.99))

    assert assessment.confidence == 0.0
    assert assessment.reasons == (
        VlmReason.NO_TEXT_LAYER,
        VlmReason.LOW_CHAR_COUNT,
        VlmReason.HIGH_IMAGE_AREA,
    )
    # Stored verbatim on document_pages.vlm_reason, so cost is attributable.
    assert assessment.reason_text == "no_text_layer,low_char_count,high_image_area"


def test_assessment_is_a_pure_function_of_its_metrics() -> None:
    given = metrics(char_count=50, image_area_ratio=0.9)

    assert assess_page(given) == assess_page(given)


def test_document_confidence_averages_rather_than_taking_the_worst() -> None:
    pages = [assess_page(metrics()), assess_page(metrics(has_text_layer=False, char_count=0))]

    # One scanned appendix must not describe a 300-page prospectus as unparsed.
    assert document_confidence(pages) == pytest.approx(0.5)


def test_document_confidence_of_an_empty_document_is_zero() -> None:
    assert document_confidence([]) == 0.0
