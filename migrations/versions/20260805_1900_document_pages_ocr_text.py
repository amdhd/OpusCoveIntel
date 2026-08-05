"""store the text a VLM page actually transcribed

Phase 5 shipped the VLM fallback without anywhere to put its output: the OCR
result was read off the provider response and dropped, while the page was still
marked `vlm_used`. That flag excludes the page from every later attempt, so the
spend was both wasted and unrepeatable.

`ocr_text` is the destination, and the new CHECK makes the two facts inseparable
in the same way `vlm_use_requires_reason` already does for the reason.

Revision ID: 4c1f2a7be913
Revises: 950444bd060a
Create Date: 2026-08-05 19:00:00.000000+00:00

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "4c1f2a7be913"
down_revision: str | Sequence[str] | None = "950444bd060a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("document_pages", sa.Column("ocr_text", sa.Text(), nullable=True))
    # Any pre-existing row marked vlm_used has no transcription to restore --
    # the text was never written. Clear the flag so those pages become eligible
    # for OCR again rather than failing the new constraint.
    op.execute("UPDATE document_pages SET vlm_used = false WHERE vlm_used AND ocr_text IS NULL")
    op.create_check_constraint(
        "vlm_use_requires_ocr_text",
        "document_pages",
        "NOT vlm_used OR (ocr_text IS NOT NULL AND length(ocr_text) > 0)",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("vlm_use_requires_ocr_text", "document_pages", type_="check")
    op.drop_column("document_pages", "ocr_text")
