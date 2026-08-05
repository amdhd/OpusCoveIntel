"""Linking a document to the instrument it describes.

Shared by both extractors. It lived inside `RuleExtractionService` while the
LLM pipeline had no equivalent, so covenants from the LLM path were persisted
with `instrument_id = NULL` -- and a covenant with no instrument is invisible
to every portfolio query, which is the system's headline feature. The two paths
must agree on which issuer a document belongs to, so they now share the code
that decides.
"""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models.instruments import Instrument
from app.db.repositories.documents import DocumentChunkRepository

logger = get_logger(__name__)

# How much of the document to search for an issuer name. The front matter names
# the issuer; reading further invites a mention of some other party.
_CHUNKS_SCANNED = 20


async def resolve_instrument(session: AsyncSession, document_id: uuid.UUID) -> uuid.UUID | None:
    """Link a document to an instrument by issuer name.

    Deliberately literal: an issuer's registered name must appear verbatim in
    the document text. Fuzzy matching here would attach covenants to the wrong
    issuer, and a covenant on the wrong instrument is worse than a covenant on
    none -- it produces a confident, wrong portfolio answer.

    Returns None when nothing matches *and* when several issuers match. An
    ambiguous link is not better than no link.
    """
    chunks = await DocumentChunkRepository(session).list_for_document(
        document_id, limit=_CHUNKS_SCANNED
    )
    # Whitespace is collapsed on both sides before matching. A PDF wraps
    # "Synthetic Retail REIT\nBerhad" across a line, and a plain substring test
    # then silently fails to link the document to its instrument -- which looks
    # like "no covenants found" rather than like a bug.
    haystack = collapse(" ".join(chunk.chunk_text for chunk in chunks))
    if not haystack:
        return None

    result = await session.execute(select(Instrument))
    matches = [
        instrument
        for instrument in result.scalars().all()
        if collapse(instrument.issuer_name) in haystack
    ]
    if len(matches) == 1:
        return matches[0].id
    if len(matches) > 1:
        logger.warning(
            "document names several issuers; leaving it unlinked",
            extra={
                "document_id": str(document_id),
                "issuers": [item.issuer_name for item in matches],
            },
        )
    return None


def collapse(text: str) -> str:
    """Lower-case with runs of whitespace collapsed, for line-wrap-safe matching."""
    return " ".join(text.split()).lower()
