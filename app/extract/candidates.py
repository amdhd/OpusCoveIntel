"""Candidate detection — narrows a document to the spans worth LLM attention.

CLAUDE.md S2, cost lever #1: "Candidate narrowing - never send a whole document
to Opus. ~20x saving."

The primary narrowing mechanism is regex: the same patterns in `app/extract/patterns.py`
that the rule extractor uses. They are free, fast, and were written to be precise
rather than greedy — a pattern that fires only on explicit covenant language gives
us a high-signal candidate list.

FTS and kNN are reserved for a future enhancement. The interface accepts them so
the caller does not change when they land.

Each candidate is a widened span — the regex hit plus surrounding context — so the
LLM has enough text to understand what it is reading.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, Protocol

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.repositories.documents import DocumentChunkRepository
from app.domain.enums import ClauseType
from app.extract.patterns import ALL_PATTERNS


class _HasChunkText(Protocol):
    """Any object that carries chunk_text + page metadata — real or fake."""

    id: uuid.UUID
    chunk_text: str
    page_number: int
    section_title: str | None


logger = get_logger(__name__)

# How many characters of context to include on each side of a regex hit.
# Wide enough to capture related sentences without pulling in unrelated clauses.
CONTEXT_CHARS: Final[int] = 400

# Upper bound on the number of candidates returned. A document that produces
# more than this is either a very long document or the patterns are firing on
# boilerplate — both worth logging rather than silently spending on.
MAX_CANDIDATES: Final[int] = 200


@dataclass(frozen=True, slots=True)
class Candidate:
    """One span of a chunk that might contain extractable covenants."""

    chunk_id: uuid.UUID
    text: str
    char_start: int
    char_end: int
    clause_type_hints: frozenset[ClauseType] = field(default_factory=frozenset)
    page_number: int = 1
    section_title: str | None = None

    def __repr__(self) -> str:
        text_preview = self.text[:80].replace("\n", " ")
        return (
            f"Candidate(chunk={self.chunk_id}, page={self.page_number}, "
            f"span=({self.char_start}, {self.char_end}), hints={set(self.clause_type_hints)}, "
            f'text="{text_preview}...")'
        )


class CandidateDetectionService:
    """Find spans a document chunk that are likely to contain covenants.

    Regex-based narrowing is the free, high-precision first pass. FTS and kNN
    are reserved for future phases — the interface leaves room for them without
    changing callers.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._chunks = DocumentChunkRepository(session)

    async def detect(
        self,
        document_id: uuid.UUID,
        *,
        limit: int = MAX_CANDIDATES,
    ) -> list[Candidate]:
        """Return candidate spans for a document, sorted by page then position.

        The caller is responsible for cost-limiting the number actually sent to
        the LLM — this returns everything the patterns found, up to `limit`.
        """
        chunks = await self._chunks.list_for_document(document_id, limit=100_000)
        if not chunks:
            logger.warning(
                "no chunks for document; nothing to detect",
                extra={"document_id": str(document_id)},
            )
            return []

        candidates = _detect_regex(chunks, limit=limit)

        logger.info(
            "candidate detection complete",
            extra={
                "document_id": str(document_id),
                "chunks": len(chunks),
                "candidates": len(candidates),
            },
        )
        return candidates


def _detect_regex(chunks: Sequence[_HasChunkText], *, limit: int) -> list[Candidate]:
    """Run every pattern over every chunk; widen and deduplicate the hits."""
    raw: list[tuple[int, int, _HasChunkText, ClauseType]] = []

    for chunk in chunks:
        text = chunk.chunk_text
        for pattern in ALL_PATTERNS:
            for match in pattern.regex.finditer(text):
                raw.append((match.start(), match.end(), chunk, pattern.clause_type))

    if not raw:
        return []

    # Widen to include context, then deduplicate overlapping spans.
    widened = _widen_and_deduplicate(raw)
    return _build_candidates(widened)[:limit]


def _widen_and_deduplicate(
    hits: list[tuple[int, int, _HasChunkText, ClauseType]],
) -> list[tuple[int, int, _HasChunkText, frozenset[ClauseType]]]:
    """Widen each hit to its context window, then merge overlapping spans.

    Two patterns firing on adjacent sentences in the same chunk should produce
    one candidate spanning both, not two overlapping ones that waste LLM calls.
    """
    # Sort by chunk then position.
    hits.sort(key=lambda item: (str(item[2].id), item[0]))

    spans: list[tuple[int, int, _HasChunkText, set[ClauseType]]] = []
    for start, end, chunk, clause_type in hits:
        wide_start = max(0, start - CONTEXT_CHARS)
        wide_end = min(len(chunk.chunk_text), end + CONTEXT_CHARS)

        # Merge with the previous span if they overlap and share a chunk.
        if spans and spans[-1][2].id == chunk.id and wide_start <= spans[-1][1]:
            prev = spans[-1]
            spans[-1] = (prev[0], max(prev[1], wide_end), prev[2], prev[3] | {clause_type})
        else:
            spans.append((wide_start, wide_end, chunk, {clause_type}))

    return [(s[0], s[1], s[2], frozenset(s[3])) for s in spans]


def _build_candidates(
    spans: list[tuple[int, int, _HasChunkText, frozenset[ClauseType]]],
) -> list[Candidate]:
    """Build Candidate objects from (start, end, chunk, hints) tuples."""
    result: list[Candidate] = []
    for start, end, chunk, hints in spans:
        text = chunk.chunk_text[start:end]
        if not text.strip():
            continue
        result.append(
            Candidate(
                chunk_id=chunk.id,
                text=text,
                char_start=start,
                char_end=end,
                clause_type_hints=hints,
                page_number=chunk.page_number,
                section_title=chunk.section_title,
            )
        )
    return result
