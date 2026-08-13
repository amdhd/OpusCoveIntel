"""Document, page and chunk repositories."""

from __future__ import annotations

import re
import typing
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import CursorResult, cast, func, select, update
from sqlalchemy.dialects.postgresql import REGCONFIG

from app.db.models.documents import Document, DocumentChunk, DocumentPage
from app.db.repositories.base import BaseRepository
from app.domain.enums import DocumentStatus


class DocumentRepository(BaseRepository[Document]):
    model = Document

    async def get_by_sha256(self, sha256: str) -> Document | None:
        """Deduplication lookup -- the same bytes are the same document."""
        result = await self.session.execute(select(Document).where(Document.sha256 == sha256))
        return result.scalar_one_or_none()

    async def exists_by_sha256(self, sha256: str) -> bool:
        result = await self.session.execute(
            select(func.count()).select_from(Document).where(Document.sha256 == sha256)
        )
        return int(result.scalar_one()) > 0

    async def list_by_status(
        self, status: DocumentStatus, *, limit: int = 100
    ) -> Sequence[Document]:
        result = await self.session.execute(
            select(Document)
            .where(Document.status == status)
            .order_by(Document.created_at)
            .limit(limit)
        )
        return result.scalars().all()


class DocumentPageRepository(BaseRepository[DocumentPage]):
    model = DocumentPage

    async def list_for_document(self, document_id: uuid.UUID) -> Sequence[DocumentPage]:
        result = await self.session.execute(
            select(DocumentPage)
            .where(DocumentPage.document_id == document_id)
            .order_by(DocumentPage.page_number)
        )
        return result.scalars().all()

    async def list_needing_vlm(
        self, document_id: uuid.UUID, *, confidence_threshold: float
    ) -> Sequence[DocumentPage]:
        """Low-confidence pages not yet sent to the VLM.

        The VLM budget cap (CLAUDE.md 4) is applied by the caller against the
        length of this list -- a document that would blow the cap must fail
        loudly rather than silently process the first N pages.
        """
        result = await self.session.execute(
            select(DocumentPage)
            .where(
                DocumentPage.document_id == document_id,
                DocumentPage.confidence < confidence_threshold,
                DocumentPage.vlm_used.is_(False),
            )
            .order_by(DocumentPage.page_number)
        )
        return result.scalars().all()

    async def count_vlm_used(self, document_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(DocumentPage)
            .where(DocumentPage.document_id == document_id, DocumentPage.vlm_used.is_(True))
        )
        return int(result.scalar_one())


def _tsquery_terms(query: str) -> str:
    """Build a safe OR-ed tsquery string from free text.

    Tokens are reduced to `[0-9A-Za-z]+` before being joined, so no `to_tsquery`
    operator (`&`, `|`, `!`, `:`, parentheses) can survive from user input --
    the value is still bound as a parameter, but a syntactically invalid
    tsquery would raise rather than return nothing, and that is a worse failure
    than dropping punctuation.
    """
    tokens = re.findall(r"[0-9A-Za-z]+", query)
    return " | ".join(tokens)


# Both retrieval legs order by a score that ties freely: `ts_rank_cd` returns
# the same rank for chunks matching the same terms, and a lexical embedder
# returns identical cosine distances for near-identical text. Without a
# deterministic tiebreak Postgres is free to return tied rows in any order, and
# since RRF fuses *ranks*, a reshuffle inside a tie changes the fused result --
# which made the Phase 4 acceptance test fail roughly one run in three. The
# hybrid module already sorts its own output stably; this is the other half.
_TIE_BREAK = (DocumentChunk.page_number, DocumentChunk.ordinal, DocumentChunk.id)


class DocumentChunkRepository(BaseRepository[DocumentChunk]):
    model = DocumentChunk

    async def list_for_document(
        self, document_id: uuid.UUID, *, limit: int = 1000
    ) -> Sequence[DocumentChunk]:
        result = await self.session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.page_number, DocumentChunk.ordinal)
            .limit(limit)
        )
        return result.scalars().all()

    async def get_by_hash(self, document_id: uuid.UUID, chunk_hash: str) -> DocumentChunk | None:
        result = await self.session.execute(
            select(DocumentChunk).where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.hash == chunk_hash,
            )
        )
        return result.scalar_one_or_none()

    async def list_unembedded(self, *, limit: int = 500) -> Sequence[DocumentChunk]:
        """Chunks with no vector yet -- the indexer's work queue."""
        result = await self.session.execute(
            select(DocumentChunk)
            .where(DocumentChunk.embedding.is_(None))
            .order_by(DocumentChunk.document_id, DocumentChunk.ordinal)
            .limit(limit)
        )
        return result.scalars().all()

    async def refresh_fts(self, document_id: uuid.UUID) -> int:
        """Rebuild the tsvector column for one document's chunks.

        The configuration is read *per row* from `chunks.fts_config`: a single
        Malaysian prospectus mixes English and Bahasa Malaysia, and Postgres
        ships no Malay stemmer, so BM rows index under `simple` while English
        rows use `english` (CLAUDE.md 6). One document-wide config would stem
        Malay with English rules and quietly lose the clause.
        """
        statement = (
            update(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .values(
                fts=func.to_tsvector(
                    cast(DocumentChunk.fts_config, REGCONFIG), DocumentChunk.chunk_text
                )
            )
        )
        # An UPDATE always yields a CursorResult, but `execute()` is typed as
        # returning the base Result, which does not declare `rowcount`.
        result = typing.cast("CursorResult[Any]", await self.session.execute(statement))
        await self.session.flush()
        return int(result.rowcount or 0)

    async def search_by_fts(
        self,
        query: str,
        *,
        limit: int = 20,
        document_id: uuid.UUID | None = None,
    ) -> Sequence[tuple[DocumentChunk, float]]:
        """Keyword leg of hybrid retrieval, ranked by `ts_rank_cd`.

        Matches each row against a tsquery built with that row's own
        configuration, so a Malay chunk is queried the way it was indexed.

        Terms are OR-ed, not AND-ed. `plainto_tsquery` requires *every* term to
        be present, which for a natural-language question is close to useless:
        "cross default threshold above RM30 million" returns nothing against a
        clause that says exactly that, because the clause never uses the word
        "threshold". OR plus `ts_rank_cd` is the search-engine behaviour --
        chunks matching more of the query, more densely, rank higher.
        """
        terms = _tsquery_terms(query)
        if not terms:
            return []
        tsquery = func.to_tsquery(cast(DocumentChunk.fts_config, REGCONFIG), terms)
        score = func.ts_rank_cd(DocumentChunk.fts, tsquery)
        stmt = (
            select(DocumentChunk, score.label("score"))
            .where(DocumentChunk.fts.is_not(None), DocumentChunk.fts.op("@@")(tsquery))
            .order_by(score.desc(), *_TIE_BREAK)
            .limit(limit)
        )
        if document_id is not None:
            stmt = stmt.where(DocumentChunk.document_id == document_id)
        result = await self.session.execute(stmt)
        return [(row[0], float(row[1])) for row in result.all()]

    async def embedding_models(self) -> set[str]:
        """Which models produced the embeddings currently in the index.

        Normally one. More than one means a re-embed was started and not
        finished, and knowing that is what turns "the vector leg found nothing"
        from a mystery into a sentence.
        """
        result = await self.session.execute(
            select(DocumentChunk.embedding_model)
            .where(DocumentChunk.embedding_model.is_not(None))
            .distinct()
        )
        return {model for model in result.scalars().all() if model}

    async def search_by_vector(
        self,
        embedding: list[float],
        *,
        limit: int = 20,
        document_id: uuid.UUID | None = None,
        embedding_model: str | None = None,
    ) -> Sequence[tuple[DocumentChunk, float]]:
        """Vector leg of hybrid retrieval, ranked by cosine similarity.

        `embedding_model` filters to chunks embedded by the same model.
        Comparing vectors from two different models is meaningless, and the
        filter is what stops a half-migrated corpus from returning nonsense
        instead of an error.
        """
        distance = DocumentChunk.embedding.cosine_distance(embedding)
        stmt = (
            select(DocumentChunk, distance.label("distance"))
            .where(DocumentChunk.embedding.is_not(None))
            .order_by(distance, *_TIE_BREAK)
            .limit(limit)
        )
        if document_id is not None:
            stmt = stmt.where(DocumentChunk.document_id == document_id)
        if embedding_model is not None:
            stmt = stmt.where(DocumentChunk.embedding_model == embedding_model)
        result = await self.session.execute(stmt)
        # Cosine distance is 1 - similarity; callers rank, so hand back similarity.
        return [(row[0], 1.0 - float(row[1])) for row in result.all()]

    async def count_embedded(self, document_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(func.count())
            .select_from(DocumentChunk)
            .where(
                DocumentChunk.document_id == document_id,
                DocumentChunk.embedding.is_not(None),
            )
        )
        return int(result.scalar_one())
