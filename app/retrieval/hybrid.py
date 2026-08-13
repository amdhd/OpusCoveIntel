"""Hybrid retrieval with reciprocal-rank fusion.

Two legs, because they fail differently. Keyword search finds "RM30,000,000"
and "Ijarah" exactly and is useless when the question and the document choose
different words. Vector search tolerates paraphrase and is happy to return
something plausible and wrong. Neither is reliable alone on legal text, where
the exact number *is* the answer and the surrounding prose is boilerplate.

Fusion is **reciprocal rank fusion** rather than a weighted score blend:

    score(chunk) = sum over legs of 1 / (k + rank_in_that_leg)

RRF combines *ranks*, not scores, which is what makes it safe here -- cosine
similarity and `ts_rank_cd` are not on the same scale, have no shared zero, and
drift apart the moment the embedder changes. Normalising them against each
other would produce a number that looks principled and means nothing. The
constant `k` damps the influence of the top rank so one leg cannot dominate
purely by being confident.

A chunk found by both legs outranks a chunk found by one, which is the whole
point: agreement between independent retrieval strategies is evidence.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.db.models.documents import DocumentChunk
from app.db.repositories.documents import DocumentChunkRepository
from app.llm.embeddings import Embedder, get_embedder

logger = get_logger(__name__)

# The standard RRF damping constant (Cormack et al., 2009). Large enough that
# rank 1 and rank 2 are close, so a single leg's top hit does not win outright.
RRF_K = 60

DEFAULT_LIMIT = 10
# Each leg looks deeper than the final cut, so fusion has room to promote a
# chunk that both legs ranked mid-table.
LEG_OVERSAMPLE = 4


@dataclass(frozen=True)
class SearchHit:
    """One fused result, carrying where each leg placed it."""

    chunk: DocumentChunk
    score: float
    vector_rank: int | None = None
    fts_rank: int | None = None
    vector_score: float | None = None
    fts_score: float | None = None

    @property
    def found_by_both(self) -> bool:
        return self.vector_rank is not None and self.fts_rank is not None

    @property
    def legs(self) -> tuple[str, ...]:
        found: list[str] = []
        if self.vector_rank is not None:
            found.append("vector")
        if self.fts_rank is not None:
            found.append("fts")
        return tuple(found)


@dataclass
class _Accumulator:
    chunk: DocumentChunk
    score: float = 0.0
    vector_rank: int | None = None
    fts_rank: int | None = None
    vector_score: float | None = None
    fts_score: float | None = None
    _legs: set[str] = field(default_factory=set)


class HybridSearcher:
    """Runs both legs and fuses them. Owns no SQL; the repository does."""

    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder | None = None,
        *,
        rrf_k: int = RRF_K,
    ) -> None:
        self._chunks = DocumentChunkRepository(session)
        self._embedder = embedder or get_embedder()
        self._rrf_k = rrf_k

    async def search(
        self,
        query: str,
        *,
        limit: int = DEFAULT_LIMIT,
        document_id: uuid.UUID | None = None,
    ) -> list[SearchHit]:
        """Fused results, best first."""
        depth = limit * LEG_OVERSAMPLE
        vector_hits = await self.search_vector(query, limit=depth, document_id=document_id)
        fts_hits = await self.search_fts(query, limit=depth, document_id=document_id)

        fused = self._fuse(vector_hits, fts_hits)
        logger.info(
            "hybrid search",
            extra={
                "query_chars": len(query),
                "vector_hits": len(vector_hits),
                "fts_hits": len(fts_hits),
                "fused": len(fused),
            },
        )
        return fused[:limit]

    async def search_vector(
        self,
        query: str,
        *,
        limit: int = DEFAULT_LIMIT,
        document_id: uuid.UUID | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        vectors = await self._embedder.embed([query])
        hits = list(
            await self._chunks.search_by_vector(
                vectors[0],
                limit=limit,
                document_id=document_id,
                embedding_model=self._embedder.model_id,
            )
        )
        if not hits:
            await self._explain_an_empty_vector_leg()
        return hits

    async def _explain_an_empty_vector_leg(self) -> None:
        """Say why, when the vector leg matched nothing at all.

        `search_by_vector` filters to chunks embedded by the *same* model,
        because comparing vectors from two models is meaningless. That filter
        is right, and its failure mode is silence: point a deployment with a
        real key at a corpus indexed by the placeholder and every query quietly
        becomes keyword-only, with results that look like ordinary bad results.

        Only reached when the leg is empty, so the count runs once per failed
        query rather than on the hot path.
        """
        indexed = await self._chunks.embedding_models()
        others = sorted(model for model in indexed if model != self._embedder.model_id)
        if not others:
            return  # An empty or unindexed corpus, which is not a mismatch.
        logger.warning(
            "the vector leg matched nothing: the corpus was indexed by another model",
            extra={
                "querying_with": self._embedder.model_id,
                "corpus_indexed_by": others,
                "remedy": "re-index with `opuscovintel index` so both agree",
            },
        )

    async def search_fts(
        self,
        query: str,
        *,
        limit: int = DEFAULT_LIMIT,
        document_id: uuid.UUID | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        return list(await self._chunks.search_by_fts(query, limit=limit, document_id=document_id))

    def _fuse(
        self,
        vector_hits: Sequence[tuple[DocumentChunk, float]],
        fts_hits: Sequence[tuple[DocumentChunk, float]],
    ) -> list[SearchHit]:
        merged: dict[uuid.UUID, _Accumulator] = {}

        for position, (chunk, score) in enumerate(vector_hits, start=1):
            entry = merged.setdefault(chunk.id, _Accumulator(chunk=chunk))
            entry.vector_rank = position
            entry.vector_score = score
            entry.score += 1.0 / (self._rrf_k + position)

        for position, (chunk, score) in enumerate(fts_hits, start=1):
            entry = merged.setdefault(chunk.id, _Accumulator(chunk=chunk))
            entry.fts_rank = position
            entry.fts_score = score
            entry.score += 1.0 / (self._rrf_k + position)

        hits = [
            SearchHit(
                chunk=entry.chunk,
                score=entry.score,
                vector_rank=entry.vector_rank,
                fts_rank=entry.fts_rank,
                vector_score=entry.vector_score,
                fts_score=entry.fts_score,
            )
            for entry in merged.values()
        ]
        # Ties broken by page then ordinal so results are stable across runs --
        # a flapping result order makes an eval score meaningless.
        hits.sort(key=lambda hit: (-hit.score, hit.chunk.page_number, hit.chunk.ordinal))
        return hits


def reciprocal_rank_score(rank: int, *, k: int = RRF_K) -> float:
    """The RRF contribution of a single leg placing something at `rank`."""
    if rank < 1:
        raise ValueError("rank is 1-based")
    return 1.0 / (k + rank)
