"""Candidate detection — narrows a document to the spans worth LLM attention.

CLAUDE.md S2, cost lever #1: "Candidate narrowing - never send a whole document
to Opus. ~20x saving."

Three legs: regex over `app/extract/patterns.py`, Postgres FTS, and pgvector kNN,
the last two against the clause-type exemplars in `app/extract/exemplars.py`.
This is what docs/plan.md 2 specifies, and for a long time only the first of the three
was built.

**That mattered more than "an enhancement is pending" suggested.** The LLM
extractor sees only what this module returns, so while regex was the sole leg,
Opus could not find a covenant the regex had already missed — the billable path
was capped by the free path it exists to outperform. The eval harness measured
exactly that: rule recall 0.98, LLM recall 0.70, on a corpus whose fixtures were
written to suit the regexes. The gap was the ceiling, not the model.

Regex candidates are widened spans — the hit plus surrounding context. Semantic
candidates are whole chunks, because that is the unit both retrieval legs score;
narrowing one further would invent a coordinate the retrieval never produced.
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
from app.extract.exemplars import exemplar_queries
from app.extract.patterns import ALL_PATTERNS
from app.llm.embeddings import Embedder, get_embedder


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

# Semantic candidates are capped independently of the regex leg. Every one is an
# LLM call, and the regex leg is the precise one -- so a document whose prose
# merely resembles an exemplar must not be able to double its own extraction
# cost. Deliberately much smaller than MAX_CANDIDATES.
MAX_SEMANTIC_CANDIDATES: Final[int] = 25

# Chunks each exemplar retrieves, per leg. Small: an exemplar that matches
# thirty chunks in one document has told us the document is about covenants,
# not which chunk holds one.
EXEMPLAR_HITS: Final[int] = 3

# Relevance floors, without which the semantic legs are a spending bug.
#
# FTS and kNN are *ranked* retrieval: they return their best N whether or not
# anything is relevant. Unfloored, a Ministry of Finance press release holding
# no covenant language at all produced nine semantic candidates -- nine Opus
# calls, about $0.09, on a document the regex leg had correctly passed over. The
# legs do not decide relevance; a floor does.
#
# Calibrated by measuring both legs across every exemplar on a document that
# holds covenants and one that does not:
#
#     leg   with covenants   without      floor
#     fts   max 2.70         max 0.90     1.00
#     knn   max 0.81         max 0.34     0.40
#
# **Tuned on synthetic fixtures, so treat them as provisional.** The same
# caveat the eval harness prints about its own numbers applies here: these
# separate the two synthetic documents cleanly, which is not evidence that they
# separate a real prospectus from a real press release. Re-measure when licensed
# documents arrive — the script that produced the table above is three lines of
# `search_by_fts` / `search_by_vector` over `exemplar_queries()`.
MIN_FTS_RANK: Final[float] = 1.0
MIN_KNN_SIMILARITY: Final[float] = 0.40

# The semantic legs are **off by default**, on the evidence rather than on
# principle. Measured across the labelled corpus, regex candidates already
# contain **15 of 15** labelled covenants: the LLM was never short of spans, so
# these legs had no recall left to recover. Switched on they contributed one
# candidate — a "PARTIES TO THE TRANSACTION" boilerplate section matched by the
# Shariah exemplar because it names a Shariah adviser — which is one paid call
# and no covenant.
#
# They are kept, not reverted, because that measurement says more about the
# fixtures than about the legs. docs/plan.md 9 is explicit that the synthetic corpus
# was written to suit the regexes and that real prospectuses will not be; the
# case where regex coverage is 15/15 is exactly the case where an additional
# leg cannot show a benefit. Turn them on and re-measure the day a licensed
# document lands, which is the first honest test of them.
#
# What the LLM's recall gap actually was: `LLMCovenantExtraction` returns one
# covenant per span while a span routinely holds two or three. See
# `app/extract/schemas.py`.
SEMANTIC_LEGS_DEFAULT: Final[bool] = False


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
    # Which detection legs found this span. Provenance rather than decoration:
    # it is what lets the eval harness report whether the semantic legs are
    # earning the LLM calls they add.
    found_by: frozenset[str] = field(default_factory=lambda: frozenset({"regex"}))

    def __repr__(self) -> str:
        text_preview = self.text[:80].replace("\n", " ")
        return (
            f"Candidate(chunk={self.chunk_id}, page={self.page_number}, "
            f"span=({self.char_start}, {self.char_end}), hints={set(self.clause_type_hints)}, "
            f'text="{text_preview}...")'
        )


class CandidateDetectionService:
    """Find spans of a document that are likely to contain covenants.

    Three legs, and they fail differently:

    * **regex** — precise and narrow. `patterns.py` is written to fire only on
      explicit covenant language, so it misses every paraphrase.
    * **FTS** — a chunk scored on how many exemplar terms it carries, and how
      densely. Catches "undertakes not to encumber" where the negative-pledge
      regex wanted "create or permit to subsist".
    * **kNN** — nearest chunks to an exemplar in vector space.

    **Why the other two legs matter more than they look.** The LLM extractor
    only ever sees spans this service returns. While regex was the only leg, an
    Opus call could never find a covenant the regex had already missed — the
    expensive path was capped by the free one it exists to outperform, and the
    eval harness measured that ceiling as poor LLM recall. Cost lever #1 in
    CLAUDE.md 2 is candidate narrowing; narrowing on one signal alone narrows
    away real covenants.

    **The cost this adds is real and is capped.** Every extra candidate is an
    LLM call. The semantic legs are bounded by `MAX_SEMANTIC_CANDIDATES`
    independently of the regex leg, so a document cannot quietly become twice
    as expensive because its prose happens to resemble an exemplar.

    **Known limit of the kNN leg today.** Without `QWEN_API_KEY` the embedder is
    `HashingEmbedder`, which is lexical — it has no notion that "leverage" and
    "gearing" are related. The leg still contributes (bag-of-words overlap is a
    different signal from regex phrase structure) but it is not yet semantic.
    Real embeddings are the change that makes it so; nothing here needs to.
    """

    def __init__(self, session: AsyncSession, embedder: Embedder | None = None) -> None:
        self._session = session
        self._chunks = DocumentChunkRepository(session)
        self._embedder = embedder or get_embedder()

    async def detect(
        self,
        document_id: uuid.UUID,
        *,
        limit: int = MAX_CANDIDATES,
        semantic: bool = SEMANTIC_LEGS_DEFAULT,
        semantic_limit: int = MAX_SEMANTIC_CANDIDATES,
    ) -> list[Candidate]:
        """Return candidate spans for a document, sorted by page then position.

        The caller is responsible for cost-limiting the number actually sent to
        the LLM — this returns everything the legs found, up to the caps.

        `semantic=True` adds the FTS and kNN legs. Off by default — see
        `SEMANTIC_LEGS_DEFAULT` for the measurement that decided that.
        """
        chunks = await self._chunks.list_for_document(document_id, limit=100_000)
        if not chunks:
            logger.warning(
                "no chunks for document; nothing to detect",
                extra={"document_id": str(document_id)},
            )
            return []

        candidates = _detect_regex(chunks, limit=limit)
        regex_count = len(candidates)

        semantic_count = 0
        if semantic:
            found = await self._detect_semantic(document_id, chunks, limit=semantic_limit)
            # Only chunks the regex leg did not already cover. A chunk both legs
            # find is one candidate, not two -- the point is reach, not volume.
            covered = {candidate.chunk_id for candidate in candidates}
            extra = [item for item in found if item.chunk_id not in covered]
            semantic_count = len(extra)
            candidates = (candidates + extra)[:limit]

        candidates.sort(key=lambda item: (item.page_number, item.char_start))

        logger.info(
            "candidate detection complete",
            extra={
                "document_id": str(document_id),
                "chunks": len(chunks),
                "candidates": len(candidates),
                "from_regex": regex_count,
                "from_semantic": semantic_count,
            },
        )
        return candidates

    async def _detect_semantic(
        self,
        document_id: uuid.UUID,
        chunks: Sequence[_HasChunkText],
        *,
        limit: int,
    ) -> list[Candidate]:
        """Chunks that FTS or kNN associate with a clause-type exemplar.

        Whole chunks, not spans: both legs score a chunk, and inventing a
        narrower span inside one would be a coordinate the retrieval never
        actually produced (CLAUDE.md 1.2).
        """
        by_id = {chunk.id: chunk for chunk in chunks}
        hints: dict[uuid.UUID, set[ClauseType]] = {}
        legs: dict[uuid.UUID, set[str]] = {}
        # Best score per chunk, so the cap keeps the strongest matches rather
        # than whichever exemplar happened to run first.
        score: dict[uuid.UUID, float] = {}

        for clause_type, query in exemplar_queries():
            for chunk_id, leg, value in await self._search_both_legs(document_id, query):
                if chunk_id not in by_id:
                    continue
                hints.setdefault(chunk_id, set()).add(clause_type)
                legs.setdefault(chunk_id, set()).add(leg)
                score[chunk_id] = max(score.get(chunk_id, 0.0), value)

        ranked = sorted(score, key=lambda cid: -score[cid])[:limit]

        candidates: list[Candidate] = []
        for chunk_id in ranked:
            chunk = by_id[chunk_id]
            text = chunk.chunk_text
            if not text.strip():
                continue
            candidates.append(
                Candidate(
                    chunk_id=chunk_id,
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    clause_type_hints=frozenset(hints[chunk_id]),
                    page_number=chunk.page_number,
                    section_title=chunk.section_title,
                    found_by=frozenset(legs[chunk_id]),
                )
            )
        return candidates

    async def _search_both_legs(
        self, document_id: uuid.UUID, query: str
    ) -> list[tuple[uuid.UUID, str, float]]:
        """`(chunk_id, leg, score)` for one exemplar, from FTS and kNN.

        Failures are logged and swallowed per leg: candidate detection is an
        optimisation, and a document with no embeddings yet should fall back to
        the regex leg rather than fail extraction outright.
        """
        results: list[tuple[uuid.UUID, str, float]] = []

        try:
            for chunk, value in await self._chunks.search_by_fts(
                query, limit=EXEMPLAR_HITS, document_id=document_id
            ):
                if float(value) >= MIN_FTS_RANK:
                    results.append((chunk.id, "fts", float(value)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("candidate fts leg failed", extra={"error": str(exc)})

        try:
            vectors = await self._embedder.embed([query])
            for chunk, value in await self._chunks.search_by_vector(
                vectors[0],
                limit=EXEMPLAR_HITS,
                document_id=document_id,
                embedding_model=self._embedder.model_id,
            ):
                if float(value) >= MIN_KNN_SIMILARITY:
                    results.append((chunk.id, "knn", float(value)))
        except Exception as exc:  # noqa: BLE001
            logger.warning("candidate knn leg failed", extra={"error": str(exc)})

        return results


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
