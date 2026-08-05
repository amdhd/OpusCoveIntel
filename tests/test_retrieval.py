"""Indexing and hybrid retrieval.

The headline test is `test_hybrid_beats_either_leg_alone` -- PLAN.md's Phase 4
acceptance criterion. It is measured with MRR over the retrieval golden set
rather than asserted per query, because on any single query one leg will
sometimes win; the claim is about the aggregate.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.documents import DocumentChunkRepository
from app.evals.golden import RETRIEVAL_CASES
from app.llm.embeddings import HashingEmbedder
from app.retrieval.hybrid import RRF_K, HybridSearcher, reciprocal_rank_score
from app.retrieval.indexing import IndexingService

pytestmark = pytest.mark.usefixtures("storage_root")


def reciprocal_rank(texts: list[str], needle: str) -> float:
    for position, text in enumerate(texts, start=1):
        if needle.lower() in text.lower():
            return 1.0 / position
    return 0.0


# -- indexing --------------------------------------------------------------


async def test_indexing_populates_both_retrieval_columns(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    chunks = await DocumentChunkRepository(db_session).list_for_document(indexed_corpus[0])

    assert chunks
    assert all(chunk.embedding is not None for chunk in chunks)
    assert all(chunk.fts is not None for chunk in chunks)
    assert all(chunk.embedding_model == "hashing-bow-v1" for chunk in chunks)


async def test_reindexing_is_a_no_op(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    outcome = await IndexingService(db_session).index_document(indexed_corpus[0])

    # CLAUDE.md 1.7: same identity, no work, no spend.
    assert outcome.skipped is True


async def test_indexing_an_unknown_document_raises(db_session: AsyncSession) -> None:
    with pytest.raises(LookupError):
        await IndexingService(db_session).index_document(uuid.uuid4())


async def test_changing_the_embedding_model_forces_a_reindex(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    class OtherModel(HashingEmbedder):
        @property
        def model_id(self) -> str:
            return "some-other-model-v9"

    outcome = await IndexingService(db_session, OtherModel()).index_document(indexed_corpus[0])

    # The model is part of the job identity: vectors from two models are not
    # comparable, so switching must re-embed rather than silently mix spaces.
    assert outcome.skipped is False
    assert outcome.chunks_embedded > 0


# -- the two legs ----------------------------------------------------------


async def test_the_keyword_leg_finds_an_exact_figure(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    hits = await HybridSearcher(db_session).search_fts("RM30,000,000", limit=5)

    assert any("RM30,000,000" in chunk.chunk_text for chunk, _ in hits)


async def test_the_keyword_leg_does_not_require_every_term_to_match(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    # The clause never uses the word "threshold"; an AND-ed tsquery would
    # return nothing at all here.
    hits = await HybridSearcher(db_session).search_fts(
        "cross default threshold above RM30 million", limit=5
    )

    assert any("RM30,000,000" in chunk.chunk_text for chunk, _ in hits)


async def test_the_keyword_leg_searches_malay_under_its_own_configuration(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    hits = await HybridSearcher(db_session).search_fts("nisbah gearan", limit=5)

    assert any("nisbah gearan" in chunk.chunk_text.lower() for chunk, _ in hits)


async def test_the_vector_leg_returns_similarity_ordered_hits(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    hits = await HybridSearcher(db_session).search_vector("gearing ratio covenant", limit=5)

    assert hits
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


async def test_a_query_of_only_punctuation_returns_nothing_rather_than_erroring(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    # An invalid tsquery raises rather than returning nothing, which would be a
    # 500 on a user typing "???".
    assert await HybridSearcher(db_session).search_fts("???", limit=5) == []


# -- fusion ----------------------------------------------------------------


async def test_a_chunk_found_by_both_legs_outranks_one_found_by_either(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    hits = await HybridSearcher(db_session).search("cross default RM30,000,000", limit=10)

    both = [hit for hit in hits if hit.found_by_both]
    assert both
    # Agreement between independent retrieval strategies is evidence, and the
    # ranking has to reflect that.
    assert hits[0].found_by_both


def test_reciprocal_rank_scores_damp_the_top_rank() -> None:
    first = reciprocal_rank_score(1)
    second = reciprocal_rank_score(2)

    assert first > second
    # With k=60 the gap is small, so one confident leg cannot dominate.
    assert first / second < 1.05
    assert first == pytest.approx(1.0 / (RRF_K + 1))


def test_reciprocal_rank_rejects_a_zero_based_rank() -> None:
    with pytest.raises(ValueError, match="1-based"):
        reciprocal_rank_score(0)


async def test_results_are_stable_across_identical_searches(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    searcher = HybridSearcher(db_session)

    first = [hit.chunk.id for hit in await searcher.search("gearing covenant", limit=5)]
    second = [hit.chunk.id for hit in await searcher.search("gearing covenant", limit=5)]

    # A flapping result order makes an eval score meaningless.
    assert first == second


# -- the acceptance criterion ----------------------------------------------


async def test_hybrid_beats_either_leg_alone(
    db_session: AsyncSession, indexed_corpus: list[uuid.UUID]
) -> None:
    """PLAN.md, Phase 4 acceptance.

    Measured as mean reciprocal rank over the retrieval golden set. Hybrid wins
    where the legs disagree -- a paraphrased query the keyword leg ranks low,
    an exact figure the vector leg ranks low -- and RRF promotes what both
    found. The margin is modest here because the stand-in embedder is lexical,
    so the two legs are more correlated than they will be once Phase 5 swaps in
    real embeddings.
    """
    searcher = HybridSearcher(db_session)
    totals = {"vector": 0.0, "fts": 0.0, "hybrid": 0.0}

    for case in RETRIEVAL_CASES:
        vector_hits = await searcher.search_vector(case.query, limit=10)
        vector = [chunk.chunk_text for chunk, _ in vector_hits]
        fts = [chunk.chunk_text for chunk, _ in await searcher.search_fts(case.query, limit=10)]
        hybrid = [hit.chunk.chunk_text for hit in await searcher.search(case.query, limit=10)]

        totals["vector"] += reciprocal_rank(vector, case.expected_substring)
        totals["fts"] += reciprocal_rank(fts, case.expected_substring)
        totals["hybrid"] += reciprocal_rank(hybrid, case.expected_substring)

    count = len(RETRIEVAL_CASES)
    mrr = {leg: total / count for leg, total in totals.items()}

    assert mrr["hybrid"] > mrr["vector"], mrr
    assert mrr["hybrid"] > mrr["fts"], mrr
