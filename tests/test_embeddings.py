"""The offline embedder.

Its job is to make the vector leg of hybrid search real without spending
anything. These tests pin the properties retrieval depends on -- determinism,
dimension, normalisation, and lexical similarity actually ordering the way
similarity should -- and one test pins the *limit*, so nobody mistakes it for
a semantic model before Phase 5 swaps in Qwen.
"""

from __future__ import annotations

import math

import pytest

from app.llm.embeddings import Embedder, HashingEmbedder, get_embedder

CLAUSE = (
    "The Issuer shall maintain a consolidated gearing ratio of not more than "
    "1.75 times, tested semi-annually."
)


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dimension=1024)


async def test_the_same_text_always_embeds_identically(embedder: HashingEmbedder) -> None:
    first = await embedder.embed([CLAUSE])
    second = await embedder.embed([CLAUSE])

    # blake2b rather than hash(): Python's string hash is randomised per
    # process, so this would differ between the API and the worker.
    assert first == second


async def test_vectors_match_the_configured_dimension(embedder: HashingEmbedder) -> None:
    vectors = await embedder.embed([CLAUSE, "short"])

    assert embedder.dimension == 1024
    assert all(len(vector) == 1024 for vector in vectors)


async def test_a_batch_returns_one_vector_per_input_in_order(
    embedder: HashingEmbedder,
) -> None:
    vectors = await embedder.embed(["alpha", "beta", "gamma"])

    assert len(vectors) == 3
    assert vectors[0] != vectors[1] != vectors[2]


def test_vectors_are_unit_length_so_cosine_is_meaningful(embedder: HashingEmbedder) -> None:
    norm = math.sqrt(sum(value * value for value in embedder.embed_one(CLAUSE)))

    assert norm == pytest.approx(1.0)


def test_empty_text_gives_a_zero_vector_rather_than_an_invented_direction(
    embedder: HashingEmbedder,
) -> None:
    assert set(embedder.embed_one("   ")) == {0.0}


def test_lexical_overlap_orders_similarity_correctly(embedder: HashingEmbedder) -> None:
    query = embedder.embed_one("gearing ratio covenant")
    related = embedder.embed_one(CLAUSE)
    unrelated = embedder.embed_one(
        "The Shariah adviser has confirmed the structure complies with Shariah principles."
    )

    def cosine(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert cosine(query, related) > cosine(query, unrelated)


def test_it_models_lexical_overlap_and_not_meaning(embedder: HashingEmbedder) -> None:
    """The documented limitation, pinned so it is not mistaken for a feature.

    "leverage" and "gearing" mean the same thing to a credit analyst and
    nothing to a hashing vectoriser. Phase 5 closes this gap; until then, the
    FTS leg carries queries that restate the document's own words.
    """
    gearing = embedder.embed_one("gearing")
    leverage = embedder.embed_one("leverage")

    assert sum(x * y for x, y in zip(gearing, leverage, strict=True)) == 0.0


def test_repeated_terms_are_damped_sub_linearly(embedder: HashingEmbedder) -> None:
    once = embedder.embed_one("sukuk")
    many = embedder.embed_one("sukuk sukuk sukuk sukuk sukuk sukuk sukuk sukuk")

    # A clause repeating "sukuk" eight times is about a sukuk, not eight times
    # more about one -- and after normalisation both are the same direction.
    assert once == pytest.approx(many)


def test_the_default_embedder_satisfies_the_protocol() -> None:
    assert isinstance(get_embedder(), Embedder)


def test_the_model_id_is_recorded_so_vector_spaces_are_never_mixed(
    embedder: HashingEmbedder,
) -> None:
    # Chunks embedded by different models are not comparable; search filters on
    # this, which only works because indexing writes it.
    assert embedder.model_id == "hashing-bow-v1"
