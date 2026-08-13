"""The embedding seam, and an offline embedder that costs nothing.

Phase 5 swaps in Qwen `text-embedding-v4`. Until then retrieval has to work,
be testable, and cost $0 -- so this ships the interface plus a deterministic
local implementation behind it.

**`HashingEmbedder` is not a stub that returns noise.** A random vector would
make the vector leg of hybrid search pure decoration, and "hybrid beats either
leg alone" would be measuring nothing. It is a hashing bag-of-words vectoriser:
tokens are hashed into dimensions, weighted by sub-linear term frequency, and
the vector is L2-normalised so cosine similarity is a real lexical similarity.

What it therefore does *not* have is semantics. "gearing" and "leverage" are
unrelated to it, and that is the gap Qwen closes in Phase 5. Stating the limit
matters: this embedder makes the plumbing honest, not the retrieval smart.

Vectors are `settings.VECTOR_DIMENSION` wide (1024) so the column, the HNSW
index and the Phase 5 provider all agree without a migration.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Final, Protocol, runtime_checkable

from app.core.config import get_settings
from app.core.logging import get_logger

TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-z]+(?:[.\-'][0-9a-z]+)*")

# Model identifier recorded on `document_chunks.embedding_model`. Versioned so
# a re-embed is detectable: chunks embedded by different models must never be
# compared, and this is what makes that checkable in SQL.
FAKE_EMBEDDING_MODEL: Final[str] = "hashing-bow-v1"

logger = get_logger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """What retrieval needs from an embedding provider, and nothing more."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch. Order of the result matches order of the input."""
        ...


class HashingEmbedder:
    """Deterministic, offline, free. See the module docstring for its limits."""

    def __init__(self, dimension: int | None = None) -> None:
        self._dimension = dimension or get_settings().VECTOR_DIMENSION

    @property
    def model_id(self) -> str:
        return FAKE_EMBEDDING_MODEL

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        counts: dict[int, float] = {}
        for token in TOKEN_RE.findall(text.lower()):
            bucket = self._bucket(token)
            # Sub-linear term frequency: a clause repeating "sukuk" eight times
            # is about a sukuk, not eight times more about one.
            counts[bucket] = counts.get(bucket, 0.0) + 1.0

        vector = [0.0] * self._dimension
        for bucket, count in counts.items():
            vector[bucket] = 1.0 + math.log(count)

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # An empty or punctuation-only chunk. Returning zeros keeps cosine
            # distance defined (pgvector treats it as maximally distant) rather
            # than inventing a direction for text that has none.
            return vector
        return [value / norm for value in vector]

    def _bucket(self, token: str) -> int:
        # blake2b rather than hash(): Python's string hash is randomised per
        # process, which would make embeddings differ between the API and the
        # worker -- and silently, since both would still look like vectors.
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dimension


class QwenEmbedder:
    """Real Qwen text-embedding-v4 via the OpenAI-compatible DashScope endpoint.

    Implements the Embedder protocol so it is a drop-in replacement for
    HashingEmbedder. Batches requests (EMBED_BATCH_SIZE at a time) because
    Qwen bills per API call.

    Semantics: this closes the gap documented on HashingEmbedder — "gearing"
    and "leverage" are now related. Multilingual (EN + BM), 1024 dims.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._model_id = settings.EMBEDDING_MODEL
        self._dimension = settings.VECTOR_DIMENSION
        # Lazy-init on first call so imports don't fail when keys are absent
        # and the mock is in use.
        self._adapter: object | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if self._adapter is None:
            from app.llm.adapters.qwen import QwenAdapter

            self._adapter = QwenAdapter()
        adapter: QwenAdapter = self._adapter  # type: ignore[assignment]
        return await adapter.embed(texts, model_id=self._model_id)


def get_embedder() -> Embedder:
    """The embedder for this deployment.

    Returns a QwenEmbedder when EMBEDDING_MODEL is set to a real provider and
    QWEN_API_KEY is configured; falls back to HashingEmbedder otherwise (CI,
    dev without keys, $0 demos).

    Phase 5: the single call site that changed. IndexingService and
    HybridSearcher consume this and don't know which implementation they got.

    **The fallback is announced.** `HashingEmbedder` is a bag of words with a
    hash for a vector: it makes the vector leg of hybrid retrieval run, and it
    makes it meaningless -- "Dubai negative pledge" returned a page of legal
    advisers and two director biographies above the negative-pledge clause,
    because only the keyword leg was doing real work. That is survivable in CI
    and for a $0 demo, and it is not something a deployment should discover
    from its search results (docs/review.md, finding 11).
    """
    settings = get_settings()

    # When testing or running without keys, use the free offline embedder.
    # This also preserves the "make test makes zero paid API calls" invariant.
    if settings.ENVIRONMENT == "test":
        return HashingEmbedder()

    if settings.QWEN_API_KEY is None:
        _warn_placeholder("QWEN_API_KEY is not set")
        return HashingEmbedder()

    # Check if EMBEDDING_MODEL names a real provider rather than the fake one.
    if settings.EMBEDDING_MODEL in ("text-embedding-v4", "text-embedding-v3"):
        return QwenEmbedder()

    _warn_placeholder(f"EMBEDDING_MODEL={settings.EMBEDDING_MODEL!r} names no real provider")
    return HashingEmbedder()


# Said once per process. Every indexed chunk and every query would otherwise
# repeat it, and a warning that appears ten thousand times is one nobody reads.
_placeholder_announced = False


def _warn_placeholder(reason: str) -> None:
    global _placeholder_announced
    if _placeholder_announced:
        return
    _placeholder_announced = True
    logger.warning(
        "semantic search is disabled: retrieval is running on the placeholder embedder",
        extra={
            "reason": reason,
            "embedder": FAKE_EMBEDDING_MODEL,
            "consequence": (
                "the vector leg of hybrid retrieval carries no semantic signal; "
                "only keyword matching is answering"
            ),
        },
    )
