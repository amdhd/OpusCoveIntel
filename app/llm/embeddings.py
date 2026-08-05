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

TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[0-9a-z]+(?:[.\-'][0-9a-z]+)*")

# Model identifier recorded on `document_chunks.embedding_model`. Versioned so
# a re-embed is detectable: chunks embedded by different models must never be
# compared, and this is what makes that checkable in SQL.
FAKE_EMBEDDING_MODEL: Final[str] = "hashing-bow-v1"


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


def get_embedder() -> Embedder:
    """The embedder for this deployment.

    One call site to change in Phase 5, when `EMBEDDING_MODEL` starts naming a
    real provider and this returns a budget-guarded adapter instead.
    """
    return HashingEmbedder()
