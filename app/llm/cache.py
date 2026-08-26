"""Response cache — keyed on sha256(prompt_version | model_id | content).

docs/plan.md 2, cost reducer #3: re-running an unchanged pipeline costs $0. Two tiers:

1. **In-process LRU** — sub-ms lookup for repeated calls within a single run
   (same chunk, same prompt). Avoids the DB round-trip for the common case.
2. **Database** (`llm_cache` table) — survives process restarts. The cost
   ledger already has the `cache_hit_is_free` CHECK constraint.

The cache key is `sha256(prompt_version | model_id | content_hash)`, which
means a prompt change, a model change, or a content change all invalidate
the cached response. This is deliberate: comparing vectors from two different
models is meaningless, and reusing a response against a different prompt is a
correctness bug, not an optimisation.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from decimal import Decimal
from typing import TYPE_CHECKING

from sqlalchemy.exc import IntegrityError

from app.core.logging import get_logger

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


logger = get_logger(__name__)

# In-process cache: default 256 entries, LRU eviction.
_DEFAULT_LRU_SIZE = 256


class ResponseCache:
    """Two-tier cache: in-process LRU + database.

    Usage:
        cache = ResponseCache(session)
        entry = await cache.get(cache_key)
        if entry is not None:
            return entry["response_json"], True  # cache hit, $0
        # ... make the LLM call ...
        await cache.put(cache_key, response_json, estimated_cost)
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        lru_size: int = _DEFAULT_LRU_SIZE,
    ) -> None:
        self._session = session
        self._lru_size = lru_size
        self._lru: OrderedDict[str, dict[str, object]] = OrderedDict()

    async def get(self, cache_key: str) -> dict[str, object] | None:
        """Look up a cached response. In-process first, then DB.

        Returns the parsed response JSON on hit, or None on miss.
        """
        # Tier 1: in-process LRU
        if cache_key in self._lru:
            self._lru.move_to_end(cache_key)
            logger.debug("cache.hit.lru", extra={"cache_key": cache_key[:16]})
            return self._lru[cache_key]

        # Tier 2: database
        from app.db.repositories.ops import LLMCacheRepository

        repo = LLMCacheRepository(self._session)
        row = await repo.get_by_key(cache_key)
        if row is None:
            return None

        await repo.record_hit(row)
        logger.debug(
            "cache.hit.db",
            extra={"cache_key": cache_key[:16], "hit_count": row.hit_count},
        )

        # Promote to LRU
        response: dict[str, object] = row.response_json
        self._lru[cache_key] = response
        if len(self._lru) > self._lru_size:
            self._lru.popitem(last=False)

        return response

    async def put(
        self,
        cache_key: str,
        prompt_hash: str,
        model_id: str,
        prompt_version: str,
        response_json: dict[str, object],
        estimated_cost_usd: Decimal,
    ) -> None:
        """Store a response in both tiers.

        The DB row includes the cost avoided, for cache-savings reporting.
        """
        from app.db.models.ops import LLMCache
        from app.db.repositories.ops import LLMCacheRepository

        # Tier 1: in-process LRU
        self._lru[cache_key] = response_json
        if len(self._lru) > self._lru_size:
            self._lru.popitem(last=False)

        # Tier 2: database. A SELECT-then-INSERT is not atomic, so two workers
        # racing on the same key would have the loser raise IntegrityError --
        # which in SQLAlchemy poisons the *whole* surrounding transaction, not
        # just this write. The nested SAVEPOINT confines the failure: losing
        # the race is a no-op, and the caller's extraction still commits.
        repo = LLMCacheRepository(self._session)
        existing = await repo.get_by_key(cache_key)
        if existing is not None:
            return

        entry = LLMCache(
            cache_key=cache_key,
            prompt_hash=prompt_hash,
            model_id=model_id,
            prompt_version=prompt_version,
            response_json=response_json,
            estimated_cost_usd=estimated_cost_usd,
            hit_count=1,
        )
        try:
            async with self._session.begin_nested():
                await repo.add(entry)
        except IntegrityError:
            logger.debug("cache.store_lost_race", extra={"cache_key": cache_key[:16]})
            return

        logger.debug(
            "cache.store",
            extra={
                "cache_key": cache_key[:16],
                "model_id": model_id,
                "prompt_version": prompt_version,
            },
        )


def build_cache_key(
    *,
    prompt_version: str,
    model_id: str,
    content: str,
) -> str:
    """Produce a deterministic cache key from the three identity components.

    The key is the hex digest of sha256(prompt_version | model_id | content).
    Changing any one component invalidates the cache entry.
    """
    material = f"{prompt_version}|{model_id}|{content}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_cache_key_bytes(
    *,
    prompt_version: str,
    model_id: str,
    content_bytes: bytes,
) -> str:
    """As above, but the content is already bytes (e.g. an image for VLM)."""
    prefix = f"{prompt_version}|{model_id}|".encode()
    digest = hashlib.sha256(prefix + content_bytes).hexdigest()
    return digest
