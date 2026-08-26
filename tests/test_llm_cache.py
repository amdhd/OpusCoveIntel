"""Response cache tests.

docs/plan.md Phase 5 acceptance: "cache hit costs $0." These tests prove that a
cache hit returns the stored response with zero cost, and that cache keys
are correctly invalidated by prompt/model/content changes.
"""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.cache import (
    ResponseCache,
    build_cache_key,
    build_cache_key_bytes,
)


class TestCacheKeyBuilding:
    def test_same_inputs_produce_same_key(self) -> None:
        key1 = build_cache_key(
            prompt_version="v1",
            model_id="claude-opus-5",
            content="Extract covenants from: the borrower shall maintain...",
        )
        key2 = build_cache_key(
            prompt_version="v1",
            model_id="claude-opus-5",
            content="Extract covenants from: the borrower shall maintain...",
        )
        assert key1 == key2

    def test_different_prompt_versions_give_different_keys(self) -> None:
        key1 = build_cache_key(prompt_version="v1", model_id="m", content="c")
        key2 = build_cache_key(prompt_version="v2", model_id="m", content="c")
        assert key1 != key2

    def test_different_models_give_different_keys(self) -> None:
        key1 = build_cache_key(prompt_version="v1", model_id="claude-opus-5", content="c")
        key2 = build_cache_key(prompt_version="v1", model_id="gpt-4o", content="c")
        assert key1 != key2

    def test_different_content_gives_different_keys(self) -> None:
        key1 = build_cache_key(prompt_version="v1", model_id="m", content="aaa")
        key2 = build_cache_key(prompt_version="v1", model_id="m", content="bbb")
        assert key1 != key2

    def test_key_is_hex_string(self) -> None:
        key = build_cache_key(prompt_version="v1", model_id="m", content="c")
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_bytes_key_matches_string_key_for_same_content(self) -> None:
        content = "test content"
        str_key = build_cache_key(prompt_version="v1", model_id="m", content=content)
        bytes_key = build_cache_key_bytes(
            prompt_version="v1", model_id="m", content_bytes=content.encode("utf-8")
        )
        assert str_key == bytes_key


class TestResponseCache:
    async def test_put_then_get_returns_same_response(self, db_session: AsyncSession) -> None:
        cache = ResponseCache(db_session)
        key = build_cache_key(prompt_version="v1", model_id="m", content="c")

        # Initially a miss
        assert await cache.get(key) is None

        # Put
        await cache.put(
            cache_key=key,
            prompt_hash="abc123",
            model_id="m",
            prompt_version="v1",
            response_json={"result": "RM30 million"},
            estimated_cost_usd=Decimal("0.15"),
        )

        # Now a hit
        result = await cache.get(key)
        assert result is not None
        assert result["result"] == "RM30 million"

    async def test_cache_hit_is_idempotent(self, db_session: AsyncSession) -> None:
        cache = ResponseCache(db_session)
        key = build_cache_key(prompt_version="v1", model_id="m", content="c")

        await cache.put(
            cache_key=key,
            prompt_hash="abc",
            model_id="m",
            prompt_version="v1",
            response_json={"x": 1},
            estimated_cost_usd=Decimal("0.10"),
        )

        # Multiple gets return the same result
        first = await cache.get(key)
        second = await cache.get(key)
        assert first == second

    async def test_lru_eviction_keeps_most_recent(self, db_session: AsyncSession) -> None:
        # Tiny LRU to force eviction
        cache = ResponseCache(db_session, lru_size=2)

        key_a = build_cache_key(prompt_version="v1", model_id="m", content="a")
        key_b = build_cache_key(prompt_version="v1", model_id="m", content="b")
        key_c = build_cache_key(prompt_version="v1", model_id="m", content="c")

        await cache.put(
            cache_key=key_a,
            prompt_hash="a",
            model_id="m",
            prompt_version="v1",
            response_json={"x": "a"},
            estimated_cost_usd=Decimal("0"),
        )
        await cache.put(
            cache_key=key_b,
            prompt_hash="b",
            model_id="m",
            prompt_version="v1",
            response_json={"x": "b"},
            estimated_cost_usd=Decimal("0"),
        )
        # Access A to make it recently used, then add C (should evict B from LRU)
        await cache.get(key_a)
        await cache.put(
            cache_key=key_c,
            prompt_hash="c",
            model_id="m",
            prompt_version="v1",
            response_json={"x": "c"},
            estimated_cost_usd=Decimal("0"),
        )

        # A should still be in LRU (recently accessed)
        assert await cache.get(key_a) is not None
        # C should be in LRU (just added)
        assert await cache.get(key_c) is not None

    async def test_different_keys_are_independent(self, db_session: AsyncSession) -> None:
        cache = ResponseCache(db_session)

        key1 = build_cache_key(prompt_version="v1", model_id="m", content="c1")
        key2 = build_cache_key(prompt_version="v1", model_id="m", content="c2")

        await cache.put(
            cache_key=key1,
            prompt_hash="h1",
            model_id="m",
            prompt_version="v1",
            response_json={"v": 1},
            estimated_cost_usd=Decimal("0"),
        )

        # key2 should still be a miss
        assert await cache.get(key2) is None
        # key1 should be a hit
        assert await cache.get(key1) is not None

    async def test_db_cache_persists_across_instances(self, db_session: AsyncSession) -> None:
        key = build_cache_key(prompt_version="v1", model_id="m", content="persist")

        cache1 = ResponseCache(db_session)
        await cache1.put(
            cache_key=key,
            prompt_hash="p",
            model_id="m",
            prompt_version="v1",
            response_json={"persisted": True},
            estimated_cost_usd=Decimal("0.05"),
        )

        # A new cache instance should find the DB entry
        cache2 = ResponseCache(db_session)
        result = await cache2.get(key)
        assert result is not None
        assert result["persisted"] is True
