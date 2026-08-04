"""Shared test fixtures.

CLAUDE.md 7: CI must never hit a paid API. `pytest_collection_modifyitems` below
enforces that structurally -- tests marked `live_llm` are skipped unless
RUN_LIVE_LLM_TESTS=1 is set explicitly.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

# Pin the environment before anything imports settings.
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("LOG_LEVEL", "WARNING")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip billable tests unless explicitly opted in."""
    if os.getenv("RUN_LIVE_LLM_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="billable; set RUN_LIVE_LLM_TESTS=1 to run")
    for item in items:
        if "live_llm" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """Drop cached settings/engines so env monkeypatching takes effect per test."""
    from app.core.config import get_settings
    from app.db.session import (
        get_engine,
        get_readonly_engine,
        get_readonly_sessionmaker,
        get_sessionmaker,
    )

    caches = (
        get_settings,
        get_engine,
        get_readonly_engine,
        get_sessionmaker,
        get_readonly_sessionmaker,
    )
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A TestClient over the real app. No database required."""
    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
