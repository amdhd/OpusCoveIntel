"""The local object store, and the seam that will become S3 in Phase 8."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingest.storage import (
    LocalFileStore,
    ObjectNotFoundError,
    ObjectStore,
    ObjectStoreError,
)

KEY = "documents/ab/cd/abcd.pdf"


@pytest.fixture
def store(tmp_path: Path) -> LocalFileStore:
    return LocalFileStore(tmp_path / "storage")


async def test_put_then_get_round_trips(store: LocalFileStore) -> None:
    uri = await store.put(KEY, b"%PDF-1.7 body", content_type="application/pdf")

    assert await store.get(KEY) == b"%PDF-1.7 body"
    assert uri.startswith("file://")


async def test_uri_for_predicts_what_put_returns(store: LocalFileStore) -> None:
    predicted = store.uri_for(KEY)

    assert await store.put(KEY, b"%PDF-") == predicted


async def test_put_creates_intermediate_directories(store: LocalFileStore) -> None:
    await store.put(KEY, b"%PDF-")

    assert (store.root / KEY).is_file()


async def test_overwriting_a_key_is_allowed_and_atomic(store: LocalFileStore) -> None:
    await store.put(KEY, b"first")
    await store.put(KEY, b"second")

    assert await store.get(KEY) == b"second"
    # The temp file used for the atomic replace is not left behind.
    assert list((store.root / "documents/ab/cd").glob("*.tmp")) == []


async def test_missing_object_raises_a_storage_error(store: LocalFileStore) -> None:
    with pytest.raises(ObjectNotFoundError):
        await store.get("documents/00/00/nothing.pdf")


@pytest.mark.parametrize(
    "key",
    [
        "../escape.pdf",
        "documents/../../escape.pdf",
        "/absolute.pdf",
        "documents/nul\x00.pdf",
        "",
    ],
)
async def test_keys_cannot_escape_the_storage_root(store: LocalFileStore, key: str) -> None:
    with pytest.raises(ObjectStoreError):
        await store.put(key, b"%PDF-")


def test_local_store_satisfies_the_storage_protocol(store: LocalFileStore) -> None:
    # The Phase 8 swap to S3 is only cheap while callers depend on this and
    # nothing wider.
    assert isinstance(store, ObjectStore)
