"""UUIDv7 tests.

The point of v7 over v4 is time-ordering. If sortability or monotonicity break,
index locality quietly degrades and `ORDER BY id` silently stops meaning
"creation order" -- neither of which surfaces as an error, so both are tested.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.domain.ids import uuid7, uuid7_timestamp_ms


def test_version_and_variant_bits() -> None:
    value = uuid7()
    assert value.version == 7
    # RFC 9562 variant is 0b10 in the two most significant bits of octet 8.
    assert (value.int >> 62) & 0b11 == 0b10


def test_ids_are_unique() -> None:
    ids = {uuid7() for _ in range(10_000)}
    assert len(ids) == 10_000


def test_ids_sort_in_creation_order() -> None:
    ids = [uuid7() for _ in range(1_000)]
    assert ids == sorted(ids)


def test_monotonic_within_a_single_millisecond() -> None:
    """Generated in a tight loop, so most of these share a timestamp."""
    ids = [uuid7() for _ in range(5_000)]
    timestamps = {uuid7_timestamp_ms(i) for i in ids}
    assert len(timestamps) < len(ids), "test is only meaningful if ms collide"
    assert ids == sorted(ids)


def test_embedded_timestamp_tracks_wall_clock() -> None:
    before = time.time_ns() // 1_000_000
    value = uuid7()
    after = time.time_ns() // 1_000_000
    assert before <= uuid7_timestamp_ms(value) <= after + 1


def test_thread_safety() -> None:
    """The counter is shared mutable state; concurrent minting must not collide."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        batches = list(pool.map(lambda _: [uuid7() for _ in range(500)], range(8)))
    all_ids = [i for batch in batches for i in batch]
    assert len(set(all_ids)) == len(all_ids)


def test_timestamp_extraction_rejects_other_versions() -> None:
    import uuid as uuid_module

    with pytest.raises(ValueError, match="not a UUIDv7"):
        uuid7_timestamp_ms(uuid_module.uuid4())
