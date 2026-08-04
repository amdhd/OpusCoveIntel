"""UUIDv7 generation (RFC 9562).

CLAUDE.md 7 mandates UUIDv7 primary keys. Version 7 embeds a millisecond
timestamp in its high bits, so ids sort chronologically -- B-tree inserts stay
at the right edge of the index instead of scattering across it the way UUIDv4
does, and `ORDER BY id` is a usable creation order.

`uuid.uuid7` only arrived in the Python 3.14 stdlib; we target 3.12, so this is
a local implementation rather than a dependency.

Layout (128 bits):
    48  unix_ts_ms
     4  version (0b0111)
    12  rand_a    -- used here as a monotonic counter within a millisecond
     2  variant (0b10)
    62  rand_b
"""

from __future__ import annotations

import os
import threading
import time
from uuid import UUID

_VERSION = 0x7
_VARIANT = 0b10
_COUNTER_BITS = 12
_COUNTER_MAX = (1 << _COUNTER_BITS) - 1

_lock = threading.Lock()
_last_ms = -1
_counter = 0


def uuid7() -> UUID:
    """Return a time-ordered UUIDv7.

    Monotonic even within a single millisecond: `rand_a` doubles as a counter,
    so two ids minted in the same millisecond still compare in creation order.
    On counter overflow (>4096 ids in one millisecond) the timestamp is borrowed
    forward by 1ms rather than allowing a collision or a backwards sort.
    """
    global _last_ms, _counter

    with _lock:
        now_ms = time.time_ns() // 1_000_000

        if now_ms > _last_ms:
            _last_ms = now_ms
            # Seed randomly so concurrent processes don't march in lockstep,
            # leaving headroom before overflow.
            _counter = int.from_bytes(os.urandom(2), "big") & 0x0FF
        else:
            _counter += 1
            if _counter > _COUNTER_MAX:
                _last_ms += 1
                _counter = 0
            now_ms = _last_ms

        ts = now_ms
        counter = _counter

    rand_b = int.from_bytes(os.urandom(8), "big") & ((1 << 62) - 1)

    value = (ts & ((1 << 48) - 1)) << 80 | _VERSION << 76 | counter << 64 | _VARIANT << 62 | rand_b
    return UUID(int=value)


def uuid7_timestamp_ms(value: UUID) -> int:
    """Extract the embedded millisecond timestamp from a UUIDv7."""
    if value.version != 7:
        raise ValueError(f"not a UUIDv7: version={value.version}")
    return value.int >> 80
