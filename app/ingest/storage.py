"""Object storage behind an S3-shaped interface.

Phase 3 writes to the local filesystem; Phase 8 swaps in S3/MinIO (CLAUDE.md 9
defers the infrastructure, not the seam). The interface is deliberately narrow
-- `put`, `get`, `uri_for` -- so the swap touches this module and nothing else.

Keys are POSIX-style relative paths (`documents/ab/cd/<sha256>.pdf`), which map
equally onto a directory tree and a bucket prefix. They are content-addressed,
so the same bytes always land in the same place.

The protocol is async even though the local backend is not: an S3 client will
be, and a synchronous seam would have to be redesigned rather than swapped.
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Keys are generated, never user-supplied, but validating here means a future
# caller cannot turn a filename into a path traversal.
_SAFE_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,511}$")


class ObjectStoreError(RuntimeError):
    """Storage failed. Callers surface this rather than a backend-specific error."""


class ObjectNotFoundError(ObjectStoreError):
    pass


@runtime_checkable
class ObjectStore(Protocol):
    """The whole storage contract. Resist widening it."""

    async def put(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> str:
        """Store `data` under `key` and return its URI."""
        ...

    async def get(self, key: str) -> bytes:
        """Return the bytes stored under `key`, or raise `ObjectNotFoundError`."""
        ...

    def uri_for(self, key: str) -> str:
        """Return the URI `put` would produce, without touching the backend."""
        ...


class LocalFileStore:
    """Filesystem-backed store rooted at `settings.STORAGE_DIR`.

    Writes are atomic (temp file plus `os.replace`) so a crashed upload cannot
    leave a truncated PDF that hashes to nothing. Blocking file I/O runs on a
    worker thread to keep the event loop free.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()

    @property
    def root(self) -> Path:
        return self._root

    def _path_for(self, key: str) -> Path:
        if not _SAFE_KEY.match(key) or ".." in key.split("/"):
            raise ObjectStoreError(f"unsafe storage key: {key!r}")
        path = (self._root / key).resolve()
        if not path.is_relative_to(self._root):
            raise ObjectStoreError(f"storage key escapes the root: {key!r}")
        return path

    def uri_for(self, key: str) -> str:
        return self._path_for(key).as_uri()

    async def put(
        self, key: str, data: bytes, *, content_type: str = "application/octet-stream"
    ) -> str:
        # `content_type` is part of the S3 contract and ignored locally; it is
        # accepted here so callers do not change when the backend does.
        del content_type
        path = self._path_for(key)
        await asyncio.to_thread(self._write_atomic, path, data)
        logger.info("stored object", extra={"key": key, "bytes": len(data)})
        return path.as_uri()

    @staticmethod
    def _write_atomic(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    async def get(self, key: str) -> bytes:
        path = self._path_for(key)
        try:
            return await asyncio.to_thread(path.read_bytes)
        except FileNotFoundError as exc:
            raise ObjectNotFoundError(f"no object at {key!r}") from exc


@lru_cache(maxsize=1)
def get_object_store() -> ObjectStore:
    """Process-wide store. Also usable directly as a FastAPI dependency."""
    return LocalFileStore(get_settings().STORAGE_DIR)
