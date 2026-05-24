"""
StorageBackend Protocol — object storage for artifacts (STEP, STL, NPZ, etc.).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for object storage backends (MinIO, S3, local FS)."""

    name: str

    async def put(self, key: str, data: bytes, metadata: dict[str, str] | None = None) -> str:
        """Store an object and return its URI.

        Args:
            key: Object key / path.
            data: Binary content.
            metadata: Optional key-value tags.

        Returns:
            URI (e.g. s3://bucket/key or file:///path).
        """
        ...

    async def get(self, key: str) -> bytes:
        """Retrieve object bytes by key.

        Args:
            key: Object key / path.

        Returns:
            Raw bytes.
        """
        ...

    async def delete(self, key: str) -> None:
        """Delete an object."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if an object exists."""
        ...

    async def list(self, prefix: str) -> list[str]:
        """List keys with a given prefix."""
        ...

    async def health_check(self) -> bool:
        """Check storage backend connectivity."""
        ...
