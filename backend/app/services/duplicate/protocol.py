"""Protocol for duplicate detection service."""

from typing import Protocol, runtime_checkable, List


@runtime_checkable
class DuplicateService(Protocol):
    """Protocol for duplicate detection service."""

    async def scan_all(self, modes: List[str], similarity_threshold: float = 0.92) -> None: ...