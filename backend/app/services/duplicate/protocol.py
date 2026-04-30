"""Protocol for duplicate detection service."""

from typing import Protocol, runtime_checkable, List


@runtime_checkable
class DuplicateService(Protocol):
    """Protocol for duplicate detection service."""

    async def scan_all(self, modes: List[str], similarity_threshold: float = 0.92) -> None: ...

    async def ignore_group(self, doc_ids: List[int]) -> dict: ...

    async def list_ignored(self) -> list: ...

    async def remove_ignore(self, doc_id_a: int, doc_id_b: int) -> None: ...