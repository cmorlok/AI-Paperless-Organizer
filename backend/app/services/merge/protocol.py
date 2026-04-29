"""Protocol for merge service."""

from typing import Protocol, runtime_checkable, List, Dict


@runtime_checkable
class MergeService(Protocol):
    """Protocol for merge service."""

    async def merge_correspondents(self, target_id: int, target_name: str, source_ids: List[int]) -> Dict: ...

    async def merge_tags(self, target_id: int, target_name: str, source_ids: List[int]) -> Dict: ...

    async def merge_document_types(self, target_id: int, target_name: str, source_ids: List[int]) -> Dict: ...

    async def get_history(self, entity_type: str | None = None) -> List[Dict]: ...