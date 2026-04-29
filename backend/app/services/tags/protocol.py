"""Protocol for tags business-logic service."""

from typing import Protocol, runtime_checkable, Dict, List, Optional


@runtime_checkable
class TagsService(Protocol):
    """Protocol for tags business-logic service."""

    async def estimate_tags(
        self,
        analysis_type: str = "nonsense",
    ) -> Dict: ...

    async def delete_empty_tags(self) -> Dict: ...

    async def bulk_delete_tags(self, tag_ids: List[int]) -> Dict: ...

    async def remove_tags_from_saved_analyses(self, tag_ids: List[int]) -> Dict: ...

    async def save_analysis(
        self,
        entity_type: str,
        analysis_type: str,
        groups: list,
        stats: dict,
        items_count_key: str = "items_count",
    ) -> None: ...
