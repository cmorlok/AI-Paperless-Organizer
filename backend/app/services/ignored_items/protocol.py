"""Protocol for ignored items service."""

from typing import Optional, Protocol, runtime_checkable, List


@runtime_checkable
class IgnoredItemsService(Protocol):
    """Protocol for ignored items service."""

    async def list_ignored(
        self,
        entity_type: Optional[str] = None,
        analysis_type: Optional[str] = None,
    ) -> list: ...

    async def add_ignored(
        self,
        item_id: int,
        item_name: str,
        entity_type: str,
        analysis_type: str,
        reason: str = "",
    ) -> dict: ...

    async def remove_ignored(self, item_id: int) -> None: ...

    async def check_if_ignored(
        self,
        entity_type: str,
        analysis_type: str,
        paperless_item_id: int,
    ) -> dict: ...

    async def get_ignored_ids(
        self,
        entity_type: str,
        analysis_type: str,
    ) -> List[int]: ...
